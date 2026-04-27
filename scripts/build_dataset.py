"""
Combina pares conversacionales (output del parser de WhatsApp) en un dataset
de fine-tuning para Gemma 3, con split estratificado train/val/test.

Cada par se transforma a un mensaje de chat template:
  - user:      header del chat + contexto formateado + "[Tu próximo mensaje:]"
  - assistant: turno del usuario (target a aprender)

Ver docs/CHATBOT_DESIGN.md para las decisiones de diseño.
"""
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Optional

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_ID = "google/gemma-3-4b-it"
MAX_TOKEN_LEN = 2048

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        # Workaround bug transformers 4.57.6: tokenizer_config.json de Gemma 4 trae
        # extra_special_tokens como list ['<|video|>'] pero _set_model_specific_special_tokens
        # espera dict. Pasar {} sobrescribe el config. <|video|> es solo para multimodal.
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, extra_special_tokens={})
    return _tokenizer


def _participants_phrase(participants: list[str]) -> str:
    """'A', 'A y B', 'A, B y C', etc."""
    if not participants:
        return ""
    if len(participants) == 1:
        return participants[0]
    if len(participants) == 2:
        return f"{participants[0]} y {participants[1]}"
    return ", ".join(participants[:-1]) + f" y {participants[-1]}"


def _format_user_prompt(pair: dict) -> str:
    """
    Construye el contenido del 'user' del chat template a partir del par
    conversacional.

    Formato:

        [Chat: NombreGrupo (con A, B y C)]                    # si is_group
        [Chat con NombreContacto]                             # si 1:1

        A: ...
        B: ...
        AuthorName: ...
        ...

        [Tu próximo mensaje:]
    """
    if pair["is_group"]:
        header = f"[Chat: {pair['chat_name']} (con {_participants_phrase(pair['participants'])})]"
    else:
        header = f"[Chat con {pair['chat_name']}]"

    msg_lines = [f"{m['author']}: {m['text']}" for m in pair["context"]]
    return f"{header}\n\n" + "\n".join(msg_lines) + "\n\n[Tu próximo mensaje:]"


def format_example(pair: dict, tokenizer) -> Optional[dict]:
    """
    Convierte un par conversacional al formato de chat de Gemma 3.
    Retorna None si supera MAX_TOKEN_LEN.
    """
    user_content = _format_user_prompt(pair)
    target = pair["target"]

    chat = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target},
    ]
    formatted = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=False,
    )
    if tokenizer.eos_token and not formatted.endswith(tokenizer.eos_token):
        formatted = formatted + tokenizer.eos_token

    token_len = len(tokenizer.encode(formatted))
    if token_len > MAX_TOKEN_LEN:
        logger.debug(
            "Par descartado: %d tokens > %d (chat=%s)",
            token_len, MAX_TOKEN_LEN, pair.get("chat_name"),
        )
        return None

    return {
        "text": formatted,
        "chat_name": pair["chat_name"],
        "is_group": pair["is_group"],
        "target": target,
    }


def _item_hash(text: str) -> str:
    """Hash sha1 del texto normalizado (lower + strip + collapse whitespace)."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.sha1(normalized.encode()).hexdigest()


def _load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _save_jsonl(data: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _dedup(pairs: list[dict]) -> list[dict]:
    """Deduplica pares por hash del target normalizado."""
    seen: set[str] = set()
    result = []
    for p in pairs:
        h = _item_hash(p.get("target", ""))
        if h not in seen:
            seen.add(h)
            result.append(p)
    return result


def gather_pairs(raw_dir: str, processed_path: str, author_name: str,
                 context_size: int = 20, gap_hours: float = 6.0,
                 min_target_chars: int = 30) -> list[dict]:
    """
    Parsea todos los .txt de WhatsApp en `raw_dir` y guarda los pares
    unificados en `processed_path`. Retorna la lista combinada.
    """
    from scripts.parse_whatsapp import parse_whatsapp

    raw = Path(raw_dir)
    files = sorted(raw.glob("*.txt"))
    all_pairs: list[dict] = []
    for f in files:
        pairs = parse_whatsapp(
            str(f), author_name,
            context_size=context_size,
            gap_hours=gap_hours,
            min_target_chars=min_target_chars,
        )
        logger.info("%s: %d pares", f.name, len(pairs))
        all_pairs.extend(pairs)

    out_path = Path(processed_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_jsonl(all_pairs, out_path)
    logger.info("Total %d pares guardados en %s", len(all_pairs), out_path)
    return all_pairs


def build_dataset(
    pairs_file: str,
    output_dir: str = "data/dataset",
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_total: Optional[int] = None,
    seed: int = 42,
):
    from scripts.seed import set_all_seeds
    import random
    set_all_seeds(seed)
    rng = random.Random(seed)

    pairs = _load_jsonl(pairs_file)
    logger.info("Cargados %d pares", len(pairs))

    pairs = _dedup(pairs)
    logger.info("Post-dedup: %d pares", len(pairs))

    rng.shuffle(pairs)
    if max_total is not None:
        pairs = pairs[:max_total]

    tokenizer = _get_tokenizer()

    examples: list[dict] = []
    skipped_long = 0
    for p in pairs:
        try:
            ex = format_example(p, tokenizer)
            if ex is None:
                skipped_long += 1
            else:
                examples.append(ex)
        except Exception as e:
            logger.warning("Error formateando par: %s", e)

    if skipped_long:
        logger.info("Descartados %d pares por superar %d tokens", skipped_long, MAX_TOKEN_LEN)

    from sklearn.model_selection import train_test_split

    idx = list(range(len(examples)))
    strata = [ex["chat_name"] for ex in examples]
    counts = Counter(strata)

    if min(counts.values(), default=0) < 2:
        logger.warning("Algún chat tiene <2 ejemplos; split sin estratificar")
        idx_trainval, idx_test = train_test_split(idx, test_size=test_split, random_state=seed)
        effective_val = val_split / (1 - test_split)
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=effective_val, random_state=seed,
        )
    else:
        idx_trainval, idx_test = train_test_split(
            idx, test_size=test_split, stratify=strata, random_state=seed,
        )
        strata_tv = [strata[i] for i in idx_trainval]
        effective_val = val_split / (1 - test_split)
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=effective_val, stratify=strata_tv, random_state=seed,
        )

    train_set = [examples[i] for i in idx_train]
    val_set = [examples[i] for i in idx_val]
    test_set = [examples[i] for i in idx_test]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _save_jsonl(train_set, output_path / "train.jsonl")
    _save_jsonl(val_set,   output_path / "val.jsonl")
    _save_jsonl(val_set,   output_path / "valid.jsonl")  # mlx-lm usa valid.jsonl
    _save_jsonl(test_set,  output_path / "test.jsonl")

    import sklearn
    import transformers as _tf
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "max_total": max_total,
        "max_token_len": MAX_TOKEN_LEN,
        "sklearn_version": sklearn.__version__,
        "transformers_version": _tf.__version__,
        "counts": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "train_by_chat": dict(Counter(ex["chat_name"] for ex in train_set)),
        "test_by_chat": dict(Counter(ex["chat_name"] for ex in test_set)),
        "sha256": {
            "train": _sha256_file(output_path / "train.jsonl"),
            "val":   _sha256_file(output_path / "val.jsonl"),
            "test":  _sha256_file(output_path / "test.jsonl"),
        },
    }
    with open(output_path / "manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    print(f"\n✓ Dataset final:")
    print(f"  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print(f"  Distribución train por chat:")
    for chat, count in Counter(ex["chat_name"] for ex in train_set).most_common():
        print(f"    {chat}: {count}")

    return train_set, val_set, test_set


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    AUTHOR = os.environ.get("AUTHOR_NAME") or "Author"

    # Paso 1: parsear todos los chats y guardar pares unificados
    gather_pairs(
        raw_dir="data/raw/whatsapp",
        processed_path="data/processed/whatsapp_pairs.jsonl",
        author_name=AUTHOR,
    )

    # Paso 2: split, formato chat template y manifest
    build_dataset(pairs_file="data/processed/whatsapp_pairs.jsonl")
