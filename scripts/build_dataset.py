"""
Combina los tres registros, construye examples en formato chat de MLX-LM
y divide en train/valid/test (80/10/10) con split estratificado.

Formato de salida por línea JSONL (compatible con mlx_lm.lora --data en modo chat):
  {"messages": [
      {"role": "system", "content": "<system prompt del registro>"},
      {"role": "user",   "content": "[REGISTRO] <prompt del catálogo>"},
      {"role": "assistant", "content": "<texto original del usuario>"}
  ], "register": "<registro>"}

MLX-LM aplica tokenizer.apply_chat_template() automáticamente en modo chat.
No pre-renderizamos el template aquí para garantizar que siempre coincide
con el template del modelo usado en el training.
"""
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# Límite conservador de tokens. Se usa estimación por chars para evitar
# cargar el tokenizer en sistemas sin HF login. ~3 chars/token es conservador
# para español (sobreestima → menos descartes falsos).
MAX_CHARS_APPROX = 3000  # ≈ 1000 tokens


def _load_prompts(register: str) -> list[str]:
    fname = "email_prof" if register == "email_prof" else register
    path = Path("data/prompts") / f"{fname}.txt"
    if not path.exists():
        logger.warning("Catálogo de prompts no encontrado: %s", path)
        return [f"Escribí un texto en registro {register}."]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_system_prompt(register: str) -> str:
    fname = "email_prof" if register == "email_prof" else register
    path = Path("data/system_prompts") / f"{fname}.txt"
    if not path.exists():
        logger.warning("System prompt no encontrado: %s", path)
        return ""
    return path.read_text(encoding="utf-8").strip()


_prompts_cache: dict[str, list[str]] = {}
_system_cache: dict[str, str] = {}


def _get_prompts(register: str) -> list[str]:
    if register not in _prompts_cache:
        _prompts_cache[register] = _load_prompts(register)
    return _prompts_cache[register]


def _get_system_prompt(register: str) -> str:
    if register not in _system_cache:
        _system_cache[register] = _load_system_prompt(register)
    return _system_cache[register]


def _item_hash(text: str) -> str:
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.sha1(normalized.encode()).hexdigest()


def format_example(item: dict, rng) -> Optional[dict]:
    """
    Convierte un item raw al formato messages de MLX-LM chat.
    El prompt se muestrea del catálogo — NO contiene palabras del target.
    No aplica chat template: MLX-LM lo hace automáticamente.
    """
    register = item["register"]
    text = item["text"]

    # Estimación de longitud para descartar ejemplos excesivamente largos
    if len(text) > MAX_CHARS_APPROX:
        logger.debug(
            "Ejemplo descartado: %d chars > %d (register=%s)", len(text), MAX_CHARS_APPROX, register
        )
        return None

    instruction = rng.choice(_get_prompts(register))
    system_prompt = _get_system_prompt(register)
    tag = register.replace("_", "-").upper()

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"[{tag}] {instruction}"})
    messages.append({"role": "assistant", "content": text})

    return {
        "messages": messages,
        "register": register,
    }


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


def _dedup(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for it in items:
        h = _item_hash(it.get("text", ""))
        if h not in seen:
            seen.add(h)
            result.append(it)
    return result


def build_dataset(
    casual_file: str,
    email_file: str,
    academic_file: str,
    output_dir: str = "data/dataset",
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_per_register: int = 800,
    seed: int = 42,
):
    from scripts.seed import set_all_seeds
    import random
    set_all_seeds(seed)
    rng = random.Random(seed)

    casual   = _load_jsonl(casual_file)
    email    = _load_jsonl(email_file)
    academic = _load_jsonl(academic_file)
    logger.info("Raw — casual: %d, email: %d, academic: %d", len(casual), len(email), len(academic))

    casual   = _dedup(casual)
    email    = _dedup(email)
    academic = _dedup(academic)
    logger.info("Post-dedup — casual: %d, email: %d, academic: %d", len(casual), len(email), len(academic))

    rng.shuffle(casual);   casual   = casual[:max_per_register]
    rng.shuffle(email);    email    = email[:max_per_register]
    rng.shuffle(academic); academic = academic[:max_per_register]

    all_examples = []
    skipped_long = 0
    for item in casual + email + academic:
        try:
            ex = format_example(item, rng)
            if ex is None:
                skipped_long += 1
            else:
                all_examples.append(ex)
        except Exception as e:
            logger.warning("Error formateando item: %s", e)

    if skipped_long:
        logger.info("Descartados %d ejemplos por superar %d chars", skipped_long, MAX_CHARS_APPROX)

    from sklearn.model_selection import train_test_split

    idx    = list(range(len(all_examples)))
    strata = [ex["register"] for ex in all_examples]

    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_split, stratify=strata, random_state=seed,
    )
    strata_tv = [strata[i] for i in idx_trainval]
    effective_val = val_split / (1 - test_split)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=effective_val, stratify=strata_tv, random_state=seed,
    )

    train_set = [all_examples[i] for i in idx_train]
    val_set   = [all_examples[i] for i in idx_val]
    test_set  = [all_examples[i] for i in idx_test]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _save_jsonl(train_set, output_path / "train.jsonl")
    _save_jsonl(val_set,   output_path / "valid.jsonl")   # mlx-lm usa valid.jsonl
    _save_jsonl(test_set,  output_path / "test.jsonl")

    import sklearn
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_id": MODEL_ID,
        "mlx_format": "chat",
        "mask_prompt": True,
        "seed": seed,
        "max_per_register": max_per_register,
        "max_chars_approx": MAX_CHARS_APPROX,
        "sklearn_version": sklearn.__version__,
        "counts": {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
        "train_by_register": dict(Counter(ex["register"] for ex in train_set)),
        "test_by_register":  dict(Counter(ex["register"] for ex in test_set)),
        "sha256": {
            "train": _sha256_file(output_path / "train.jsonl"),
            "valid": _sha256_file(output_path / "valid.jsonl"),
            "test":  _sha256_file(output_path / "test.jsonl"),
        },
        "prompts_sha1": {
            reg: hashlib.sha1("\n".join(_get_prompts(reg)).encode()).hexdigest()
            for reg in ("casual", "email_prof", "academic")
        },
    }
    with open(output_path / "manifest.json", "w") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    reg_counts = Counter(ex["register"] for ex in train_set)
    print(f"\n✓ Dataset final:")
    print(f"  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print(f"  Distribución train: {dict(reg_counts)}")

    return train_set, val_set, test_set


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_dataset(
        casual_file="data/processed/casual.jsonl",
        email_file="data/processed/email_prof.jsonl",
        academic_file="data/processed/academic.jsonl",
    )
