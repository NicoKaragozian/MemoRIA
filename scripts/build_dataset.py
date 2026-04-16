"""
Combina los tres registros, formatea con el chat template real de Gemma 4,
balancea y divide en train/val/test (80/10/10).
"""
import json
import random
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-4-E2B-it"

INSTRUCTIONS_BY_REGISTER = {
    "casual": [
        "[CASUAL] Escribí un mensaje de WhatsApp: {topic}",
        "[CASUAL] Respondé de manera informal: {topic}",
        "[CASUAL] Mensaje casual de Nico: {topic}",
    ],
    "email_prof": [
        "[EMAIL-PROF] Redactá un email profesional sobre: {topic}",
        "[EMAIL-PROF] Escribí un correo formal: {topic}",
        "[EMAIL-PROF] Email profesional de Nico sobre: {topic}",
    ],
    "academic": [
        "[ACADÉMICO] Escribí un párrafo académico sobre: {topic}",
        "[ACADÉMICO] Redactá en estilo académico: {topic}",
        "[ACADÉMICO] Fragmento de paper o ensayo: {topic}",
    ],
}

# Cargado una vez — lazy
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return _tokenizer


def _load_prompts(register: str) -> list[str]:
    """Carga los prompts independientes del catálogo para este registro."""
    path = Path("data/prompts") / f"{register}.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _extract_topic(text: str) -> str:
    """
    Usa las primeras 3-4 palabras como topic para reducir leakage.
    El model aprende estilo, no continuación literal.
    """
    words = text.split()[:4]
    return " ".join(words) + "..."


def format_example(item: dict, tokenizer) -> dict:
    """
    Convierte un item raw al formato de chat de Gemma 4 usando apply_chat_template.
    Esto garantiza que el template en train sea byte-idéntico al de inferencia.
    """
    register = item["register"]
    text = item["text"]
    topic = _extract_topic(text)

    templates = INSTRUCTIONS_BY_REGISTER.get(register, [])
    instruction = random.choice(templates).format(topic=topic)

    chat = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": text},
    ]
    formatted = tokenizer.apply_chat_template(chat, tokenize=False)

    return {
        "text": formatted,
        "register": register,
        "original_text": text,
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


def build_dataset(
    casual_file: str,
    email_file: str,
    academic_file: str,
    output_dir: str = "data/dataset",
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_per_register: int = 1000,
    seed: int = 42,
):
    random.seed(seed)

    casual   = _load_jsonl(casual_file)
    email    = _load_jsonl(email_file)
    academic = _load_jsonl(academic_file)

    print(f"Raw — casual: {len(casual)}, email: {len(email)}, academic: {len(academic)}")

    random.shuffle(casual);   casual   = casual[:max_per_register]
    random.shuffle(email);    email    = email[:max_per_register]
    random.shuffle(academic); academic = academic[:max_per_register]

    tokenizer = _get_tokenizer()

    all_examples = []
    for item in casual + email + academic:
        try:
            all_examples.append(format_example(item, tokenizer))
        except Exception as e:
            print(f"  ⚠ Error formateando item: {e}")

    random.shuffle(all_examples)

    n = len(all_examples)
    n_val  = int(n * val_split)
    n_test = int(n * test_split)

    test_set  = all_examples[:n_test]
    val_set   = all_examples[n_test:n_test + n_val]
    train_set = all_examples[n_test + n_val:]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _save_jsonl(train_set, output_path / "train.jsonl")
    _save_jsonl(val_set,   output_path / "val.jsonl")
    _save_jsonl(val_set,   output_path / "valid.jsonl")   # mlx-lm usa valid.jsonl
    _save_jsonl(test_set,  output_path / "test.jsonl")

    reg_counts = Counter(ex["register"] for ex in train_set)
    print(f"\n✓ Dataset final:")
    print(f"  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print(f"  Distribución train: {dict(reg_counts)}")

    return train_set, val_set, test_set


if __name__ == "__main__":
    build_dataset(
        casual_file="data/processed/casual.jsonl",
        email_file="data/processed/email_prof.jsonl",
        academic_file="data/processed/academic.jsonl",
    )
