"""
Tests para scripts/build_dataset.py.
Cubre: no-leakage (prompts no contienen palabras del target),
formato messages de MLX-LM chat, split estratificado, dedup por hash, manifest escrito.
"""
import json
import random
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "prompts"
    d.mkdir(parents=True)
    (d / "casual.txt").write_text(
        "Contame algo de tu día\nQué hiciste hoy?\nTenés planes para el finde?",
        encoding="utf-8",
    )
    (d / "email_prof.txt").write_text(
        "Redactá un email de seguimiento de proyecto\nEscribí un resumen ejecutivo",
        encoding="utf-8",
    )
    (d / "academic.txt").write_text(
        "Escribí una introducción sobre la fotosíntesis\nAnalizá las causas de la Primera Guerra Mundial",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def system_prompts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "system_prompts"
    d.mkdir(parents=True)
    (d / "casual.txt").write_text("Sos Nico, escribís en tono coloquial argentino.", encoding="utf-8")
    (d / "email_prof.txt").write_text("Sos Nico Karagozian. Escribís emails profesionales.", encoding="utf-8")
    (d / "academic.txt").write_text("Sos el autor de un texto académico en español.", encoding="utf-8")
    return d


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "dataset"
    d.mkdir(parents=True)
    return d


def _make_items(register: str, n: int, prefix: str = "") -> list[dict]:
    return [
        {"text": f"{prefix}Texto ejemplo número {i} del registro {register} para testing.", "register": register}
        for i in range(n)
    ]


# ── Tests ────────────────────────────────────────────────────────────────────

def test_format_example_chat_format(tmp_path, prompts_dir, system_prompts_dir, monkeypatch):
    """format_example debe retornar un dict con campo 'messages' en formato MLX-LM chat."""
    import scripts.build_dataset as bd
    bd._prompts_cache.clear()
    bd._system_cache.clear()
    monkeypatch.chdir(tmp_path)

    rng = random.Random(42)
    item = {"text": "Hola, ¿qué hacés? Te paso el número de Mati.", "register": "casual"}
    result = bd.format_example(item, rng)

    assert result is not None
    assert "messages" in result
    assert result["register"] == "casual"
    messages = result["messages"]
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles
    assistant_content = next(m["content"] for m in messages if m["role"] == "assistant")
    assert assistant_content == item["text"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "[CASUAL]" in user_content


def test_no_leakage(tmp_path, prompts_dir, system_prompts_dir, monkeypatch):
    """Ningún prompt del catálogo debe contener las palabras del texto target."""
    import scripts.build_dataset as bd
    bd._prompts_cache.clear()
    bd._system_cache.clear()
    monkeypatch.chdir(tmp_path)

    rng = random.Random(42)
    item = {"text": "Texto de ejemplo único para el test de leakage.", "register": "casual"}

    prompt_path = prompts_dir / "casual.txt"
    prompts = [l.strip() for l in prompt_path.read_text().splitlines() if l.strip()]

    for prompt in prompts:
        prompt_words = set(prompt.lower().split())
        assert not prompt_words.issuperset({"texto", "ejemplo", "único", "leakage"}), \
            f"Prompt '{prompt}' contiene palabras del target"


def test_deduplication():
    """Textos duplicados (mismo hash normalizado) deben tener el mismo hash."""
    from scripts.build_dataset import _item_hash

    text_a = "Este es un mensaje de prueba para el test de deduplicación."
    text_b = "  Este   es  un  mensaje  de  prueba  para  el  test  de  deduplicación.  "

    assert _item_hash(text_a) == _item_hash(text_b), \
        "Textos equivalentes normalizados deben tener el mismo hash"


def test_item_hash_normalization():
    """_item_hash debe normalizar whitespace y case."""
    from scripts.build_dataset import _item_hash

    assert _item_hash("Hola Mundo") == _item_hash("hola mundo")
    assert _item_hash("a  b   c") == _item_hash("a b c")
    assert _item_hash("  texto  ") == _item_hash("texto")


def test_stratified_registers():
    """Textos distintos deben tener hashes distintos."""
    from scripts.build_dataset import _item_hash

    texts = [f"Texto número {i} completamente diferente." for i in range(10)]
    hashes = [_item_hash(t) for t in texts]
    assert len(set(hashes)) == len(hashes), "Textos distintos deben tener hashes distintos"


def test_manifest_written(tmp_path, prompts_dir, system_prompts_dir, monkeypatch):
    """build_dataset debe escribir manifest.json con campos requeridos."""
    import scripts.build_dataset as bd
    bd._prompts_cache.clear()
    bd._system_cache.clear()
    monkeypatch.chdir(tmp_path)

    casual_file = tmp_path / "casual.jsonl"
    email_file = tmp_path / "email.jsonl"
    academic_file = tmp_path / "academic.jsonl"
    dataset_path = tmp_path / "data" / "dataset"
    dataset_path.mkdir(parents=True)

    for fpath, register in [
        (casual_file, "casual"),
        (email_file, "email_prof"),
        (academic_file, "academic"),
    ]:
        with open(fpath, "w") as f:
            for i in range(20):
                f.write(json.dumps({"text": f"Texto {i} de {register} lo suficientemente largo.", "register": register}) + "\n")

    bd.build_dataset(
        casual_file=str(casual_file),
        email_file=str(email_file),
        academic_file=str(academic_file),
        output_dir=str(dataset_path),
        seed=42,
    )

    manifest_path = dataset_path / "manifest.json"
    assert manifest_path.exists(), "manifest.json debe existir"
    manifest = json.loads(manifest_path.read_text())
    assert "seed" in manifest
    assert "model_id" in manifest
    assert "mlx_format" in manifest
    assert manifest["mlx_format"] == "chat"
    assert manifest["mask_prompt"] is True
    assert "counts" in manifest
    assert "train_by_register" in manifest


def test_output_format_is_messages(tmp_path, prompts_dir, system_prompts_dir, monkeypatch):
    """Las líneas del train.jsonl deben tener campo 'messages', no 'text'."""
    import scripts.build_dataset as bd
    bd._prompts_cache.clear()
    bd._system_cache.clear()
    monkeypatch.chdir(tmp_path)

    casual_file = tmp_path / "casual.jsonl"
    email_file = tmp_path / "email.jsonl"
    academic_file = tmp_path / "academic.jsonl"
    dataset_path = tmp_path / "data" / "dataset"
    dataset_path.mkdir(parents=True)

    for fpath, register in [
        (casual_file, "casual"),
        (email_file, "email_prof"),
        (academic_file, "academic"),
    ]:
        with open(fpath, "w") as f:
            for i in range(20):
                f.write(json.dumps({"text": f"Texto {i} de {register} suficientemente largo.", "register": register}) + "\n")

    bd.build_dataset(
        casual_file=str(casual_file),
        email_file=str(email_file),
        academic_file=str(academic_file),
        output_dir=str(dataset_path),
        seed=42,
    )

    train_file = dataset_path / "train.jsonl"
    assert train_file.exists()
    with open(train_file) as f:
        first_line = json.loads(f.readline())

    assert "messages" in first_line, "Las líneas deben tener campo 'messages'"
    assert "text" not in first_line, "Las líneas NO deben tener campo 'text' (modo chat, no texto plano)"
    roles = [m["role"] for m in first_line["messages"]]
    assert "user" in roles
    assert "assistant" in roles
