"""
Tests para scripts/build_dataset.py.

Cubre:
- Normalización del hash (dedup robusto a whitespace y case)
- Dedup por target
- Formato del user prompt (header 1:1 vs grupal, contexto, marcador final)
- Frase de participantes (1, 2, 3+ personas)
- format_example con tokenizer mock (retorna texto y descarta si supera MAX_TOKEN_LEN)
"""
from unittest.mock import MagicMock


def _sample_pair_one_on_one() -> dict:
    return {
        "chat_name": "Mi Amiga",
        "is_group": False,
        "participants": ["Mi Amiga"],
        "context": [
            {"author": "Mi Amiga", "text": "Hola!"},
            {"author": "Mi Amiga", "text": "Cómo va?"},
        ],
        "target": "Bien todo, vos qué onda?",
    }


def _sample_pair_group() -> dict:
    return {
        "chat_name": "Grupo X",
        "is_group": True,
        "participants": ["Delfi", "Luna"],
        "context": [
            {"author": "Delfi", "text": "Hoy comí lo más rico"},
            {"author": "Luna", "text": "pasa la receta!!"},
            {"author": "Delfi", "text": "te paso por insta"},
        ],
        "target": "yo tmb quiero!! mandame ahora",
    }


# ── Hash y dedup ────────────────────────────────────────────────────────────

def test_item_hash_normalizes_case_and_whitespace():
    from scripts.build_dataset import _item_hash
    assert _item_hash("Hola Mundo") == _item_hash("hola mundo")
    assert _item_hash("a  b   c") == _item_hash("a b c")
    assert _item_hash("  texto  ") == _item_hash("texto")


def test_item_hash_distinguishes_different_texts():
    from scripts.build_dataset import _item_hash
    texts = [f"Texto distinto número {i}" for i in range(10)]
    hashes = [_item_hash(t) for t in texts]
    assert len(set(hashes)) == len(hashes)


def test_dedup_by_target():
    from scripts.build_dataset import _dedup
    pairs = [
        {"target": "Mismo mensaje", "chat_name": "A"},
        {"target": "  mismo  MENSAJE  ", "chat_name": "B"},  # equivalente al anterior
        {"target": "Distinto", "chat_name": "C"},
    ]
    result = _dedup(pairs)
    assert len(result) == 2
    targets = {p["target"] for p in result}
    assert "Mismo mensaje" in targets
    assert "Distinto" in targets


# ── Frase de participantes ──────────────────────────────────────────────────

def test_participants_phrase_one_two_three():
    from scripts.build_dataset import _participants_phrase
    assert _participants_phrase([]) == ""
    assert _participants_phrase(["Ana"]) == "Ana"
    assert _participants_phrase(["Ana", "Beto"]) == "Ana y Beto"
    assert _participants_phrase(["Ana", "Beto", "Cami"]) == "Ana, Beto y Cami"
    assert _participants_phrase(["A", "B", "C", "D"]) == "A, B, C y D"


# ── Formato del user prompt ─────────────────────────────────────────────────

def test_user_prompt_one_on_one_header():
    from scripts.build_dataset import _format_user_prompt
    prompt = _format_user_prompt(_sample_pair_one_on_one())
    assert prompt.startswith("[Chat con Mi Amiga]")
    assert "[Tu próximo mensaje:]" in prompt


def test_user_prompt_group_header_lists_participants():
    from scripts.build_dataset import _format_user_prompt
    prompt = _format_user_prompt(_sample_pair_group())
    assert prompt.startswith("[Chat: Grupo X (con Delfi y Luna)]")
    assert "[Tu próximo mensaje:]" in prompt


def test_user_prompt_includes_authored_messages():
    from scripts.build_dataset import _format_user_prompt
    prompt = _format_user_prompt(_sample_pair_group())
    assert "Delfi: Hoy comí lo más rico" in prompt
    assert "Luna: pasa la receta!!" in prompt


def test_user_prompt_no_target_in_input():
    """El target NO debe filtrarse al user prompt."""
    from scripts.build_dataset import _format_user_prompt
    pair = _sample_pair_one_on_one()
    prompt = _format_user_prompt(pair)
    assert pair["target"] not in prompt


# ── format_example ──────────────────────────────────────────────────────────

def _mock_tokenizer(token_count: int = 100, eos: str = "<eos>"):
    tok = MagicMock()
    tok.apply_chat_template.return_value = "<bos>chat formateado<eos>"
    tok.eos_token = eos
    tok.encode.return_value = [0] * token_count
    return tok


def test_format_example_returns_expected_fields():
    from scripts.build_dataset import format_example
    tok = _mock_tokenizer(token_count=200)
    result = format_example(_sample_pair_group(), tok)
    assert result is not None
    assert set(result.keys()) == {"text", "chat_name", "is_group", "target"}
    assert result["chat_name"] == "Grupo X"
    assert result["is_group"] is True
    assert result["target"] == "yo tmb quiero!! mandame ahora"


def test_format_example_skips_oversized():
    from scripts.build_dataset import format_example, MAX_TOKEN_LEN
    tok = _mock_tokenizer(token_count=MAX_TOKEN_LEN + 1)
    result = format_example(_sample_pair_group(), tok)
    assert result is None


def test_format_example_passes_chat_to_tokenizer():
    """El chat template debe llamarse con user + assistant en ese orden."""
    from scripts.build_dataset import format_example
    tok = _mock_tokenizer(token_count=200)
    format_example(_sample_pair_one_on_one(), tok)
    args, kwargs = tok.apply_chat_template.call_args
    chat = args[0]
    assert len(chat) == 2
    assert chat[0]["role"] == "user"
    assert chat[1]["role"] == "assistant"
    assert chat[1]["content"] == "Bien todo, vos qué onda?"
