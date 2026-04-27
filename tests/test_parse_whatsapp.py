"""
Tests para scripts/parse_whatsapp.py (parser conversacional).

Cubre extracción de pares (contexto, target):
- Estructura del par y campos requeridos
- Primer turno del usuario sin par (no tiene contexto previo)
- Turnos consecutivos del usuario unidos en un único target
- Filtro de min_target_chars
- Segmentación por gap_hours
- Mensajes de sistema filtrados
- Casefold del autor
- Nombre del chat desde el archivo
- Detección de grupo y participantes
- Encoding (BOM, LRM/RLM)
"""
from pathlib import Path
from unittest.mock import patch


def _no_anon():
    """Patch del anonimizador para que no toque el texto en tests."""
    return patch("scripts.parse_whatsapp.anonymize", side_effect=lambda x: x)


def _write(tmp_path: Path, content: str, name: str = "WhatsApp Chat - Amiga.txt") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ── Estructura del par ──────────────────────────────────────────────────────

def test_pair_has_expected_fields(tmp_path):
    content = (
        "1/3/24, 10:30 - Amiga: Hola, cómo andás? Te quería contar algo.\n"
        "1/3/24, 10:31 - Nico: Bien todo, contame qué pasó este finde por allá.\n"
    )
    p = _write(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico")
    assert len(pairs) == 1
    pair = pairs[0]
    assert set(pair.keys()) == {"chat_name", "is_group", "participants", "context", "target"}
    assert pair["chat_name"] == "Amiga"
    assert pair["is_group"] is False
    assert pair["participants"] == ["Amiga"]
    assert pair["context"] == [{"author": "Amiga", "text": "Hola, cómo andás? Te quería contar algo."}]
    assert "Bien todo" in pair["target"]


# ── Filtros ─────────────────────────────────────────────────────────────────

def test_first_turn_user_no_pair(tmp_path):
    """Si el primer turno de la conversación es del usuario, no genera par."""
    content = (
        "1/3/24, 10:30 - Nico: Soy el primero en hablar acá. Largo suficiente para target.\n"
        "1/3/24, 10:31 - Amiga: Respuesta corta.\n"
    )
    p = _write(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico")
    assert pairs == []


def test_consecutive_user_messages_join_in_target(tmp_path):
    """Mensajes consecutivos del usuario se unen como un solo target."""
    content = (
        "1/3/24, 10:30 - Amiga: Hola, te escribo para preguntarte algo importante.\n"
        "1/3/24, 10:31 - Nico: Hola.\n"
        "1/3/24, 10:32 - Nico: Decime tranqui qué onda.\n"
        "1/3/24, 10:33 - Nico: Acá estoy disponible para charlar.\n"
    )
    p = _write(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico")
    assert len(pairs) == 1
    target = pairs[0]["target"]
    assert "Hola." in target
    assert "Decime tranqui" in target
    assert "disponible para charlar" in target


def test_min_target_chars_filters_short_replies(tmp_path):
    """Targets muy cortos se descartan pero siguen sirviendo como contexto."""
    content = (
        "1/3/24, 10:30 - Amiga: Te quería preguntar si estás libre el jueves a la noche.\n"
        "1/3/24, 10:31 - Nico: Dale.\n"
        "1/3/24, 10:32 - Amiga: Te paso la dirección entonces.\n"
        "1/3/24, 10:33 - Nico: Buenísimo, ahí confirmo más cerca de la fecha.\n"
    )
    p = _write(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico", min_target_chars=30)
    # "Dale." se descarta como target pero aparece como contexto del siguiente par
    assert len(pairs) == 1
    context_texts = [c["text"] for c in pairs[0]["context"]]
    assert "Dale." in context_texts


# ── Segmentación por gap ────────────────────────────────────────────────────

def test_gap_hours_splits_conversations(tmp_path):
    """Gap > gap_hours cierra la conversación; el siguiente turno del usuario
    en la nueva conversación es 'primer turno' y no genera par."""
    content = (
        "1/3/24, 10:30 - Amiga: Mensaje inicial de la primera conversación del día.\n"
        "1/3/24, 10:31 - Nico: Respuesta dentro de la primera conversación, suficientemente larga.\n"
        # Gap de 8 hs → segunda conversación
        "1/3/24, 18:31 - Nico: Primer mensaje de la segunda conversación, no tiene par.\n"
    )
    p = _write(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico", gap_hours=6.0)
    # Solo el de la primera conversación cuenta
    assert len(pairs) == 1
    assert "Respuesta dentro de la primera" in pairs[0]["target"]


# ── Mensajes de sistema ─────────────────────────────────────────────────────

def test_system_messages_filtered(tmp_path):
    """Mensajes de sistema no aparecen como contexto y no rompen turnos."""
    content = (
        "1/3/24, 10:30 - Amiga: Te paso el video que te decía recién.\n"
        "1/3/24, 10:31 - Amiga: <Multimedia omitido>\n"
        "1/3/24, 10:32 - Nico: Buenísimo, ahora lo miro y te digo qué me pareció.\n"
    )
    p = _write(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico")
    assert len(pairs) == 1
    context_texts = [c["text"] for c in pairs[0]["context"]]
    assert all("Multimedia" not in t for t in context_texts)
    # Aunque <Multimedia omitido> está en el medio, los dos mensajes reales
    # de Amiga quedan como un único turno "lógico" (mismo autor, sin
    # interrupción real).
    assert len(context_texts) == 1


# ── Detección de autor ──────────────────────────────────────────────────────

def test_author_casefold(tmp_path):
    """NICO en mayúsculas debe matchear cuando se pasa 'nico' como autor."""
    content = (
        "1/3/24, 10:30 - Amiga: Te tiro una pregunta para ver qué opinás vos.\n"
        "1/3/24, 10:31 - NICO: Mi opinión es que está todo bien y deberías hacerlo.\n"
    )
    p = _write(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "nico")
    assert len(pairs) == 1


# ── Metadata del chat ───────────────────────────────────────────────────────

def test_chat_name_from_filename(tmp_path):
    p = _write(tmp_path, "1/3/24, 10:30 - Amiga: hola\n", name="WhatsApp Chat - Foo Bar.txt")
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico")
    # No hay pares pero sí podemos chequear el nombre con un caso que sí genera
    # — hacemos un test más completo aquí.
    content = (
        "1/3/24, 10:30 - Amiga: Algo para arrancar la conversación con suficiente texto.\n"
        "1/3/24, 10:31 - Nico: Respuesta del usuario que es lo suficientemente larga como target.\n"
    )
    p2 = _write(tmp_path, content, name="WhatsApp Chat - Mi Amiga.txt")
    with _no_anon():
        pairs = parse_whatsapp(str(p2), "Nico")
    assert pairs[0]["chat_name"] == "Mi Amiga"


def test_is_group_detection_and_participants(tmp_path):
    """is_group=True con >2 autores; participants excluye al usuario."""
    content = (
        "1/3/24, 10:30 - Amiga1: Hola chicas qué onda esta semana cómo va todo.\n"
        "1/3/24, 10:31 - Amiga2: Acá complicada con trabajo pero nada raro la verdad.\n"
        "1/3/24, 10:32 - Nico: Yo bien, planeando hacer algo el finde si pueden sumarse.\n"
    )
    p = _write(tmp_path, content, name="WhatsApp Chat - Grupo X.txt")
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico")
    assert len(pairs) == 1
    assert pairs[0]["is_group"] is True
    assert sorted(pairs[0]["participants"]) == ["Amiga1", "Amiga2"]


def test_one_on_one_is_not_group(tmp_path):
    content = (
        "1/3/24, 10:30 - Amiga: Mensaje uno con texto suficiente para ser válido.\n"
        "1/3/24, 10:31 - Nico: Mensaje dos del usuario también con texto suficiente.\n"
    )
    p = _write(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico")
    assert pairs[0]["is_group"] is False
    assert pairs[0]["participants"] == ["Amiga"]


# ── Encoding ────────────────────────────────────────────────────────────────

def test_bom_encoding(tmp_path):
    content = (
        "﻿1/3/24, 10:30 a. m. - Amiga: Mensaje con BOM al principio del archivo de exportación.\n"
        "1/3/24, 10:31 a. m. - Nico: Respuesta posterior del usuario lo suficientemente larga acá.\n"
    )
    p = tmp_path / "WhatsApp Chat - Amiga.txt"
    p.write_bytes(content.encode("utf-8"))
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico")
    assert len(pairs) == 1


def test_lrm_rlm_stripped(tmp_path):
    content = (
        "\u200E1/3/24, 10:30 - Amiga: Mensaje con marca LRM al principio de la línea.\n"
        "1/3/24, 10:31 - Nico: Respuesta del usuario con texto suficientemente largo para target.\n"
    )
    p = _write(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico")
    assert len(pairs) == 1


# ── Multilinea ──────────────────────────────────────────────────────────────

def test_multiline_message(tmp_path):
    """Continuaciones multilinea deben unirse al mensaje en curso."""
    content = (
        "1/3/24, 10:30 - Amiga: Primera línea\n"
        "segunda línea sin timestamp\n"
        "tercera línea también continuación.\n"
        "1/3/24, 10:31 - Nico: Respuesta del usuario suficientemente larga para entrar como target.\n"
    )
    p = _write(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico")
    assert len(pairs) == 1
    ctx_text = pairs[0]["context"][0]["text"]
    assert "segunda línea" in ctx_text
    assert "tercera línea" in ctx_text


# ── context_size ────────────────────────────────────────────────────────────

def test_context_size_limits_history(tmp_path):
    """El contexto se trunca a los últimos context_size mensajes."""
    lines = []
    for i in range(10):
        lines.append(f"1/3/24, 10:{i:02d} - Amiga: Mensaje número {i} de Amiga acá.")
    lines.append("1/3/24, 10:30 - Nico: Respuesta final del usuario con suficiente texto para target.")
    content = "\n".join(lines) + "\n"
    p = _write(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with _no_anon():
        pairs = parse_whatsapp(str(p), "Nico", context_size=3)
    assert len(pairs[0]["context"]) == 3
    # Los últimos 3 antes del turno de Nico
    assert "Mensaje número 7" in pairs[0]["context"][0]["text"]
    assert "Mensaje número 9" in pairs[0]["context"][2]["text"]
