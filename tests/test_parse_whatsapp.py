"""
Tests para scripts/parse_whatsapp.py.
Cubre: iOS (BOM + narrow NBSP), Android (espacio normal), casefold del autor,
líneas de sistema ignoradas, mensajes multilinea.
"""
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _parse(filename: str, author: str = "Nico", min_len: int = 30) -> list[dict]:
    from scripts.parse_whatsapp import parse_whatsapp
    return parse_whatsapp(str(FIXTURES / filename), author, min_len=min_len)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _write_tmp(tmp_path: Path, content: str, name: str = "chat.txt") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ── Tests ────────────────────────────────────────────────────────────────────

def test_ios_export_parsed(tmp_path):
    content = (
        "1/3/24, 10:30\u202fa.\u202fm. - Nico: Hola, esto es un mensaje largo suficiente para superar el mínimo.\n"
        "1/3/24, 10:31\u202fa.\u202fm. - Amiga: Respuesta corta\n"
        "1/3/24, 10:32\u202fa.\u202fm. - Nico: Segundo mensaje del autor que también es suficientemente largo.\n"
    )
    p = _write_tmp(tmp_path, content)
    with patch("scripts.parse_whatsapp.anonymize", side_effect=lambda x: x):
        results = _parse.__wrapped__(str(p), "Nico") if hasattr(_parse, "__wrapped__") else []
        from scripts.parse_whatsapp import parse_whatsapp
        with patch("scripts.parse_whatsapp.anonymize", side_effect=lambda x: x):
            results = parse_whatsapp(str(p), "Nico")
    assert len(results) == 2
    assert all(r["register"] == "casual" for r in results)


def test_android_export_parsed(tmp_path):
    content = (
        "1/3/24, 10:30 - Nico: Mensaje Android con espacio normal en la hora del chat.\n"
        "1/3/24, 10:31 - Otro: Alguien más escribe algo aquí.\n"
        "1/3/24, 10:32 - Nico: Segundo mensaje de Android que tiene suficiente longitud.\n"
    )
    p = _write_tmp(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with patch("scripts.parse_whatsapp.anonymize", side_effect=lambda x: x):
        results = parse_whatsapp(str(p), "Nico")
    assert len(results) == 2


def test_author_casefold(tmp_path):
    """NICO (mayúsculas) debe matchear cuando el autor esperado es 'nico'."""
    content = (
        "1/3/24, 10:30 - NICO: Este mensaje en mayúsculas debe ser capturado igualmente.\n"
        "1/3/24, 10:31 - Amiga: Respuesta breve aquí.\n"
    )
    p = _write_tmp(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with patch("scripts.parse_whatsapp.anonymize", side_effect=lambda x: x):
        results = parse_whatsapp(str(p), "nico")
    assert len(results) == 1


def test_system_messages_ignored(tmp_path):
    """Líneas de sistema no deben incluirse en los resultados."""
    content = (
        "1/3/24, 10:30 - Nico: <Multimedia omitido>\n"
        "1/3/24, 10:31 - Nico: Se cifran los mensajes de extremo a extremo en esta conversación.\n"
        "1/3/24, 10:32 - Nico: Este mensaje real sí debe aparecer en los resultados finales.\n"
    )
    p = _write_tmp(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with patch("scripts.parse_whatsapp.anonymize", side_effect=lambda x: x):
        results = parse_whatsapp(str(p), "Nico")
    assert len(results) == 1
    assert "real" in results[0]["text"]


def test_multiline_message(tmp_path):
    """Mensajes que continúan en líneas sin timestamp deben unirse."""
    content = (
        "1/3/24, 10:30 - Nico: Primera línea del mensaje\n"
        "segunda línea del mismo mensaje\n"
        "tercera línea, todo junto.\n"
        "1/3/24, 10:31 - Otro: Fin.\n"
    )
    p = _write_tmp(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with patch("scripts.parse_whatsapp.anonymize", side_effect=lambda x: x):
        results = parse_whatsapp(str(p), "Nico")
    assert len(results) == 1
    assert "segunda línea" in results[0]["text"]
    assert "tercera línea" in results[0]["text"]


def test_bom_encoding(tmp_path):
    """Archivo con BOM (utf-8-sig, exports iOS) se parsea correctamente."""
    content = (
        "\ufeff1/3/24, 10:30\u202fa.\u202fm. - Nico: Mensaje con BOM al inicio del archivo de exportación.\n"
        "1/3/24, 10:31\u202fa.\u202fm. - Otro: Ok.\n"
    )
    p = tmp_path / "bom_chat.txt"
    p.write_bytes(content.encode("utf-8"))
    from scripts.parse_whatsapp import parse_whatsapp
    with patch("scripts.parse_whatsapp.anonymize", side_effect=lambda x: x):
        results = parse_whatsapp(str(p), "Nico")
    assert len(results) == 1


def test_min_len_filter(tmp_path):
    """Mensajes más cortos que min_len deben descartarse."""
    content = (
        "1/3/24, 10:30 - Nico: Corto.\n"
        "1/3/24, 10:31 - Nico: Este mensaje es lo suficientemente largo como para pasar el filtro mínimo.\n"
    )
    p = _write_tmp(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with patch("scripts.parse_whatsapp.anonymize", side_effect=lambda x: x):
        results = parse_whatsapp(str(p), "Nico", min_len=30)
    assert len(results) == 1


def test_lrm_rlm_stripped(tmp_path):
    """Marcas LRM/RLM de exports iOS no deben romper el parsing."""
    content = (
        "\u200e1/3/24, 10:30 - Nico: Mensaje con marcas de dirección Unicode al inicio de la línea.\n"
        "1/3/24, 10:31 - Otro: Respuesta normal aquí.\n"
    )
    p = _write_tmp(tmp_path, content)
    from scripts.parse_whatsapp import parse_whatsapp
    with patch("scripts.parse_whatsapp.anonymize", side_effect=lambda x: x):
        results = parse_whatsapp(str(p), "Nico")
    assert len(results) == 1
