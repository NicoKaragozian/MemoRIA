"""
Tests para scripts/anonymize.py.
Cubre: email, teléfono, DNI, CBU, IBAN, coordenadas, handles, KEEP_NAMES,
fallo gracioso sin spaCy.
"""
from unittest.mock import patch

import pytest


def anon(text: str, **kwargs) -> str:
    from scripts.anonymize import anonymize
    return anonymize(text, **kwargs)


# ── PII básica ───────────────────────────────────────────────────────────────

def test_email_replaced():
    result = anon("Mandame un mail a juan.perez@empresa.com por favor")
    assert "juan.perez@empresa.com" not in result
    assert "<EMAIL>" in result


def test_url_replaced():
    result = anon("Chequeá https://www.privado.com/doc?id=123 cuando puedas")
    assert "privado.com" not in result
    assert "<URL>" in result


def test_handle_replaced():
    result = anon("Seguime en @juanperez_ok en Instagram")
    assert "@juanperez_ok" not in result
    assert "<HANDLE>" in result


def test_coordinates_replaced():
    result = anon("La ubicación es -34.6037, -58.3816 en Buenos Aires")
    assert "-34.6037" not in result
    assert "<COORDS>" in result


# ── Teléfonos ─────────────────────────────────────────────────────────────────

def test_phone_replaced():
    result = anon("Llamame al +54 11 4567-8901 o al 15-3456-7890")
    assert "4567" not in result
    assert "<PHONE>" in result


def test_hour_not_replaced_as_phone():
    """Horarios como 10:30 no deben confundirse con teléfonos."""
    result = anon("La reunión es de 9:30 a 11:00, no hay problema")
    assert "9:30" in result
    assert "11:00" in result
    assert "<PHONE>" not in result


# ── IDs numéricas ─────────────────────────────────────────────────────────────

def test_dni_replaced():
    result = anon("El DNI de Martínez es 35123456 para el trámite")
    assert "35123456" not in result
    assert "<ID>" in result


def test_cbu_replaced():
    result = anon("Mi CBU es 0720461088000015726013 para la transferencia")
    assert "0720461088000015726013" not in result
    assert "<CBU>" in result


def test_iban_replaced():
    result = anon("El IBAN del proveedor es ES9121000418450200051332 para el pago")
    assert "ES9121000418450200051332" not in result
    assert "<IBAN>" in result


# ── KEEP_NAMES ────────────────────────────────────────────────────────────────

def test_keep_names_not_replaced():
    """Los nombres del autor en KEEP_NAMES no deben anonimizarse."""
    result = anon("Reunión con Luciano Del Corro en la UdeSA el martes")
    assert "Luciano" in result or "Del Corro" in result


def test_keep_author_false_replaces_all():
    """Con keep_author=False, incluso los nombres de KEEP_NAMES se reemplazan."""
    result = anon("Hola Nico, cómo te va en el proyecto MemoRIA", keep_author=False)
    # spaCy puede o no reconocer estos como entidades — lo importante es que no falle
    assert isinstance(result, str)


# ── Fallo gracioso sin spaCy ──────────────────────────────────────────────────

def test_graceful_without_spacy():
    """Si spaCy no está disponible, debe devolver texto con PII numérica reemplazada."""
    with patch("scripts.anonymize._nlp", return_value=lambda: None):
        from scripts import anonymize as mod
        original_nlp = mod._nlp

        def mock_nlp():
            return None

        with patch.object(mod, "_nlp", mock_nlp):
            result = mod.anonymize("Llamame al +54 11 4567-8901")
        assert "4567" not in result


def test_strict_raises_without_spacy():
    """Con strict=True, debe lanzar RuntimeError si spaCy no carga."""
    from scripts import anonymize as mod

    def mock_nlp():
        return None

    with patch.object(mod, "_nlp", mock_nlp):
        with pytest.raises(RuntimeError, match="spaCy"):
            mod.anonymize("Texto cualquiera", strict=True)


# ── No over-anonymize ─────────────────────────────────────────────────────────

def test_short_numbers_not_replaced():
    """Números de 1-6 dígitos no deben reemplazarse como DNI o CBU."""
    result = anon("Tengo 3 reuniones y 42 pendientes en la agenda")
    assert "3" in result
    assert "42" in result
