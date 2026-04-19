"""
Tests para scripts/parse_gmail.py.
Cubre: text/plain, text/html-only, multipart, firma, quotes Outlook/Gmail,
match exacto de dirección de email.
"""
import mailbox
import textwrap
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_mbox(tmp_path: Path, messages: list) -> str:
    mbox_path = str(tmp_path / "test.mbox")
    mbox = mailbox.mbox(mbox_path, create=True)
    for msg in messages:
        mbox.add(msg)
    mbox.flush()
    mbox.close()
    return mbox_path


def _simple_msg(sender: str, subject: str, body: str, content_type="text/plain") -> MIMEText:
    msg = MIMEText(body, content_type.split("/")[1], "utf-8")
    msg["From"] = sender
    msg["Subject"] = subject
    msg["X-Gmail-Labels"] = "Sent"
    return msg


def _parse(mbox_path: str, email: str = "nico@example.com", min_len: int = 10):
    from scripts.parse_gmail import parse_mbox
    with patch("scripts.parse_gmail.anonymize", side_effect=lambda x: x):
        return parse_mbox(mbox_path, email, min_len=min_len)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_plain_text_extracted(tmp_path):
    msg = _simple_msg("Nico <nico@example.com>", "Hola", "Este es el cuerpo del email.")
    path = _make_mbox(tmp_path, [msg])
    results = _parse(path)
    assert len(results) == 1
    assert "cuerpo del email" in results[0]["text"]
    assert results[0]["register"] == "email_prof"


def test_html_only_fallback(tmp_path):
    """Si no hay text/plain, debe usar text/html con html2text."""
    html = "<html><body><p>Este es un email en <b>HTML</b> solamente.</p></body></html>"
    msg = _simple_msg("Nico <nico@example.com>", "HTML", html, "text/html")
    path = _make_mbox(tmp_path, [msg])
    results = _parse(path)
    assert len(results) == 1
    assert "HTML" in results[0]["text"] or "html" in results[0]["text"].lower()


def test_multipart_prefers_plain(tmp_path):
    """En mensajes multipart, text/plain tiene preferencia sobre text/html."""
    msg = MIMEMultipart("alternative")
    msg["From"] = "Nico <nico@example.com>"
    msg["Subject"] = "Multipart"
    msg["X-Gmail-Labels"] = "Sent"
    msg.attach(MIMEText("Texto plano del email.", "plain", "utf-8"))
    msg.attach(MIMEText("<p>Texto HTML del email.</p>", "html", "utf-8"))
    path = _make_mbox(tmp_path, [msg])
    results = _parse(path)
    assert len(results) == 1
    assert "plano" in results[0]["text"]


def test_gmail_quote_stripped(tmp_path):
    body = "Mi respuesta original aquí.\n\nEl mar, 3 ene 2024 a las 10:00 escribió:\n> Texto citado que no queremos."
    msg = _simple_msg("Nico <nico@example.com>", "Re: algo", body)
    path = _make_mbox(tmp_path, [msg])
    results = _parse(path)
    assert len(results) == 1
    assert "citado" not in results[0]["text"]
    assert "respuesta original" in results[0]["text"]


def test_outlook_quote_stripped(tmp_path):
    body = "Mi texto limpio de respuesta.\n\nDe: alguien@empresa.com\nEnviado: martes 3 de enero\nPara: nico@example.com\n> Hilo anterior"
    msg = _simple_msg("Nico <nico@example.com>", "Re: reunión", body)
    path = _make_mbox(tmp_path, [msg])
    results = _parse(path)
    assert len(results) == 1
    assert "Hilo anterior" not in results[0]["text"]


def test_signature_stripped(tmp_path):
    body = "Contenido real del email que quiero guardar en el dataset.\n\n--\nNico K.\nico@example.com"
    msg = _simple_msg("Nico <nico@example.com>", "Test", body)
    path = _make_mbox(tmp_path, [msg])
    results = _parse(path)
    assert len(results) == 1
    assert "Nico K." not in results[0]["text"]


def test_exact_email_match(tmp_path):
    """No debe capturar emails cuya dirección solo contiene substring del sender."""
    msg_other = _simple_msg("otro_nico@example.com", "Spam", "Este no debería incluirse.")
    msg_self = _simple_msg("nico@example.com", "Mío", "Este sí debería incluirse en el resultado.")
    path = _make_mbox(tmp_path, [msg_other, msg_self])
    results = _parse(path, email="nico@example.com")
    assert len(results) == 1
    assert "sí debería" in results[0]["text"]


def test_sent_label_filter(tmp_path):
    """Si hay label Gmail, debe preferir carpeta Sent."""
    msg_inbox = MIMEText("Mensaje de inbox.", "plain", "utf-8")
    msg_inbox["From"] = "nico@example.com"
    msg_inbox["X-Gmail-Labels"] = "Inbox"

    msg_sent = MIMEText("Mensaje enviado con suficiente largo.", "plain", "utf-8")
    msg_sent["From"] = "nico@example.com"
    msg_sent["X-Gmail-Labels"] = "Sent"

    path = _make_mbox(tmp_path, [msg_inbox, msg_sent])
    results = _parse(path, email="nico@example.com")
    assert len(results) == 1
    assert "enviado" in results[0]["text"]


def test_min_len_filter(tmp_path):
    msg = _simple_msg("nico@example.com", "Corto", "Ok.")
    path = _make_mbox(tmp_path, [msg])
    results = _parse(path, min_len=100)
    assert len(results) == 0
