import email.utils as email_utils
import logging
import mailbox
import json
import re
from pathlib import Path
from email.header import decode_header

from scripts.anonymize import anonymize

logger = logging.getLogger(__name__)

_STOP_PATTERNS = [
    re.compile(r'^>'),
    re.compile(r'^On .+ <.+@.+> wrote:', re.IGNORECASE),
    re.compile(r'^El .+ escribió:', re.IGNORECASE),
    re.compile(r'^-{3,}'),
    re.compile(r'^_{3,}'),
    re.compile(r'^--\s*$'),
    re.compile(r'^Enviado desde mi (iPhone|Android|iPad)', re.IGNORECASE),
    re.compile(r'^Este mensaje y sus adjuntos', re.IGNORECASE),
    re.compile(r'^This email and any attachments', re.IGNORECASE),
    re.compile(r'^\bCONFIDENTIAL\b', re.IGNORECASE),
]

# Patrón de cita multilínea (Outlook: "De: ... Enviado: ... Para: ...")
_QUOTE_RE = re.compile(
    r'(El .{5,80} escribió:|De:.*\nEnviado:.*\nPara:.*\n)',
    re.IGNORECASE | re.DOTALL,
)

_MAX_PAYLOAD_BYTES = 500_000


def _decode_str(s) -> str:
    if s is None:
        return ""
    parts = decode_header(s)
    result = []
    for part, enc in parts:
        if isinstance(part, bytes):
            result.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            result.append(str(part))
    return " ".join(result)


def _get_plain_from_html(html_bytes: bytes, charset: str) -> str:
    """Convierte HTML a texto plano. Requiere html2text; fallback con regex básico."""
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.ignore_emphasis = True
        return h.handle(html_bytes.decode(charset, errors="replace"))
    except ImportError:
        html_str = html_bytes.decode(charset, errors="replace")
        return re.sub(r'<[^>]+>', ' ', html_str)


def _extract_text(msg) -> str:
    """
    Extrae texto plano. Fallback a text/html si no hay text/plain usable.
    Trunca partes > _MAX_PAYLOAD_BYTES en vez de descartar el mensaje entero.
    """
    plain_parts = []
    html_parts = []

    for part in (msg.walk() if msg.is_multipart() else [msg]):
        ctype = part.get_content_type()
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        charset = part.get_content_charset() or "utf-8"
        if len(payload) > _MAX_PAYLOAD_BYTES:
            payload = payload[:_MAX_PAYLOAD_BYTES]

        if ctype == "text/plain":
            plain_parts.append(payload.decode(charset, errors="replace"))
        elif ctype == "text/html":
            html_parts.append(_get_plain_from_html(payload, charset))

    if plain_parts:
        return "\n".join(plain_parts)
    return "\n".join(html_parts)


def _clean(text: str) -> str:
    # Cortar antes de bloques de cita multilinea
    text = _QUOTE_RE.split(text)[0]
    lines = text.split("\n")
    clean = []
    for line in lines:
        stripped = line.strip()
        if any(p.match(stripped) for p in _STOP_PATTERNS):
            break
        clean.append(line)
    return "\n".join(clean).strip()


def parse_mbox(filepath: str, sender_email: str, min_len: int = 100) -> list[dict]:
    """
    Extrae emails enviados por sender_email del archivo .mbox.
    Aplica anonimización de PII antes de retornar.
    Usa match exacto de dirección y prefiere carpeta Sent (X-Gmail-Labels).
    """
    mbox = mailbox.mbox(filepath)
    examples = []
    _skip_kw = {"unsubscribe", "newsletter", "noreply", "no-reply", "automated"}

    for msg in mbox:
        # Match exacto de la dirección de email
        raw_from = _decode_str(msg.get("From", ""))
        _, addr = email_utils.parseaddr(raw_from)
        if addr.lower() != sender_email.lower():
            continue

        # Si el .mbox trae etiquetas de Gmail, preferir carpeta Sent
        labels = msg.get("X-Gmail-Labels", "")
        if labels and "Sent" not in labels:
            continue

        subject = _decode_str(msg.get("Subject", ""))
        if any(kw in subject.lower() for kw in _skip_kw):
            continue

        body = _clean(_extract_text(msg))
        if len(body) < min_len:
            continue

        examples.append({
            "text": anonymize(body),
            "subject": subject,
            "register": "email_prof",
        })

    logger.info("parse_mbox: %d emails extraídos de %s", len(examples), filepath)
    return examples


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    filepath = sys.argv[1]
    email_addr = sys.argv[2]
    examples = parse_mbox(filepath, email_addr)
    print(f"Extraídos {len(examples)} emails enviados por {email_addr}")
    output = "gmail_parsed.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Guardado en {output}")
