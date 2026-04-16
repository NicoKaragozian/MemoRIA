import mailbox
import json
import re
from pathlib import Path
from email.header import decode_header

from scripts.anonymize import anonymize

# Más estrictos que la versión original para no cortar en falsos positivos
_STOP_PATTERNS = [
    re.compile(r'^>'),
    re.compile(r'^On .+ <.+@.+> wrote:', re.IGNORECASE),
    re.compile(r'^El .+ escribió:', re.IGNORECASE),
    re.compile(r'^-{3,}'),
    re.compile(r'^_{3,}'),
    re.compile(r'^Este mensaje y sus adjuntos', re.IGNORECASE),
    re.compile(r'^This email and any attachments', re.IGNORECASE),
    re.compile(r'^\bCONFIDENTIAL\b', re.IGNORECASE),
]

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


def _extract_text(msg) -> str:
    text_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload and len(payload) <= _MAX_PAYLOAD_BYTES:
                    charset = part.get_content_charset() or "utf-8"
                    text_parts.append(payload.decode(charset, errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload and len(payload) <= _MAX_PAYLOAD_BYTES:
            charset = msg.get_content_charset() or "utf-8"
            text_parts.append(payload.decode(charset, errors="replace"))
    return "\n".join(text_parts)


def _clean(text: str) -> str:
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
    """
    mbox = mailbox.mbox(filepath)
    examples = []
    _skip_kw = {"unsubscribe", "newsletter", "noreply", "no-reply", "automated"}

    for msg in mbox:
        sender = _decode_str(msg.get("From", ""))
        if sender_email.lower() not in sender.lower():
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

    return examples


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1]
    email_addr = sys.argv[2]
    examples = parse_mbox(filepath, email_addr)
    print(f"Extraídos {len(examples)} emails enviados por {email_addr}")
    output = "gmail_parsed.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Guardado en {output}")
