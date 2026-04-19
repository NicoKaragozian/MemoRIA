import logging
import re
import json
from pathlib import Path

from scripts.anonymize import anonymize

logger = logging.getLogger(__name__)

# Contempla: corchetes iOS/Android, narrow NBSP (\u202f) y NBSP (\u00a0) antes de AM/PM
_PATTERN = re.compile(
    r'\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+'
    r'(\d{1,2}:\d{2}(?::\d{2})?(?:[\s\u202f\u00a0]*[ap]\.?\s*m\.?)?)'
    r'\]?\s*[–\-]?\s*(.+?):\s+(.*)',
    re.IGNORECASE | re.UNICODE,
)

_SYSTEM_PATTERNS = [
    r'<Multimedia omitido>', r'imagen omitida', r'audio omitido',
    r'video omitido', r'sticker omitido', r'Se cifran los mensajes',
    r'Los mensajes y las llamadas', r'cambió el asunto',
    r'añadió a', r'eliminó a', r'salió del grupo',
]

# Marcas de dirección Unicode presentes en exports iOS
_LRM_RLM = re.compile(r'[\u200e\u200f]')


def _is_system(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _SYSTEM_PATTERNS)


def parse_whatsapp(filepath: str, author_name: str, min_len: int = 30) -> list[dict]:
    """
    Parsea un export .txt de WhatsApp y extrae los mensajes del autor.
    Aplica anonimización de PII antes de retornar.
    Soporta exports iOS (utf-8-sig, NBSP) y Android (utf-8, narrow NBSP).
    """
    examples = []
    current_author = None
    current_text = []
    author_norm = author_name.casefold().strip()
    total_lines = 0
    matched_lines = 0

    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = _LRM_RLM.sub('', line).strip()
            total_lines += 1
            match = _PATTERN.match(line)

            if match:
                matched_lines += 1
                if current_author is not None and current_author.casefold() == author_norm and current_text:
                    full_text = " ".join(current_text).strip()
                    if len(full_text) >= min_len and not _is_system(full_text):
                        examples.append({
                            "text": anonymize(full_text),
                            "register": "casual",
                        })
                current_author = match.group(3).strip()
                current_text = [match.group(4).strip()]
            elif current_author is not None and current_author.casefold() == author_norm and line:
                current_text.append(line)

    if current_author is not None and current_author.casefold() == author_norm and current_text:
        full_text = " ".join(current_text).strip()
        if len(full_text) >= min_len and not _is_system(full_text):
            examples.append({"text": anonymize(full_text), "register": "casual"})

    logger.info(
        "%d/%d líneas matchearon el patrón; %d mensajes del autor extraídos",
        matched_lines, total_lines, len(examples),
    )
    return examples


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    filepath = sys.argv[1]
    author = sys.argv[2]
    examples = parse_whatsapp(filepath, author)
    print(f"Extraídos {len(examples)} mensajes de {author}")
    output = Path(filepath).stem + "_parsed.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Guardado en {output}")
