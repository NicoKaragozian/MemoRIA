import re
import json
from pathlib import Path

from scripts.anonymize import anonymize

# Regex tolerante: contempla corchetes [15/4/26, 14:32:05] Nico: ...,
# narrow no-break space (\u202f) antes de AM/PM, y separadores – o -
_PATTERN = re.compile(
    r'\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+'
    r'(\d{1,2}:\d{2}(?::\d{2})?(?:[\s\u202f]*[ap]\.?\s*m\.?)?)'
    r'\]?\s*[–\-]?\s*(.+?):\s+(.*)',
    re.IGNORECASE | re.UNICODE,
)

_SYSTEM_PATTERNS = [
    r'<Multimedia omitido>', r'imagen omitida', r'audio omitido',
    r'video omitido', r'sticker omitido', r'Se cifran los mensajes',
    r'Los mensajes y las llamadas', r'cambió el asunto',
    r'añadió a', r'eliminó a', r'salió del grupo',
]


def _is_system(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _SYSTEM_PATTERNS)


def parse_whatsapp(filepath: str, author_name: str, min_len: int = 30) -> list[dict]:
    """
    Parsea un export .txt de WhatsApp y extrae los mensajes del autor.
    Aplica anonimización de PII antes de retornar.
    """
    examples = []
    current_author = None
    current_text = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = _PATTERN.match(line)

            if match:
                if current_author == author_name and current_text:
                    full_text = " ".join(current_text).strip()
                    if len(full_text) >= min_len and not _is_system(full_text):
                        examples.append({
                            "text": anonymize(full_text),
                            "register": "casual",
                        })
                current_author = match.group(3).strip()
                current_text = [match.group(4).strip()]
            elif current_author == author_name and line:
                current_text.append(line)

    if current_author == author_name and current_text:
        full_text = " ".join(current_text).strip()
        if len(full_text) >= min_len and not _is_system(full_text):
            examples.append({"text": anonymize(full_text), "register": "casual"})

    return examples


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1]
    author = sys.argv[2]
    examples = parse_whatsapp(filepath, author)
    print(f"Extraídos {len(examples)} mensajes de {author}")
    output = Path(filepath).stem + "_parsed.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Guardado en {output}")
