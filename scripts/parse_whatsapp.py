import logging
import re
import json
from pathlib import Path

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

# Mensajes que son puro ruido conversacional
_TRASH_RE = re.compile(
    r'^(?:ja+|je+|ha+|he+|xi+|xd+|k+|ok+|[😀-🙏🌀-🗿]+|[?!.,\s]+)$',
    re.IGNORECASE | re.UNICODE,
)

# Normalización de risa/alargamiento
_LAUGH_RE = re.compile(r'\b(ja|je|ha|he){3,}\b', re.IGNORECASE)
_ELONGATION_RE = re.compile(r'(.)\1{2,}')

# Marcas de dirección Unicode presentes en exports iOS
_LRM_RLM = re.compile(r'[\u200e\u200f]')


def _is_system(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _SYSTEM_PATTERNS)


def _normalize(text: str) -> str:
    text = _LAUGH_RE.sub(lambda m: m.group(1) + 'ja', text)
    text = _ELONGATION_RE.sub(r'\1\1', text)
    return text


def _is_trash(text: str) -> bool:
    if _TRASH_RE.match(text):
        return True
    words = text.split()
    if len(set(words)) < 3:
        return True
    return False


def _is_spanish(text: str) -> bool:
    try:
        from langdetect import detect
        return detect(text) == 'es'
    except Exception:
        return True  # si no está instalado o falla, no filtrar


def parse_whatsapp(
    filepath: str,
    author_name: str,
    min_len: int = 30,
    max_len: int = 1500,
    min_context_len: int = 10,  # contexto muy corto (ej: "ok") no aporta
) -> list[dict]:
    """
    Parsea un export .txt de WhatsApp y extrae los mensajes del autor.
    Cuando existe un mensaje previo de otra persona, lo guarda como 'context'
    para usarlo como USER en el par de fine-tuning.

    Soporta exports iOS (utf-8-sig, NBSP) y Android (utf-8, narrow NBSP).
    """
    examples = []
    author_norm = author_name.casefold().strip()
    total_lines = 0
    matched_lines = 0

    # Estado del parser
    current_author: str | None = None
    current_text: list[str] = []

    # Último mensaje de otra persona (candidato a contexto)
    prev_other_author: str | None = None
    prev_other_text: list[str] = []

    def _flush():
        nonlocal prev_other_author, prev_other_text

        if not (current_author and current_author.casefold() == author_norm and current_text):
            # Si el mensaje actual NO es del autor, actualizar el contexto "other"
            if current_author and current_author.casefold() != author_norm and current_text:
                full = " ".join(current_text).strip()
                if not _is_system(full) and not _is_trash(full):
                    prev_other_author = current_author
                    prev_other_text = list(current_text)
            return

        full_text = "\n".join(current_text).strip()

        # Filtros de calidad sobre la respuesta del autor
        if len(full_text) < min_len or len(full_text) > max_len:
            return
        if _is_system(full_text):
            return
        if _is_trash(full_text):
            return
        if not _is_spanish(full_text):
            return

        # Construir contexto si existe y es de calidad suficiente
        context: str | None = None
        if prev_other_text:
            ctx_raw = " ".join(prev_other_text).strip()
            if (
                len(ctx_raw) >= min_context_len
                and not _is_system(ctx_raw)
                and not _is_trash(ctx_raw)
            ):
                context = _normalize(ctx_raw)

        item: dict = {
            "text": _normalize(full_text),
            "register": "casual",
        }
        if context:
            item["context"] = context

        examples.append(item)

        # Una vez consumido el contexto, lo reseteamos para no reutilizarlo
        # en el próximo mensaje del autor (podría ser una respuesta múltiple)
        prev_other_author = None
        prev_other_text = []

    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = _LRM_RLM.sub('', line).strip()
            total_lines += 1
            match = _PATTERN.match(line)

            if match:
                matched_lines += 1
                _flush()
                current_author = match.group(3).strip()
                current_text = [match.group(4).strip()]
            elif current_author is not None and line:
                # Continuación multilínea del mensaje actual
                current_text.append(line)

    _flush()

    with_context = sum(1 for e in examples if "context" in e)
    logger.info(
        "%d/%d líneas matchearon el patrón; %d mensajes del autor extraídos "
        "(%d con contexto, %d sin contexto)",
        matched_lines, total_lines, len(examples), with_context, len(examples) - with_context,
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
