"""
Parser conversacional de WhatsApp.

Extrae pares (contexto, target) para fine-tuning de un asistente que aprende
a responder en el estilo del usuario. Cada par representa un turno del usuario
junto con los mensajes anteriores de la conversación como contexto.

Output por archivo: list[dict] con campos
  - chat_name:    nombre del chat (sale del nombre del archivo)
  - is_group:     True si hay >2 autores activos
  - participants: autores únicos del chat (excluyendo al usuario)
  - context:      list[{author, text}] con los últimos N mensajes anteriores
  - target:       turno del usuario (mensajes consecutivos unidos)

Ver docs/CHATBOT_DESIGN.md para las decisiones de diseño.
"""
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from scripts.anonymize import anonymize

logger = logging.getLogger(__name__)

# Pattern: corchetes iOS/Android, narrow NBSP ( ) y NBSP ( ) antes de AM/PM.
_PATTERN = re.compile(
    r'\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+'
    r'(\d{1,2}:\d{2}(?::\d{2})?(?:[\s  ]*[ap]\.?\s*m\.?)?)'
    r'\]?\s*[–\-]?\s*(.+?):\s+(.*)',
    re.IGNORECASE | re.UNICODE,
)

_SYSTEM_PATTERNS = [
    r'<Multimedia omitido>', r'imagen omitida', r'audio omitido',
    r'video omitido', r'sticker omitido', r'Se cifran los mensajes',
    r'Los mensajes y las llamadas', r'cambió el asunto',
    r'añadió a', r'eliminó a', r'salió del grupo',
    r'image omitted', r'video omitted', r'sticker omitted',
    r'Contact card omitted', r'GIF omitted', r'document omitted',
    r'Voice call', r'Silenced voice call',
    r'Messages and calls are end-to-end encrypted',
    r'You deleted this message', r'This message was edited',
    r'Location:',
]
_SYSTEM_RE = re.compile('|'.join(_SYSTEM_PATTERNS), re.IGNORECASE)
_LRM_RLM = re.compile(r'[\u200E\u200F]')

# Aliases del usuario que WhatsApp usa según el idioma del export.
# Se filtran del listado de participantes — siempre apuntan al usuario.
_USER_ALIASES = frozenset({"you", "tú", "tu", "yo"})

_TS_FORMATS = (
    '%d/%m/%Y %H:%M:%S',
    '%d/%m/%Y %H:%M',
    '%d/%m/%y %H:%M:%S',
    '%d/%m/%y %H:%M',
    '%d/%m/%Y %I:%M:%S %p',
    '%d/%m/%Y %I:%M %p',
    '%d/%m/%y %I:%M:%S %p',
    '%d/%m/%y %I:%M %p',
)


def _parse_timestamp(date_str: str, time_str: str) -> Optional[datetime]:
    """Parsea fecha + hora de WhatsApp en datetime, tolerante a varios formatos."""
    # Normalizar NBSPs y AM/PM ("a. m." / "a.m." / "AM" / "p.m.")
    t = time_str.replace(' ', ' ').replace(' ', ' ')
    t = re.sub(r'([ap])\s*\.?\s*m\.?', lambda m: m.group(1).upper() + 'M', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    full = f'{date_str.strip()} {t}'
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(full, fmt)
        except ValueError:
            continue
    return None


def _is_system(text: str) -> bool:
    return bool(_SYSTEM_RE.search(text))


def _parse_raw_messages(filepath: str) -> list[dict]:
    """
    Lee el archivo y devuelve list[{timestamp, author, text}] con todos los
    mensajes parseables, descartando mensajes de sistema y los que no tienen
    timestamp válido. NO aplica anonimización.
    """
    messages: list[dict] = []
    current: Optional[dict] = None

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = _LRM_RLM.sub('', line).rstrip('\n')
            stripped = line.strip()
            m = _PATTERN.match(stripped)
            if m:
                if current is not None:
                    messages.append(current)
                date_str, time_str, author, text = m.groups()
                ts = _parse_timestamp(date_str, time_str)
                current = {
                    'timestamp': ts,
                    'author': author.strip(),
                    'text': text.strip(),
                }
            elif current is not None and stripped:
                # Continuación multilínea del mensaje en curso.
                current['text'] = (current['text'] + ' ' + stripped).strip()

    if current is not None:
        messages.append(current)

    return [
        m for m in messages
        if m['timestamp'] is not None and m['text'] and not _is_system(m['text'])
    ]


def _segment_by_gap(messages: list[dict], gap_hours: float) -> list[list[dict]]:
    """
    Segmenta la lista de mensajes en conversaciones por gap de tiempo entre
    mensajes consecutivos.
    """
    if not messages:
        return []
    gap = timedelta(hours=gap_hours)
    conversations: list[list[dict]] = [[messages[0]]]
    for prev, curr in zip(messages, messages[1:]):
        if curr['timestamp'] - prev['timestamp'] >= gap:
            conversations.append([curr])
        else:
            conversations[-1].append(curr)
    return conversations


def _extract_chat_name(filepath: str) -> str:
    """Extrae el nombre del chat del nombre del archivo."""
    stem = Path(filepath).stem
    prefix = 'WhatsApp Chat - '
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def parse_whatsapp(
    filepath: str,
    author_name: str,
    context_size: int = 20,
    gap_hours: float = 6.0,
    min_target_chars: int = 30,
) -> list[dict]:
    """
    Extrae pares conversacionales de un export .txt de WhatsApp.

    Para cada turno del usuario que no sea el primero de su conversación y
    que tenga al menos `min_target_chars` caracteres, genera un par con los
    `context_size` mensajes anteriores como contexto.

    Args:
        filepath:         ruta al .txt exportado de WhatsApp
        author_name:      nombre del usuario tal como aparece en el chat
        context_size:     cantidad de mensajes previos incluidos como contexto
        gap_hours:        gap (en horas) que cierra una conversación
        min_target_chars: largo mínimo del turno del usuario para entrenar
    """
    chat_name = _extract_chat_name(filepath)
    author_norm = author_name.casefold().strip()
    chat_name_norm = chat_name.casefold().strip()

    def is_user(author: str) -> bool:
        a = author.casefold().strip()
        return a == author_norm or a in _USER_ALIASES

    raw_messages = _parse_raw_messages(filepath)
    # Anonimizar contenido (autores se preservan).
    for msg in raw_messages:
        msg['text'] = anonymize(msg['text'])
    raw_messages = [m for m in raw_messages if m['text'].strip()]

    # Normalizar aliases del usuario al author_name canónico (ej. "You" → "Clara Kearney").
    for msg in raw_messages:
        if is_user(msg['author']):
            msg['author'] = author_name

    if not raw_messages:
        logger.info('%s: sin mensajes parseables', filepath)
        return []

    # Detectar si es grupo en base a autores únicos.
    authors = {m['author'] for m in raw_messages}
    is_group = len(authors) > 2

    # En grupos, descartar mensajes cuyo autor coincide con el nombre del chat:
    # son eventos de sistema ("X cambió el nombre del grupo") que el regex
    # captura como si fueran texto normal. En 1:1 no aplica porque el
    # chat_name justamente ES el nombre del otro interlocutor.
    if is_group:
        raw_messages = [m for m in raw_messages if m['author'].casefold().strip() != chat_name_norm]
        authors = {m['author'] for m in raw_messages}

    participants = sorted(a for a in authors if a.casefold() != author_norm)

    conversations = _segment_by_gap(raw_messages, gap_hours)

    pairs: list[dict] = []
    for conv in conversations:
        i = 0
        while i < len(conv):
            if conv[i]['author'].casefold() != author_norm:
                i += 1
                continue
            # Inicio de turno del usuario: agrupar mensajes consecutivos suyos.
            j = i
            while j < len(conv) and conv[j]['author'].casefold() == author_norm:
                j += 1
            target_text = '\n'.join(m['text'] for m in conv[i:j]).strip()

            # Filtros: primer turno de la conversación o target muy corto.
            if i == 0 or len(target_text) < min_target_chars:
                i = j
                continue

            context_msgs = conv[max(0, i - context_size):i]
            pairs.append({
                'chat_name': chat_name,
                'is_group': is_group,
                'participants': participants,
                'context': [
                    {'author': m['author'], 'text': m['text']}
                    for m in context_msgs
                ],
                'target': target_text,
            })
            i = j

    logger.info(
        '%s: %d mensajes, %d conversaciones, %d pares',
        filepath, len(raw_messages), len(conversations), len(pairs),
    )
    return pairs


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    filepath = sys.argv[1]
    author = sys.argv[2]
    pairs = parse_whatsapp(filepath, author)
    print(f'Extraídos {len(pairs)} pares conversacionales de {author}')
    output = Path(filepath).stem + '_pairs.jsonl'
    with open(output, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f'Guardado en {output}')
