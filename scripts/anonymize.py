"""
Pipeline de anonimización de PII para datos de entrenamiento de MemoRIA.
Reemplaza teléfonos, emails, URLs, DNIs, CBUs, IBANs, coordenadas, handles
y nombres propios de terceros con tokens genéricos, preservando el autor.
"""

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

EMAIL_RE    = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
URL_RE      = re.compile(r'https?://\S+')

# Detectar horas antes del regex de teléfono para evitar falsos positivos
HOUR_RE     = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?\s*m\.?)?\b', re.IGNORECASE)

# Teléfono: ≥8 dígitos, no precedido ni seguido por ":" o dígito adicional
PHONE_RE    = re.compile(r'(?<![:\d])(?:\+?\d[\d\s\-().]{6,}\d)(?![:/\d])')

# DNI argentino (7-8 dígitos)
DNI_RE      = re.compile(r'\b\d{7,8}\b')

# CBU argentino (22 dígitos exactos)
CBU_RE      = re.compile(r'\b\d{22}\b')

# IBAN internacional
IBAN_RE     = re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b')

# Coordenadas geográficas decimales
COORD_RE    = re.compile(r'-?\d{1,2}\.\d{4,},\s*-?\d{1,3}\.\d{4,}')

# Handles de redes sociales (@usuario)
HANDLE_RE   = re.compile(r'@[\w_]{3,}')

# Números largos residuales (≥9 dígitos — tarjetas, cuentas)
LONG_NUM_RE = re.compile(r'\b\d{9,}\b')

# Nombres propios del autor que NO se anonimizan
KEEP_NAMES = frozenset({
    "nico", "nicolás", "nicolas", "karagozian",
    "luciano", "del corro", "memoria",
})


@lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy
        return spacy.load("es_core_news_sm", disable=["parser", "lemmatizer"])
    except (ImportError, OSError) as exc:
        logger.warning(
            "spaCy no disponible (%s) — se omite anonimización de nombres propios.", exc
        )
        return None


def anonymize(text: str, keep_author: bool = True, strict: bool = False) -> str:
    """
    Reemplaza PII en el texto con tokens genéricos.

    Args:
        text:        texto a anonimizar
        keep_author: si True, preserva los nombres del autor (KEEP_NAMES)
        strict:      si True, lanza RuntimeError cuando spaCy no está disponible
    """
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = HANDLE_RE.sub("<HANDLE>", text)
    text = COORD_RE.sub("<COORDS>", text)
    text = CBU_RE.sub("<CBU>", text)
    text = IBAN_RE.sub("<IBAN>", text)

    # Marcar horas para que PHONE_RE no las capte como teléfonos
    _hours: list[str] = []
    def _save_hour(m: re.Match) -> str:
        _hours.append(m.group())
        return f"\x00H{len(_hours)-1}\x00"
    text = HOUR_RE.sub(_save_hour, text)
    text = PHONE_RE.sub("<PHONE>", text)
    for i, h in enumerate(_hours):
        text = text.replace(f"\x00H{i}\x00", h)

    text = DNI_RE.sub("<ID>", text)
    text = LONG_NUM_RE.sub("<NUM>", text)

    nlp = _nlp()
    if nlp is None:
        if strict:
            raise RuntimeError("spaCy no disponible y strict=True")
        return text

    doc = nlp(text)
    parts = []
    last = 0

    for ent in doc.ents:
        if ent.label_ not in {"PER", "LOC", "ORG"}:
            continue
        if keep_author:
            lower = ent.text.lower()
            if any(k in lower for k in KEEP_NAMES):
                continue
        parts.append(text[last:ent.start_char])
        parts.append(f"<{ent.label_}>")
        last = ent.end_char

    parts.append(text[last:])
    return "".join(parts)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    samples = [
        "Hola Nico, te llamo al +54 11 4567-8901 o al 15-3456-7890",
        "Mandame un mail a juan.perez@empresa.com por favor",
        "El DNI de Martínez es 35123456, el de López es 27654321",
        "Che, mirá este link https://www.ejemplo.com/privado/documento",
        "Reunión con Luciano Del Corro el martes en la UdeSA",
        "Mi CBU es 0720461088000015726013 para la transferencia",
        "Seguime en @juanperez_ok si querés",
        "La reunión es de 9:30 a 11:00, no llega nadie",
    ]

    print("Test de anonimización:\n")
    for s in samples:
        result = anonymize(s)
        print(f"  Original:    {s}")
        print(f"  Anonimizado: {result}")
        print()
