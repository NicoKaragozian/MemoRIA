"""
Pipeline de anonimización de PII para datos de entrenamiento de MemoRIA.
Reemplaza teléfonos, emails, URLs, DNIs y nombres propios de terceros
con tokens genéricos, preservando los datos del autor.
"""

import re
from functools import lru_cache

PHONE_RE    = re.compile(r'\+?\d[\d\s\-().]{6,}\d')
EMAIL_RE    = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
URL_RE      = re.compile(r'https?://\S+')
DNI_RE      = re.compile(r'\b\d{7,8}\b')
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
    except (ImportError, OSError):
        return None


def anonymize(text: str, keep_author: bool = True) -> str:
    """
    Reemplaza PII en el texto con tokens genéricos.

    Args:
        text:        texto a anonimizar
        keep_author: si True, no anonimiza los nombres del autor (KEEP_NAMES)

    Returns:
        texto con PII reemplazada
    """
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = DNI_RE.sub("<ID>", text)
    text = LONG_NUM_RE.sub("<NUM>", text)

    nlp = _nlp()
    if nlp is None:
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
    samples = [
        "Hola Nico, te llamo al +54 11 4567-8901 o al 15-3456-7890",
        "Mandame un mail a juan.perez@empresa.com por favor",
        "El DNI de Martínez es 35123456, el de López es 27654321",
        "Che, mirá este link https://www.ejemplo.com/privado/documento",
        "Reunión con Luciano Del Corro el martes en la UdeSA",
    ]

    print("Test de anonimización:\n")
    for s in samples:
        result = anonymize(s)
        print(f"  Original:  {s}")
        print(f"  Anonimizado: {result}")
        print()
