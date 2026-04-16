"""
E2 — Métricas de estilo léxico-sintácticas.

Correcciones vs. versión original:
- Argentinismos usan \b...\b para evitar match en substrings (re vs. presente)
- Se eliminó content_word_ratio (redundante con TTR) y se agrega emoji_density
  e interjection_density como métricas más informativas para el registro casual
- El filtro por registro se hace sobre el dict estructurado, no sobre el texto
"""

import re
from collections import Counter

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

SPANISH_STOPWORDS = set(stopwords.words("spanish"))

ARGENTINISMOS = [
    "boludo", "che", "posta", "dale", "copado", "quilombo", "laburo",
    "laburar", "loco", "banda", "chabón", "pibe", "fiaca", "bardear",
    "morfar", "chamuyo", "trucho", "zarpado", "groso", "onda",
    "joya", "plata", "guita", "tipo",
]

INTERJECCIONES = ["jaja", "jeje", "jsjs", "ajá", "bah", "uy", "ay", "eh",
                  "mm", "uff", "jua", "xd", "lol"]

_ARGENTINISMO_RE = [re.compile(r'\b' + a + r'\b', re.IGNORECASE) for a in ARGENTINISMOS]
_INTERJ_RE       = [re.compile(r'\b' + i + r'\b', re.IGNORECASE) for i in INTERJECCIONES]
_EMOJI_RE        = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0\U000024C2-\U0001F251]+",
    flags=re.UNICODE,
)


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\sáéíóúüñ]", " ", text)
    return [w for w in text.split() if w and len(w) > 1]


def type_token_ratio(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def avg_sentence_length(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def argentinismo_score(text: str) -> float:
    """Frecuencia de argentinismos por 100 palabras (con \b para evitar falsos positivos)."""
    count = sum(len(p.findall(text)) for p in _ARGENTINISMO_RE)
    words = len(text.split())
    return (count / words * 100) if words else 0.0


def emoji_density(text: str) -> float:
    """Emojis por 100 caracteres."""
    count = len(_EMOJI_RE.findall(text))
    return (count / len(text) * 100) if text else 0.0


def interjection_density(text: str) -> float:
    """Interjecciones informales por 100 palabras."""
    count = sum(len(p.findall(text)) for p in _INTERJ_RE)
    words = len(text.split())
    return (count / words * 100) if words else 0.0


def compute_style_metrics(text: str) -> dict:
    tokens = _tokenize(text)
    return {
        "ttr":                  type_token_ratio(tokens),
        "avg_sentence_len":     avg_sentence_length(text),
        "argentinismo_score":   argentinismo_score(text),
        "emoji_density":        emoji_density(text),
        "interjection_density": interjection_density(text),
    }


def compare_styles(
    real_texts: list[str],
    generated_texts: list[str],
    register: str = "all",
) -> dict:
    real_m = [compute_style_metrics(t) for t in real_texts]
    gen_m  = [compute_style_metrics(t) for t in generated_texts]

    metrics = list(real_m[0].keys())
    results = {}

    print(f"\nComparación de estilo ({register}):")
    print(f"{'Métrica':<25} {'Real':>8} {'Generado':>10} {'Diff%':>8}")
    print("─" * 55)

    for metric in metrics:
        real_avg = sum(m[metric] for m in real_m) / len(real_m)
        gen_avg  = sum(m[metric] for m in gen_m)  / len(gen_m)
        diff_pct = abs(real_avg - gen_avg) / (real_avg + 1e-9) * 100
        flag = "✓" if diff_pct < 20 else "⚠"
        results[metric] = {
            "real": round(real_avg, 4),
            "generated": round(gen_avg, 4),
            "diff_pct": round(diff_pct, 2),
        }
        print(f"{metric:<25} {real_avg:>8.4f} {gen_avg:>10.4f} {diff_pct:>7.1f}% {flag}")

    return results
