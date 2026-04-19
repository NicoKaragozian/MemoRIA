"""
E2 — Métricas de estilo léxico-sintácticas.

Correcciones vs. versión original:
- Argentinismos usan \\b...\\b para evitar match en substrings (re vs. presente)
- "tipo" eliminado de argentinismos (demasiados falsos positivos en académico)
- Rango de emojis extendido a suplementarios modernos
- diff_pct devuelve None cuando real == 0 (evita 1e11%)
- nltk.download movido a función lazy (sin side effects en import)
- Bootstrap CI y Mann-Whitney U para significancia estadística
"""

import re
from collections import Counter

import numpy as np
from scipy import stats


def _ensure_nltk():
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    return set(stopwords.words("spanish"))


_stopwords_cache: set | None = None


def _get_stopwords() -> set:
    global _stopwords_cache
    if _stopwords_cache is None:
        _stopwords_cache = _ensure_nltk()
    return _stopwords_cache


ARGENTINISMOS = [
    "boludo", "che", "posta", "dale", "copado", "quilombo", "laburo",
    "laburar", "loco", "banda", "chabón", "pibe", "fiaca", "bardear",
    "morfar", "chamuyo", "trucho", "zarpado", "groso", "onda",
    "joya", "plata", "guita",
]

INTERJECCIONES = ["jaja", "jeje", "jsjs", "ajá", "bah", "uy", "ay", "eh",
                  "mm", "uff", "jua", "xd", "lol"]

_ARGENTINISMO_RE = [re.compile(r'\b' + a + r'\b', re.IGNORECASE) for a in ARGENTINISMOS]
_INTERJ_RE       = [re.compile(r'\b' + i + r'\b', re.IGNORECASE) for i in INTERJECCIONES]

# Rango extendido: básicos + suplementarios (🤣🤷🧠) + simbólicos modernos
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"    # símbolos y pictogramas
    "\U0001F680-\U0001F6FF"    # transporte y mapas
    "\U0001F1E0-\U0001F1FF"    # banderas
    "\U0001F900-\U0001F9FF"    # suplementarios (🤣🤷🧠)
    "\U0001FA70-\U0001FAFF"    # simbólicos extendidos-A (🪄🫀)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251]+",
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
    """Frecuencia de argentinismos por 100 palabras (sin "tipo" — demasiados FP en académico)."""
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


def _bootstrap_diff(real_vals: np.ndarray, gen_vals: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Devuelve (lower_95, upper_95) del diff promedio por bootstrap."""
    rng = np.random.default_rng(seed)
    diffs = []
    n_r, n_g = len(real_vals), len(gen_vals)
    for _ in range(n_boot):
        r = rng.choice(real_vals, size=n_r, replace=True).mean()
        g = rng.choice(gen_vals, size=n_g, replace=True).mean()
        diffs.append(abs(r - g))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


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
    print(f"{'Métrica':<25} {'Real':>8} {'Gen':>8} {'Diff%':>7} {'p-val':>7} {'CI_lo':>7} {'CI_hi':>7}")
    print("─" * 75)

    for metric in metrics:
        real_vals = np.array([m[metric] for m in real_m])
        gen_vals  = np.array([m[metric] for m in gen_m])
        real_avg  = float(real_vals.mean())
        gen_avg   = float(gen_vals.mean())

        if real_avg == 0:
            diff_pct = None
            diff_str = "  N/A"
        else:
            diff_pct = abs(real_avg - gen_avg) / real_avg * 100
            diff_str = f"{diff_pct:>6.1f}%"

        # Mann-Whitney U (no paramétrico)
        try:
            _, p_val = stats.mannwhitneyu(real_vals, gen_vals, alternative="two-sided")
            p_str = f"{p_val:.3f}"
        except Exception:
            p_val, p_str = None, "  N/A"

        ci_lo, ci_hi = _bootstrap_diff(real_vals, gen_vals)

        flag = ""
        if diff_pct is not None:
            flag = "✓" if diff_pct < 20 else "⚠"

        results[metric] = {
            "real": round(real_avg, 4),
            "generated": round(gen_avg, 4),
            "diff_pct": round(diff_pct, 2) if diff_pct is not None else None,
            "p_value": round(p_val, 4) if p_val is not None else None,
            "bootstrap_ci_diff_95": [round(ci_lo, 4), round(ci_hi, 4)],
        }
        print(
            f"{metric:<25} {real_avg:>8.4f} {gen_avg:>8.4f} "
            f"{diff_str:>7} {p_str:>7} {ci_lo:>7.4f} {ci_hi:>7.4f} {flag}"
        )

    return results
