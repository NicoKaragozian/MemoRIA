"""
E4 — Generación de pares para test ciego humano.

Corrección: los pares usan prompts del catálogo independiente, NO el inicio
del texto real como seed. Esto mide estilo, no continuación.
"""

import json
import random
from pathlib import Path


def _load_prompts(register: str) -> list[str]:
    path = Path("data/prompts") / f"{register}.txt"
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text("utf-8").splitlines() if l.strip()]


def _load_test_texts(test_file: str) -> dict[str, list[str]]:
    result = {"casual": [], "email_prof": [], "academic": []}
    with open(test_file) as f:
        for line in f:
            item = json.loads(line)
            reg = item["register"]
            if reg in result:
                result[reg].append(item["original_text"])
    return result


def generate_blind_test_pairs(
    test_file: str,
    generate_fn,
    n_per_register: int = 10,
    output_file: str = "eval/blind_test_pairs.json",
    seed: int = 42,
):
    """
    Genera pares (texto_real, texto_generado) para el test ciego.

    generate_fn(register, prompt) → str — debe usar el modelo fine-tuneado.
    Los prompts son del catálogo independiente (data/prompts/*.txt).

    Protocolo sugerido para jueces:
    - 30 pares × 3 registros = 90 total
    - Pregunta: "¿Cuál de estos dos textos escribió Nico?"
    - Confianza: 1-5
    - Si % correctas < 60% → modelo pasó el test
    """
    random.seed(seed)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    real_by_register = _load_test_texts(test_file)
    pairs = []

    for register in ("casual", "email_prof", "academic"):
        real_texts = real_by_register[register]
        prompts    = _load_prompts(register)

        real_sample   = random.sample(real_texts, min(n_per_register, len(real_texts)))
        prompt_sample = random.sample(prompts, min(n_per_register, len(prompts)))

        for real_text, prompt in zip(real_sample, prompt_sample):
            generated = generate_fn(register, prompt)
            pairs.append({
                "register": register,
                "prompt_used": prompt,
                "real": real_text[:600],
                "generated": generated[:600],
                "order": random.choice(["real_first", "gen_first"]),
            })

    random.shuffle(pairs)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    print(f"✓ {len(pairs)} pares generados en {output_file}")
    print("  Usar estos pares en Google Forms / Typeform para el test ciego.")
    print("  Pregunta: '¿Cuál de estos dos textos escribió Nico?' + confianza 1-5")
    return pairs
