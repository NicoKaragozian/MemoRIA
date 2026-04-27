"""
E4 — Generación de pares para test ciego humano.

Corrección: los pares usan prompts del catálogo independiente, NO el inicio
del texto real como seed. Esto mide estilo, no continuación.

Correcciones adicionales:
- Truncado al límite de oración (no corte duro que delata al generado)
- Dos archivos de salida: blind_test_pairs.json (sin labels, para jueces)
  y blind_test_key.json (con labels, gitignored)
- order aleatorio aplicado en la serialización
"""

import json
import re
from pathlib import Path


def _load_prompts(register: str) -> list[str]:
    fname = "email_prof" if register == "email_prof" else register
    path = Path("data/prompts") / f"{fname}.txt"
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
                result[reg].append(next(m["content"] for m in item["messages"] if m["role"] == "assistant"))
    return result


def _truncate_at_sentence(text: str, max_chars: int = 600) -> str:
    """
    Trunca en el último límite de oración (. ! ? párrafo) antes de max_chars.
    Evita cortes abruptos que delaten el texto generado al juez.
    """
    if len(text) <= max_chars:
        return text
    candidate = text[:max_chars]
    # Buscar el último . ! ? \n\n dentro del candidato
    match = None
    for m in re.finditer(r'[.!?](?:\s|$)|\n\n', candidate):
        match = m
    if match:
        return candidate[:match.end()].strip()
    return candidate.rstrip()


def generate_blind_test_pairs(
    test_file: str,
    generate_fn,
    n_per_register: int = 10,
    output_file: str = "eval/blind_test_pairs.json",
    seed: int = 42,
):
    """
    Genera pares para test ciego humano.

    Produce dos archivos:
    - blind_test_pairs.json : para los jueces — texto_a / texto_b sin etiquetas
    - blind_test_key.json   : ground truth — cuál era real en cada par (gitignored)

    generate_fn(register, prompt) → str — debe usar el modelo fine-tuneado.

    Protocolo sugerido:
    - 30 pares totales (10/registro)
    - Pregunta: "¿Cuál de estos dos textos escribió Nico?" + confianza 1-5
    - Si % correctas < 60% → modelo pasó el test
    """
    import random
    random.seed(seed)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    key_path = output_path.parent / "blind_test_key.json"

    real_by_register = _load_test_texts(test_file)
    judge_pairs = []
    key_pairs   = []

    for register in ("casual", "email_prof", "academic"):
        real_texts = real_by_register[register]
        prompts    = _load_prompts(register)

        real_sample   = random.sample(real_texts, min(n_per_register, len(real_texts)))
        prompt_sample = random.sample(prompts, min(n_per_register, len(prompts)))

        for real_text, prompt in zip(real_sample, prompt_sample):
            generated = generate_fn(register, prompt)
            real_trunc = _truncate_at_sentence(real_text)
            gen_trunc  = _truncate_at_sentence(generated)

            order = random.choice(["real_first", "gen_first"])
            if order == "real_first":
                text_a, text_b = real_trunc, gen_trunc
                real_position  = "a"
            else:
                text_a, text_b = gen_trunc, real_trunc
                real_position  = "b"

            pair_id = f"{register}_{len(judge_pairs):03d}"
            judge_pairs.append({
                "id":       pair_id,
                "register": register,
                "text_a":   text_a,
                "text_b":   text_b,
            })
            key_pairs.append({
                "id":           pair_id,
                "register":     register,
                "real_position": real_position,
                "prompt_used":  prompt,
            })

    random.shuffle(judge_pairs)
    # Reordenar key_pairs con el mismo orden que judge_pairs
    id_to_key = {kp["id"]: kp for kp in key_pairs}
    ordered_keys = [id_to_key[jp["id"]] for jp in judge_pairs]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(judge_pairs, f, ensure_ascii=False, indent=2)

    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(ordered_keys, f, ensure_ascii=False, indent=2)

    print(f"✓ {len(judge_pairs)} pares generados en {output_path}")
    print(f"  Ground truth en {key_path} (NO compartir con jueces)")
    print("  Pregunta para jueces: '¿Cuál de estos dos textos (A o B) escribió Nico?' + confianza 1-5")
    return judge_pairs
