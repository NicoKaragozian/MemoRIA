"""
E1 — Perplexidad sobre el test set de textos reales.

Usa MLX-LM (Apple Silicon) en lugar de PyTorch/PEFT, compatible con
los adaptadores LoRA generados por mlx_lm.lora.

Bootstrap CI (N=1000) para reportar PPL con intervalo de confianza 95%.
"""

import json
import logging
import math

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _compute_example_nll_mlx(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
) -> list[tuple[float, int]]:
    """
    Devuelve lista de (nll_sum, n_tokens) por ejemplo usando MLX.
    """
    import mlx.core as mx
    import mlx.nn as nn

    model.eval()
    results = []

    for text in tqdm(texts, desc="Perplexity"):
        tokens = tokenizer.encode(text)
        if len(tokens) > max_length:
            logger.warning(
                "Texto truncado: %d tokens → %d", len(tokens), max_length
            )
            tokens = tokens[:max_length]

        if len(tokens) < 2:
            continue

        input_ids = mx.array([tokens])
        # Forward pass
        logits = model(input_ids)
        # Shift: predecir token t+1 dado t
        shift_logits = logits[:, :-1, :]   # (1, T-1, vocab)
        shift_labels = mx.array([tokens[1:]])  # (1, T-1)

        # Cross-entropy token a token
        loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="none",
        )
        mx.eval(loss)
        nll_sum = float(loss.sum().item())
        n_tok = len(tokens) - 1
        results.append((nll_sum, n_tok))

    return results


def _bootstrap_ppl(
    example_nlls: list[tuple[float, int]],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Devuelve (mediana, lower_95, upper_95) de PPL por bootstrap."""
    rng = np.random.default_rng(seed)
    nlls = np.array([n for n, _ in example_nlls])
    ns   = np.array([k for _, k in example_nlls])
    ppls = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(nlls), size=len(nlls))
        ppls.append(np.exp(nlls[idx].sum() / ns[idx].sum()))
    ppls = np.array(ppls)
    return float(np.median(ppls)), float(np.percentile(ppls, 2.5)), float(np.percentile(ppls, 97.5))


def compute_perplexity_mlx(model, tokenizer, texts: list[str]) -> dict:
    """Devuelve dict con 'ppl', 'ci_low', 'ci_high' (bootstrap 95%)."""
    example_nlls = _compute_example_nll_mlx(model, tokenizer, texts)
    total_nll = sum(n for n, _ in example_nlls)
    total_tok = sum(k for _, k in example_nlls)
    ppl = math.exp(total_nll / total_tok)
    med, lo, hi = _bootstrap_ppl(example_nlls)
    return {"ppl": ppl, "ppl_bootstrap_median": med, "ci_low_95": lo, "ci_high_95": hi}


def eval_perplexity(
    test_file: str,
    model_id: str = "models/gemma3-4b-4bit",
    adapter_path: str = None,
):
    """
    Evalúa perplexidad del modelo base y fine-tuneado usando MLX-LM.

    Args:
        test_file:    Path al test.jsonl del dataset.
        model_id:     Path local al modelo base cuantizado (MLX format).
        adapter_path: Path al directorio con adapters.safetensors (MLX LoRA).
    """
    from mlx_lm import load

    # Extraer textos del turno assistant
    texts = []
    with open(test_file) as f:
        for line in f:
            item = json.loads(line)
            assistant_text = next(
                (m["content"] for m in item["messages"] if m["role"] == "assistant"),
                None,
            )
            if assistant_text:
                texts.append(assistant_text)

    print(f"Evaluando perplexidad sobre {len(texts)} textos del test set")

    # ── Modelo BASE ──────────────────────────────────────────────────────────
    print("\n[1/2] Modelo BASE...")
    base_model, tokenizer = load(model_id)
    res_base = compute_perplexity_mlx(base_model, tokenizer, texts)
    print(
        f"  PPL base: {res_base['ppl']:.2f} "
        f"[IC 95%: {res_base['ci_low_95']:.2f}–{res_base['ci_high_95']:.2f}]"
    )
    del base_model

    # ── Modelo FINE-TUNEADO ───────────────────────────────────────────────────
    if adapter_path:
        print("\n[2/2] Modelo FINE-TUNEADO...")
        ft_model, tokenizer = load(model_id, adapter_path=adapter_path)
        res_ft = compute_perplexity_mlx(ft_model, tokenizer, texts)
        print(
            f"  PPL fine-tuneado: {res_ft['ppl']:.2f} "
            f"[IC 95%: {res_ft['ci_low_95']:.2f}–{res_ft['ci_high_95']:.2f}]"
        )

        improvement = (res_base["ppl"] - res_ft["ppl"]) / res_base["ppl"] * 100
        print(f"\n  Mejora relativa: {improvement:.1f}%")
        print(f"  {'✓ EXITOSO (≥20%)' if improvement >= 20 else '⚠ INSUFICIENTE (<20%)'}")

        return {
            "base": res_base,
            "finetuned": res_ft,
            "improvement_pct": round(improvement, 2),
        }

    return {"base": res_base}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = eval_perplexity(
        test_file="data/dataset/test.jsonl",
        model_id="models/gemma3-4b-4bit",
        adapter_path="memoria-lora",
    )
    print(f"\nResultados: {results}")