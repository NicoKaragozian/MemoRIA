"""
E1 — Perplexidad sobre el test set de textos reales.

Correcciones vs. versión original:
- Usa CrossEntropyLoss(reduction="none") para acumular NLL exacto (no sesgado por shift)
- Enmascara tokens de padding con labels=-100
- Warning cuando un texto supera max_length y se trunca
- Bootstrap CI (N=1000) para reportar PPL con intervalo de confianza 95%
"""

import json
import logging
import math

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)

MODEL_ID = "google/gemma-4-E2B-it"


def _compute_example_nll(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 4,
    max_length: int = 512,
) -> list[tuple[float, int]]:
    """
    Devuelve lista de (nll_sum, n_tokens) por ejemplo.
    Usa reduction="none" para acumulación exacta sin el sesgo del shift.
    """
    model.eval()
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    results = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        for j, (orig, trunc_ids) in enumerate(zip(batch, enc["input_ids"])):
            orig_len = len(tokenizer.encode(orig, add_special_tokens=False))
            if orig_len > max_length:
                logger.warning(
                    "Texto truncado: %d tokens → %d (perdiendo %d tokens)",
                    orig_len, max_length, orig_len - max_length,
                )

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )

        # Calcular loss token a token sin el sesgo del promedio de HF
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)

        token_losses = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        for j in range(len(batch)):
            valid = (shift_labels[j] != -100)
            nll_sum = token_losses[j][valid].sum().item()
            n_tok   = valid.sum().item()
            if n_tok > 0:
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
        idx  = rng.integers(0, len(nlls), size=len(nlls))
        ppls.append(np.exp(nlls[idx].sum() / ns[idx].sum()))
    ppls = np.array(ppls)
    return float(np.median(ppls)), float(np.percentile(ppls, 2.5)), float(np.percentile(ppls, 97.5))


def compute_perplexity(model, tokenizer, texts: list[str], batch_size: int = 4) -> dict:
    """
    Devuelve dict con 'ppl', 'ci_low', 'ci_high' (bootstrap 95%).
    """
    example_nlls = _compute_example_nll(model, tokenizer, texts, batch_size)
    total_nll = sum(n for n, _ in example_nlls)
    total_tok = sum(k for _, k in example_nlls)
    ppl = math.exp(total_nll / total_tok)
    med, lo, hi = _bootstrap_ppl(example_nlls)
    return {"ppl": ppl, "ppl_bootstrap_median": med, "ci_low_95": lo, "ci_high_95": hi}


def eval_perplexity(
    test_file: str,
    model_id: str = MODEL_ID,
    adapter_path: str = None,
):
    texts = []
    with open(test_file) as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["original_text"])

    print(f"Evaluando perplexidad sobre {len(texts)} textos del test set")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, extra_special_tokens={})

    print("\n[1/2] Modelo BASE...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    res_base = compute_perplexity(base_model, tokenizer, texts)
    print(f"  PPL base: {res_base['ppl']:.2f} [IC 95%: {res_base['ci_low_95']:.2f}–{res_base['ci_high_95']:.2f}]")
    del base_model
    if device == "mps":
        torch.mps.empty_cache()

    if adapter_path:
        print("\n[2/2] Modelo FINE-TUNEADO...")
        ft_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map={"": device}
        )
        ft_model = PeftModel.from_pretrained(ft_model, adapter_path)
        res_ft = compute_perplexity(ft_model, tokenizer, texts)
        print(f"  PPL fine-tuneado: {res_ft['ppl']:.2f} [IC 95%: {res_ft['ci_low_95']:.2f}–{res_ft['ci_high_95']:.2f}]")

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
        model_id=MODEL_ID,
        adapter_path="./memoria-lora",
    )
    print(f"\nResultados: {results}")
