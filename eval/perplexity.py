"""
E1 — Perplexidad sobre el test set de textos reales.

Bug corregido vs. versión original: labels[attention_mask == 0] = -100
para que la loss no compute sobre tokens de padding.
"""

import math
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "google/gemma-4-E2B-it"


def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 4,
) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        # Enmascarar tokens de padding en labels para no computar loss sobre ellos
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=labels,
            )
            nll = outputs.loss.item()
            n_tokens = enc["attention_mask"].sum().item()
            total_nll += nll * n_tokens
            total_tokens += n_tokens

    return math.exp(total_nll / total_tokens)


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
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("\n[1/2] Modelo BASE...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    ppl_base = compute_perplexity(base_model, tokenizer, texts)
    print(f"  Perplexidad base: {ppl_base:.2f}")
    del base_model
    if device == "mps":
        torch.mps.empty_cache()

    if adapter_path:
        print("\n[2/2] Modelo FINE-TUNEADO...")
        ft_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map={"": device}
        )
        ft_model = PeftModel.from_pretrained(ft_model, adapter_path)
        ppl_ft = compute_perplexity(ft_model, tokenizer, texts)
        print(f"  Perplexidad fine-tuneado: {ppl_ft:.2f}")

        improvement = (ppl_base - ppl_ft) / ppl_base * 100
        print(f"\n  Mejora relativa: {improvement:.1f}%")
        print(f"  {'✓ EXITOSO (≥20%)' if improvement >= 20 else '⚠ INSUFICIENTE (<20%)'}")

        return {"base": ppl_base, "finetuned": ppl_ft, "improvement_pct": improvement}

    return {"base": ppl_base}


if __name__ == "__main__":
    results = eval_perplexity(
        test_file="data/dataset/test.jsonl",
        model_id=MODEL_ID,
        adapter_path="./memoria-lora",
    )
    print(f"\nResultados: {results}")
