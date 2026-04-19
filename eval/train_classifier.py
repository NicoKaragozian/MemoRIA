"""
E3 — Clasificador de autoría con BETO.

Corrección: las generaciones se producen con prompts INDEPENDIENTES del test set
(catálogo data/prompts/*.txt) para no medir continuación sino estilo.

Correcciones adicionales:
- set_all_seeds para reproducibilidad
- Stratify por (label, register) compuesto
- IC 95% Wilson para accuracy con muestras pequeñas
- Cache de textos generados para evitar re-generar en cada corrida
"""

import json
import logging
from math import sqrt
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from scripts.seed import set_all_seeds

logger = logging.getLogger(__name__)

BETO_MODEL = "dccuchile/bert-base-spanish-wwm-uncased"
_SEED = 42


def _load_prompts(register: str) -> list[str]:
    fname = "email_prof" if register == "email_prof" else register
    path = Path("data/prompts") / f"{fname}.txt"
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text("utf-8").splitlines() if l.strip()]


def _cache_path(generate_fn, register: str) -> Path:
    import hashlib
    fn_hash = hashlib.md5(str(generate_fn).encode()).hexdigest()[:8]
    cache_dir = Path("eval/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"generated_{register}_{fn_hash}.json"


def _generate_with_cache(generate_fn, register: str, prompts: list[str]) -> list[str]:
    cp = _cache_path(generate_fn, register)
    if cp.exists():
        logger.info("Usando cache de generaciones: %s", cp)
        return json.loads(cp.read_text())
    texts = [generate_fn(register, p) for p in prompts]
    cp.write_text(json.dumps(texts, ensure_ascii=False))
    return texts


def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza 95% Wilson para proporción binomial."""
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


class AuthorshipDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.enc = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def _generate_texts_for_classifier(generate_fn, n_per_register: int = 50) -> tuple[list[str], list[str]]:
    """
    Genera textos con prompts del catálogo. Devuelve (texts, registers).
    """
    import random
    texts, registers = [], []
    for register in ("casual", "email_prof", "academic"):
        prompts = _load_prompts(register)
        sample = random.sample(prompts, min(n_per_register, len(prompts)))
        generated = _generate_with_cache(generate_fn, register, sample)
        texts.extend(generated)
        registers.extend([register] * len(generated))
    return texts, registers


def train_authorship_classifier(
    real_texts: list[str],
    generate_fn,
    real_registers: list[str] | None = None,
    output_dir: str = "./authorship-classifier",
    n_samples: int = 150,
):
    """
    Entrena BETO para distinguir texto real (0) de generado (1).
    generate_fn(register, prompt) → str — usa prompts del catálogo.

    Interpretación de accuracy:
    > 80% → modelo distinguible → fine-tuning insuficiente
    60-80% → parcialmente distinguible
    < 60% → indistinguible → fine-tuning exitoso ✓
    """
    set_all_seeds(_SEED)

    generated_texts, gen_registers = _generate_texts_for_classifier(generate_fn, n_per_register=50)

    n = min(len(real_texts), len(generated_texts), n_samples)
    texts  = real_texts[:n] + generated_texts[:n]
    labels = [0] * n + [1] * n
    # Strata por (label, register) para split estratificado
    regs_real = (real_registers or ["unknown"] * len(real_texts))[:n]
    regs_gen  = gen_registers[:n]
    strata = [f"{l}-{r}" for l, r in zip(labels, regs_real + regs_gen)]

    X_tr, X_val, y_tr, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=_SEED, stratify=strata,
    )
    logger.info("Train: %d | Val: %d", len(X_tr), len(X_val))
    print(f"Train: {len(X_tr)} | Val: {len(X_val)}")

    tokenizer  = AutoTokenizer.from_pretrained(BETO_MODEL)
    train_ds   = AuthorshipDataset(X_tr, y_tr, tokenizer)
    val_ds     = AuthorshipDataset(X_val, y_val, tokenizer)
    model      = AutoModelForSequenceClassification.from_pretrained(BETO_MODEL, num_labels=2)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        seed=_SEED,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    preds = trainer.predict(val_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    acc = accuracy_score(y_val, y_pred)
    n_correct = int(sum(p == t for p, t in zip(y_pred, y_val)))
    ci_lo, ci_hi = wilson_ci(n_correct, len(y_val))

    print("\n" + "="*55)
    print("RESULTADO DEL CLASIFICADOR DE AUTORÍA (BETO)")
    print("="*55)
    print(classification_report(y_val, y_pred, target_names=["Real", "Generado"]))
    print(f"Accuracy: {acc:.3f}  [IC 95% Wilson: {ci_lo:.3f}–{ci_hi:.3f}]")
    print(f"(n={len(y_val)} — interpretar con cuidado con muestras pequeñas)")

    if acc < 0.60:
        print("✓ EXCELENTE: El modelo es indistinguible del texto real")
    elif acc < 0.75:
        print("~ ACEPTABLE: Parcialmente distinguible")
    else:
        print("✗ INSUFICIENTE: El texto generado es claramente distinguible")

    return {"accuracy": acc, "ci_low_95": ci_lo, "ci_high_95": ci_hi}
