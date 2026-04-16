"""
E3 — Clasificador de autoría con BETO.

Corrección: las generaciones se producen con prompts INDEPENDIENTES del test set
(catálogo data/prompts/*.txt) para no medir continuación sino estilo.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

BETO_MODEL = "dccuchile/bert-base-spanish-wwm-uncased"


def _load_prompts(register: str) -> list[str]:
    path = Path("data/prompts") / f"{register}.txt"
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text("utf-8").splitlines() if l.strip()]


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


def _generate_texts_for_classifier(generate_fn, n_per_register: int = 50) -> list[str]:
    """
    Genera textos con prompts del catálogo independiente (no del test set).
    generate_fn(register, prompt) → str
    """
    texts = []
    for register in ("casual", "email_prof", "academic"):
        prompts = _load_prompts(register)
        sample = random.sample(prompts, min(n_per_register, len(prompts)))
        for prompt in sample:
            texts.append(generate_fn(register, prompt))
    return texts


def train_authorship_classifier(
    real_texts: list[str],
    generate_fn,
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
    generated_texts = _generate_texts_for_classifier(generate_fn, n_per_register=50)

    n = min(len(real_texts), len(generated_texts), n_samples)
    texts  = real_texts[:n] + generated_texts[:n]
    labels = [0] * n + [1] * n

    X_tr, X_val, y_tr, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
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

    print("\n" + "="*50)
    print("RESULTADO DEL CLASIFICADOR DE AUTORÍA (BETO)")
    print("="*50)
    print(classification_report(y_val, y_pred, target_names=["Real", "Generado"]))
    print(f"Accuracy: {acc:.3f}")

    if acc < 0.60:
        print("✓ EXCELENTE: El modelo es indistinguible del texto real")
    elif acc < 0.75:
        print("~ ACEPTABLE: Parcialmente distinguible")
    else:
        print("✗ INSUFICIENTE: El texto generado es claramente distinguible")

    return acc
