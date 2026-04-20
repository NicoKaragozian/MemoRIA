"""
Inferencia con el modelo fine-tuneado (path HuggingFace MPS/CPU).
Para Mac M5, preferir `mlx_lm.generate` desde la línea de comandos.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID     = "google/gemma-3-4b-it"
ADAPTER_PATH = "./memoria-lora"

TAG_MAP = {
    "casual":     "[CASUAL]",
    "email_prof": "[EMAIL-PROF]",
    "academic":   "[ACADÉMICO]",
}


def load_model(model_id: str = MODEL_ID, adapter_path: str = ADAPTER_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_id, extra_special_tokens={})
    tokenizer.padding_side = "left"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    register: str,
    prompt: str,
    max_new_tokens: int = 300,
) -> str:
    tag = TAG_MAP.get(register, "[CASUAL]")
    instruction = f"{tag} {prompt}"

    chat = [{"role": "user", "content": instruction}]
    input_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    print("Cargando modelo fine-tuneado (HF path)...")
    model, tokenizer = load_model()

    test_cases = [
        ("casual",     "Contame cómo estuvo el finde"),
        ("email_prof", "Escribí un email pidiendo una reunión para el martes"),
        ("academic",   "Explicá el concepto de fine-tuning en NLP"),
    ]

    for register, prompt in test_cases:
        print(f"\n{'='*65}")
        print(f"Registro: {register.upper()} | Prompt: {prompt}")
        print("─" * 65)
        print(generate(model, tokenizer, register, prompt))
