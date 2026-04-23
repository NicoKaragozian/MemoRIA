# MemoRIA

Fine-tuning de Qwen 3 4B Instruct sobre textos personales para aprender estilo de escritura en tres registros: casual (WhatsApp), profesional (email) y académico.

**Equipo:** Nico Karagozian, Clara Kearney, Valen Pivotto — UdeSA MIA NLP 2026

---

## Pipeline

```
data/raw/
  whatsapp/*.txt ─┐
  gmail/*.mbox   ─┼─► parsers (sin anonimización) ─► data/processed/
  academic/      ─┘
       │
       ▼
  build_dataset.py  (formato chat MLX-LM, prompts del catálogo, split 80/10/10, dedup, manifest)
       │
       ▼
  data/dataset/train|valid|test.jsonl   ← schema: {messages: [system, user+[TAG], assistant]}
       │
       ▼
  mlx_lm.convert  (cuantización 4-bit → models/qwen3-4b-4bit/)
       │
       ▼
  mlx_lm.lora --mask-prompt --num-layers -1  (LoRA → memoria-lora/)
       │
       ▼
  merge + export GGUF Q4_K_M → memoria-q4.gguf
       │
       ▼
  ollama create memoria  ─►  FastAPI backend  ─►  demo web
```

---

## Inicio rápido (Mac Apple Silicon)

### 1. Prerequisitos

```bash
python3 -m venv venv && source venv/bin/activate
pip install mlx-lm>=0.22.0 transformers>=4.50.0 peft trl datasets \
    pdfplumber python-docx spacy nltk scikit-learn \
    fastapi httpx uvicorn slowapi html2text scipy
python3 -m spacy download es_core_news_sm
```

### 2. Variables de entorno

```bash
cp .env.example .env
# Editar .env: AUTHOR_NAME, AUTHOR_EMAIL
```

### 3. Datos personales

```
data/raw/whatsapp/*.txt    ← WhatsApp: chat → Exportar → Sin multimedia
data/raw/gmail/*.mbox      ← Google Takeout → Datos de Gmail
data/raw/academic/*.pdf    ← PDFs y DOCX propios
```

### 4. Login HuggingFace

```bash
huggingface-cli login
```

### 5. Ejecutar el notebook

```bash
jupyter notebook MemoRIA.ipynb
```

Ejecutar las celdas en orden (ver tabla en `CLAUDE.md`).

### 6. Levantar la demo

```bash
# Con Ollama corriendo y modelo registrado:
bash scripts/create_ollama_model.sh
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# O con Docker:
docker compose up
```

Abrir `http://127.0.0.1:8000`.

---

## Tests

```bash
pytest tests/
```

---

## Estructura

```
MemoRIA.ipynb               ← notebook principal
MEMORIA_PLAN.md             ← decisiones de diseño
scripts/                    ← parsers, dataset, fine-tuning, inferencia
eval/                       ← perplexidad, estilo, clasificador, test ciego
backend/                    ← FastAPI + frontend estático
data/prompts/               ← catálogos de prompts independientes por registro
data/system_prompts/        ← instrucciones de registro (casual/email_prof/academic)
tests/                      ← suite de tests automatizados
```

---

## Evaluaciones

| ID | Método | Métrica | Umbral mínimo |
|----|--------|---------|---------------|
| E1 | Perplexidad base vs. fine-tuneado | Mejora relativa | ≥ 20% |
| E2 | Métricas de estilo léxico-sintáctico + MAUVE | Diferencia promedio / score | < 20% / > 0.5 |
| E3 | Clasificador BETO de autoría | Accuracy | < 75% |
| E4 | Test ciego humano | % aciertos juez | < 65% |
| Sanity | Emisión de placeholders de anonimización | placeholder_emission_ratio | = 0% |
