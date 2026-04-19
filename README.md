# MemoRIA

Fine-tuning de Gemma 4 E2B sobre textos personales para aprender estilo de escritura en tres registros: casual (WhatsApp), profesional (email) y académico.

**Equipo:** Nico Karagozian, Clara Kearney, Valen Pivotto — UdeSA MIA NLP 2026

---

## Pipeline

```
data/raw/
  whatsapp/*.txt ─┐
  gmail/*.mbox   ─┼─► parsers ─► anonymize ─► data/processed/
  academic/      ─┘
       │
       ▼
  build_dataset.py  (prompts del catálogo, split 80/10/10, dedup, manifest)
       │
       ▼
  data/dataset/train|val|test.jsonl
       │
       ▼
  mlx_lm.convert  (cuantización 4-bit → models/gemma4-e2b-4bit/)
       │
       ▼
  mlx_lm.lora     (fine-tuning LoRA → memoria-lora/)
       │
       ▼
  merge + export GGUF → memoria-q4.gguf
       │
       ▼
  ollama create memoria  ─►  FastAPI backend  ─►  demo web
```

---

## Inicio rápido (Mac Apple Silicon)

### 1. Prerequisitos

```bash
python -m venv venv && source venv/bin/activate
pip install mlx-lm>=0.22.0 transformers>=4.50.0 peft trl datasets \
    pdfplumber python-docx spacy nltk scikit-learn \
    fastapi httpx uvicorn slowapi html2text scipy
python -m spacy download es_core_news_sm
```

### 2. Variables de entorno

```bash
cp .env.example .env
# Editar .env: AUTHOR_NAME, AUTHOR_EMAIL, MODEL_ID
```

### 3. Datos personales

```
data/raw/whatsapp/*.txt    ← WhatsApp: chat → Exportar → Sin multimedia
data/raw/gmail/*.mbox      ← Google Takeout → Datos de Gmail
data/raw/academic/*.pdf    ← PDFs y DOCX propios
```

### 4. Login HuggingFace (Gemma es gated)

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
MemoRIA.ipynb          ← notebook principal
MEMORIA_PLAN.md        ← decisiones de diseño
scripts/               ← parsers, anonimización, dataset, fine-tuning
eval/                  ← perplexidad, estilo, clasificador, test ciego
backend/               ← FastAPI + frontend estático
data/prompts/          ← catálogos de prompts independientes por registro
tests/                 ← suite de tests automatizados
```

---

## Evaluaciones

| ID | Método | Métrica | Umbral OK |
|----|--------|---------|-----------|
| E1 | Perplexidad base vs. fine-tuneado | Mejora relativa | ≥ 20% |
| E2 | Métricas de estilo léxico-sintáctico | Diferencia promedio | < 20% |
| E3 | Clasificador BETO de autoría | Accuracy | < 75% |
| E4 | Test ciego humano | % aciertos juez | < 65% |
