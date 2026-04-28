# MemoRIA

Chatbot personal que aprende a responder en el estilo del usuario a partir de sus propias conversaciones. Dado un contexto conversacional (chat 1:1 o grupal) y un mensaje recibido, genera la respuesta que el usuario hubiera escrito.

Fine-tuning con LoRA sobre Gemma 3 4B en Apple Silicon.

**Equipo:** Nico Karagozian, Clara Kearney, Valen Pivotto — UdeSA MIA NLP 2026

> **Estado actual (rama `feature/chatbot-conversacional`)**: pipeline de datos y entrenamiento end-to-end funcionando. UI y backend en proceso de adaptación al nuevo formato. Detalle en [docs/CHATBOT_DESIGN.md](docs/CHATBOT_DESIGN.md).

---

## Pipeline

```
data/raw/whatsapp/*.txt
       │
       ▼
parse_whatsapp.py        (extrae pares conversacionales: contexto + target)
       │
       ▼
data/processed/whatsapp_pairs.jsonl
       │
       ▼
build_dataset.py         (chat template Gemma, dedup, split estratificado)
       │
       ▼
data/dataset/{train,val,test}.jsonl
       │
       ▼
mlx_lm.convert           (cuantización 4-bit → models/gemma3-4b-4bit/)
       │
       ▼
mlx_lm.lora              (fine-tuning LoRA → memoria-lora/)
       │
       ▼
merge + export GGUF      → memoria-q4.gguf
       │
       ▼
ollama create memoria  ─►  FastAPI backend  ─►  demo web
```

Cada par de entrenamiento tiene: nombre del chat, lista de participantes, los últimos N mensajes anteriores con su autor, y el turno del usuario como target. Ver [docs/CHATBOT_DESIGN.md](docs/CHATBOT_DESIGN.md) para el detalle de segmentación, anonimización y filtros.

---

## Inicio rápido (Mac Apple Silicon)

### 1. Prerequisitos

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r backend/requirements-backend.txt
python -m spacy download es_core_news_sm
```

### 2. Variables de entorno

```bash
cp .env.example .env
# Editar .env: AUTHOR_NAME (cómo aparecés en los chats), AUTHOR_EMAIL
```

### 3. Datos personales

```
data/raw/whatsapp/*.txt    ← WhatsApp: chat → Exportar → Sin multimedia
```

(`data/raw/`, `data/processed/`, `data/dataset/`, `models/`, `memoria-lora/` y `memoria-*.gguf` están en `.gitignore` — los datos quedan solo en tu máquina.)

> **Sobre los registros (casual / profesional / académico):** en esta iteración el dataset es solo WhatsApp. Email y documentos académicos se incorporan en iteraciones siguientes. A diferencia de la iteración anterior, el modelo **no recibe una etiqueta explícita de registro** — aprende a modular el estilo implícitamente a partir del contexto (nombre del chat, participantes, mensajes previos). Cuando se sumen emails y académicos, el mismo mecanismo va a aprender esos patrones sin necesidad de etiquetar la fuente.

### 4. Login HuggingFace (Gemma es gated)

```bash
# Aceptar el license en https://huggingface.co/google/gemma-3-4b-it
huggingface-cli login
```

### 5. Generar dataset y entrenar

```bash
# Genera data/processed/whatsapp_pairs.jsonl + data/dataset/{train,val,test}.jsonl
python -m scripts.build_dataset

# Cuantiza Gemma 3 4B a 4-bit (~3-15 min, una sola vez) y entrena LoRA
bash scripts/finetune_mlx.sh 2000   # 2000 iters
```

### 6. Levantar la demo

```bash
bash scripts/merge_mlx.sh             # mergea LoRA → memoria-merged/
bash scripts/export_gguf.sh           # exporta a GGUF y registra en Ollama
uvicorn backend.main:app --host 127.0.0.1 --port 8000
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
docs/
  CHATBOT_DESIGN.md           ← decisiones de diseño + estado del rediseño
  OBSERVABILIDAD_EVALS.md     ← propuesta de capa de evals (no implementada)
scripts/
  parse_whatsapp.py           ← parser conversacional (pares contexto+target)
  build_dataset.py            ← chat template, dedup, split, manifest
  anonymize.py                ← PII en contenido (nombres de autores se preservan)
  finetune_mlx.sh             ← MLX-LM LoRA
  merge_mlx.sh, export_gguf.sh, create_ollama_model.sh
backend/                      ← FastAPI + frontend estático
eval/                         ← métricas de estilo léxico-sintáctico
tests/                        ← suite de tests automatizados
data/prompts/                 ← catálogos de prompts (de la iteración anterior)
MemoRIA.ipynb                 ← notebook con el flujo de la iteración anterior
```

---

## Evaluación

La iteración anterior usó perplexidad, métricas de estilo, clasificador BETO de autoría y test ciego humano (ver `eval/`). El rediseño conversacional plantea una capa nueva de evals (LLM-as-judge sobre dimensiones definidas por el usuario, triangulación con evaluadores externos) — propuesta en [docs/OBSERVABILIDAD_EVALS.md](docs/OBSERVABILIDAD_EVALS.md).
