# MemoRIA — Contexto para Claude

**Proyecto:** Fine-tuning de Gemma 3 4B sobre textos personales de Nico para aprender su estilo de escritura en tres registros: casual (WhatsApp), profesional (email) y académico.

**Equipo:** Nico Karagozian, Clara Kearney, Valen Pivotto — UdeSA MIA NLP 2026  
**Profesor:** Luciano Del Corro (pidió especialmente rigor en evaluación)

---

## Modelo

- **HuggingFace:** `google/gemma-3-4b-it`
- **Ollama:** `memoria` (modelo fine-tuneado, tag local)
- **Tamaño:** 2.3B params efectivos / 5.1B totales con embeddings
- **Context:** 128K tokens
- En 16 GB unified RAM requiere cuantización 4-bit antes de entrenar. Sin 4-bit → OOM.
- El modelo fine-tuneado **no publicar** — puede haber señales de PII aunque el pipeline de anonimización haya corrido.

---

## Estructura del proyecto

```
TP/
├── MemoRIA.ipynb              ← notebook principal, ejecutar en orden
├── MEMORIA_PLAN.md            ← plan completo con decisiones de diseño
├── README.md                  ← inicio rápido y diagrama del pipeline
├── pytest.ini                 ← configuración de tests
├── .env.example               ← plantilla de variables de entorno
├── .dockerignore              ← excluye modelos/datos/secretos del build
├── Modelfile                  ← para Docker (FROM /models/memoria.gguf)
├── docker-compose.yml
├── data/
│   ├── raw/                   ← gitignored; datos originales
│   │   ├── whatsapp/          ← .txt de exports de WhatsApp
│   │   ├── gmail/             ← .mbox de Google Takeout
│   │   └── academic/          ← PDFs y DOCX propios
│   ├── processed/             ← generado por los parsers (gitignored)
│   ├── dataset/               ← train/val/test.jsonl + manifest.json (gitignored)
│   └── prompts/               ← catálogos de prompts por registro (casual/email_prof/academic.txt)
├── scripts/
│   ├── parse_whatsapp.py
│   ├── parse_gmail.py
│   ├── parse_academic.py
│   ├── anonymize.py
│   ├── build_dataset.py
│   ├── seed.py                ← set_all_seeds() para reproducibilidad
│   ├── finetune_mlx.sh
│   ├── merge_mlx.sh
│   ├── export_gguf.sh
│   └── create_ollama_model.sh ← uso local sin Docker
├── eval/
│   ├── perplexity.py
│   ├── style_metrics.py
│   ├── train_classifier.py
│   ├── generate_blind_pairs.py
│   ├── results/               ← gitignored; JSON con resultados por timestamp
│   └── cache/                 ← gitignored; textos generados cacheados
├── backend/
│   ├── main.py                ← FastAPI con rate-limit, sanitización, security headers
│   ├── config.py              ← todas las env vars con defaults
│   ├── ollama_client.py       ← helpers reutilizables en tests
│   ├── requirements-backend.txt ← solo lo que va al contenedor
│   └── static/                ← index.html, app.js, styles.css
├── tests/
│   ├── fixtures/              ← samples de WhatsApp/Gmail para tests
│   ├── test_parse_whatsapp.py
│   ├── test_parse_gmail.py
│   ├── test_anonymize.py
│   ├── test_build_dataset.py
│   └── test_backend.py
├── models/                    ← gemma3-4b-4bit/ (gitignored)
├── memoria-lora/              ← adaptador LoRA (gitignored)
├── memoria-merged/            ← modelo mergeado (gitignored)
└── memoria-q4.gguf            ← GGUF para Ollama (gitignored)
```

---

## Cómo correr el notebook — paso a paso

### Prerequisitos

```bash
# 1. Crear y activar entorno virtual
python3 -m venv venv && source venv/bin/activate

# 2. Instalar dependencias (Mac)
pip install mlx-lm>=0.22.0 transformers>=4.50.0 peft>=0.14.0 trl>=0.12.0 \
    datasets pdfplumber python-docx spacy nltk scikit-learn \
    fastapi httpx uvicorn slowapi accelerate html2text scipy
python -m spacy download es_core_news_sm

# 3. Login HuggingFace (Gemma es gated)
huggingface-cli login

# 4. Poner los datos en sus carpetas:
#    data/raw/whatsapp/*.txt   ← exports de WhatsApp (menú chat → Exportar → Sin multimedia)
#    data/raw/gmail/*.mbox     ← de myaccount.google.com → Descargar datos → Gmail
#    data/raw/academic/*.pdf   ← papers y DOCX propios
```

### Orden de ejecución del notebook

| # | Celda | Qué hace | Tiempo aprox. |
|---|-------|----------|---------------|
| 1 | `install` | Instala dependencias | 2–5 min |
| 2 | `imports` | Imports + chequeo MPS | < 1 min |
| 3 | `config` | Define `USE_MLX`, `MODEL_ID`, rutas | < 1 min |
| 4 | `parser-whatsapp` | Parsea .txt de WhatsApp → `processed/casual.jsonl` | 1–5 min |
| 5 | `parser-gmail` | Parsea .mbox → `processed/email_prof.jsonl` | 2–10 min |
| 6 | `parser-academic` | Parsea PDFs/DOCX → `processed/academic.jsonl` | 2–5 min |
| 7 | `build-dataset` | Formatea + split 80/10/10 → `dataset/train.jsonl` etc. + manifest | < 1 min |
| 8 | `mlx-convert` | Cuantiza Gemma 3 4B a 4-bit (**una sola vez**, ~15 min, ~2.8 GB) | ~15 min |
| 9 | `finetuning-train` | Entrena LoRA con MLX-LM | ~25–40 min (M5) |
| 10 | `inferencia` | Prueba los tres registros | < 1 min |
| 11 | `merge-lora` | Mergea adaptador → `memoria-merged/` | ~5 min |
| 12 | `eval-perplexity` | Perplexidad base vs. fine-tuneado + bootstrap CI | ~5 min |
| 13 | `eval-style` | Métricas de estilo + bootstrap CI + Mann-Whitney U | < 1 min |
| 14 | `eval-classifier` | Clasificador BETO de autoría + IC Wilson | ~10 min |
| 15 | `eval-blind-test` | Genera pares ciegos (sin labels) para juez humano | ~5 min |
| 16 | `eval-summary` | Guarda `eval/results/<timestamp>.json` con todos los resultados | < 1 min |

**Importante:** las celdas `eval-style`, `eval-classifier` y `eval-blind-test` definen `generate_fn_mlx` en el namespace de Jupyter al ejecutar `eval-style`. Ejecutar esa celda antes que las siguientes.

### Flag de control

En la celda `config`:
```python
USE_MLX = True   # True → MLX-LM (Mac, recomendado)
                 # False → PyTorch MPS (experimental, al límite en 16 GB)
```

### Exportar a Ollama (después del training)

```bash
# Opción A — desde el notebook: correr la celda de exportación GGUF
# Opción B — desde terminal:
bash scripts/merge_mlx.sh          # fuse → memoria-merged/ en bf16
bash scripts/export_gguf.sh        # convierte a GGUF Q4_K_M y registra en Ollama
```

### Levantar la app demo

```bash
# Uso local (sin Docker):
bash scripts/create_ollama_model.sh   # crea el modelo en Ollama con la ruta local
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
# Abrir http://127.0.0.1:8000

# Con Docker Compose (Ollama + backend, healthcheck incluye verificar que 'memoria' está cargado):
docker compose up
```

### Tests

```bash
pytest tests/
```

---

## Decisiones de diseño importantes

### Por qué MLX-LM y no bitsandbytes

`bitsandbytes` requiere CUDA — no corre en Mac. MLX-LM usa la memoria unificada de Apple Silicon directamente, sin copias CPU↔GPU. Con Gemma 3 4B (5.1B params), 4-bit cuantización baja los pesos de ~10.2 GB a ~2.8 GB, lo que permite training holgado en 16 GB.

### Por qué los prompts de train se muestrean de catálogos independientes

Versiones anteriores usaban las primeras 4–10 palabras del texto target como "topic" del prompt. Eso introduce leakage: el modelo ve el inicio del target y puede aprender a continuar ese texto en vez de aprender estilo general. La versión actual muestrea el prompt completo del catálogo `data/prompts/{register}.txt` para cada ejemplo de train. Los catálogos contienen prompts de dominios ajenos al TP (historia, biología, filosofía, economía, etc.) para que E3/E4 midan estilo autorial real y no memorización del contenido del TP.

### Por qué se usa `tokenizer.apply_chat_template` en vez de string hardcodeado

Gemma tiene un chat template específico con tokens especiales de turno. Hardcodearlo puede diferir byte a byte del template oficial (especialmente entre Gemma 3 y Gemma 4). Usar `apply_chat_template` garantiza que train e inferencia usen el mismo template.

### Por qué los prompts de E3/E4 son independientes

Si el clasificador/test ciego usa las primeras N palabras del texto real como seed del generado, lo que mide es si el modelo continúa bien ese comienzo — no si aprendió el estilo. Los prompts en `data/prompts/*.txt` son independientes de los textos del test set.

### Split 80/10/10 y separación estricta del test set

El val set se usa solo para early stopping. El test set se toca únicamente en E1–E4. Antes del split se aplica deduplicación SHA1 (normalización lower+strip+collapse whitespace). El split es estratificado por registro con `sklearn.model_selection.train_test_split(..., stratify=...)`.

### Manifest del dataset

`build_dataset.py` escribe `data/dataset/manifest.json` con: seed, versiones de sklearn/transformers, SHA256 de cada split, counts por registro, SHA1 del catálogo de prompts y parámetros de filtrado. Permite reproducir exactamente el mismo dataset.

### Modelfile: Docker vs. local

- **Docker:** `Modelfile` usa `FROM /models/memoria.gguf` (path absoluto dentro del contenedor; `docker-compose.yml` monta el GGUF en `/models/`).
- **Local:** `scripts/create_ollama_model.sh` genera un `Modelfile.tmp` con la ruta local del GGUF y llama a `ollama create`. No requiere mantener dos Modelfiles en el repo.

### Backend: seguridad y resiliencia

- `_sanitize_prompt()` rechaza tokens especiales de Gemma (`<start_of_turn>`, `<end_of_turn>`, `<bos>`, `<eos>`, `<|...|>`) y literales de registro (`[CASUAL]`, `[EMAIL-PROF]`, `[ACADÉMICO]`) → HTTP 400.
- `slowapi` limita `/generate` a 10 req/min y `/health` a 30 req/min por IP (configurable con `RATE_LIMIT_GENERATE` y `RATE_LIMIT_HEALTH`).
- `asyncio.Semaphore(MAX_CONCURRENT_STREAMS)` limita streams simultáneos (default 3).
- `_SecurityHeadersMiddleware` agrega `X-Content-Type-Options`, `Referrer-Policy` y CSP en todas las respuestas.
- Stack traces solo en logs del servidor; el cliente recibe `{"error": "internal"}` o `{"error": "upstream_timeout"}`.

---

## Bugs corregidos respecto al código original

1. **Perplexidad sesgada:** `labels=input_ids` sin enmascarar padding → loss promediaba sobre tokens de padding. Fix: `CrossEntropyLoss(reduction="none")` acumulando NLL/tokens across batches.
2. **Chat template hardcodeado:** f-string con `<start_of_turn>` → potencial mismatch con Gemma 4. Fix: `tokenizer.apply_chat_template`.
3. **`padding_side` en inferencia:** quedaba en `"right"` durante training. Fix: `tokenizer.padding_side = "left"` antes de `generate()`.
4. **Argentinismos con substring:** `if arg in text_lower` matcheaba `"re"` en `"presente"`. Fix: `re.compile(r'\barg\b')`.
5. **Filtro `real_casual`:** filtraba textos que contenían la *palabra* "casual". Fix: filtrar sobre dict por registro.
6. **Backend streaming sin try/except:** crash silencioso con líneas vacías de Ollama. Fix: `try/except json.JSONDecodeError`.
7. **`[DONE]` solo rompía el for interno:** el while externo seguía leyendo. Fix: flag `done` que rompe ambos loops.
8. **Contador de tokens en app.js:** contaba chunks SSE en vez de tokens reales. Fix: usar `eval_count` del evento final de Ollama.
9. **Dockerfile sin requirements.txt:** instalaba deps hardcodeadas. Fix: `requirements-backend.txt` separado (sin torch/mlx).
10. **Race condition en compose:** `sleep 8 && ollama create` fallaba en hosts lentos. Fix: loop `until curl` + healthcheck que verifica que `memoria` está en `/api/tags`.
11. **Modelfile con path relativo:** `FROM ./memoria-q4.gguf` no funciona dentro del contenedor. Fix: `FROM /models/memoria.gguf`.
12. **WhatsApp BOM en iOS:** `encoding="utf-8"` fallaba con BOM. Fix: `encoding="utf-8-sig"`.
13. **Parser Gmail con substring match:** `sender_email in raw_from` capturaba `otro_nico@` si el sender era `nico@`. Fix: `email.utils.parseaddr` con comparación exacta.
14. **spaCy fallo silencioso en anonymize:** `return None` sin warning. Fix: `logger.warning` + flag `strict`.
15. **Leakage en prompts de training:** `extract_topic` usaba primeras 10 palabras del target. Fix: prompts muestreados de catálogos independientes.

---

## Variables de entorno

Copiar `.env.example` a `.env` y completar:

```
AUTHOR_NAME=Nico                                    # nombre exacto como aparece en WhatsApp
AUTHOR_EMAIL=nico.karagozian@gmail.com              # para filtrar emails enviados
MODEL_ID=google/gemma-3-4b-it
MLX_MODEL_PATH=./models/gemma3-4b-4bit
ADAPTER_PATH=./memoria-lora
OLLAMA_URL=http://localhost:11434/api/generate      # incluir el path /api/generate
OLLAMA_MODEL=memoria

# Opcionales (tienen defaults en backend/config.py)
OLLAMA_TEMPERATURE=0.8
OLLAMA_TOP_P=0.9
OLLAMA_TIMEOUT=120
MAX_CONCURRENT_STREAMS=3
RATE_LIMIT_GENERATE=10/minute
RATE_LIMIT_HEALTH=30/minute
```

---

## Criterios de éxito de la evaluación

| Evaluación | Métrica | Umbral mínimo | Umbral ideal |
|------------|---------|---------------|--------------|
| E1 Perplexidad | Mejora relativa vs. modelo base | ≥ 20% | ≥ 35% |
| E2 Estilo | Diferencia promedio en métricas | < 20% | < 10% |
| E3 Clasificador BETO | Accuracy del clasificador | < 75% | < 60% |
| E4 Test ciego humano | % jueces que identifican el real | < 65% | < 55% |

Todas las evaluaciones reportan intervalos de confianza 95%: bootstrap para E1/E2, Wilson para E3.

---

## Si algo falla

**Fine-tuning no converge:**
- Verificar que hay ≥ 400 ejemplos por registro en el train set
- Bajar learning rate a `1e-4`
- Aumentar `--iters` a 1500

**OOM durante training (MLX):**
- Cerrar browsers, Slack y apps pesadas antes de arrancar
- Verificar que `--batch-size 1` y `--grad-checkpoint` están activos
- Monitorear en Activity Monitor → columna "Memory"

**Ollama no reconoce el modelo:**
- Verificar que `ollama list` muestra `memoria`
- Si no: `bash scripts/create_ollama_model.sh` (requiere `memoria-q4.gguf` en el directorio raíz)

**Backend no conecta:**
- `curl http://127.0.0.1:8000/health` → debe devolver `{"status":"ok","models":[...]}`
- Si devuelve `"degraded"`: el modelo no está cargado en Ollama → re-ejecutar `create_ollama_model.sh`
- Si Ollama no está corriendo: `ollama serve` en otra terminal

**Celdas de eval fallan con `generate_fn_mlx not defined`:**
- Ejecutar primero la celda `eval-style` — define `generate_fn_mlx` en el namespace compartido de Jupyter
