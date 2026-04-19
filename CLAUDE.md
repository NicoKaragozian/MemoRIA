# MemoRIA — Contexto para Claude

**Proyecto:** Fine-tuning de Gemma 4 E2B sobre textos personales de Nico para aprender su estilo de escritura en tres registros: casual (WhatsApp), profesional (email) y académico.

**Equipo:** Nico Karagozian, Clara Kearney, Valen Pivotto — UdeSA MIA NLP 2026  
**Profesor:** Luciano Del Corro (pidió especialmente rigor en evaluación)

---

## Modelo

- **HuggingFace:** `google/gemma-4-E2B-it`
- **Ollama:** `gemma4:e2b`
- **Tamaño:** 2.3B params efectivos / 5.1B totales con embeddings
- **Context:** 128K tokens
- En 16 GB unified RAM requiere cuantización 4-bit antes de entrenar. Sin 4-bit → OOM.
- El modelo fine-tuneado **no publicar** — puede haber señales de PII aunque el pipeline de anonimización haya corrido.

---

## Estructura del proyecto

```
TP/
├── MemoRIA.ipynb          ← notebook principal, ejecutar en orden
├── MEMORIA_PLAN.md        ← plan completo con decisiones de diseño
├── data/
│   ├── raw/               ← gitignored; acá van los datos originales
│   │   ├── whatsapp/      ← .txt de exports de WhatsApp
│   │   ├── gmail/         ← .mbox de Google Takeout
│   │   └── academic/      ← PDFs y DOCX propios
│   ├── processed/         ← generado por los parsers (gitignored)
│   ├── dataset/           ← train/val/valid/test.jsonl (gitignored)
│   └── prompts/           ← prompts independientes para evaluación E3/E4
├── scripts/               ← módulos importados por el notebook
├── eval/                  ← módulos de evaluación
├── models/                ← gemma4-e2b-4bit/ después de mlx_lm.convert (gitignored)
├── backend/               ← FastAPI + frontend web
│   └── static/            ← index.html, app.js, styles.css
├── memoria-lora/          ← adaptador LoRA post-entrenamiento (gitignored)
├── memoria-merged/        ← modelo mergeado post-fuse (gitignored)
└── memoria-q4.gguf        ← GGUF para Ollama (gitignored)
```

---

## Cómo correr el notebook — paso a paso

### Prerequisitos

```bash
# 1. Crear y activar entorno virtual
python -m venv venv && source venv/bin/activate

# 2. Instalar dependencias (Mac)
pip install mlx-lm>=0.22.0 transformers>=4.50.0 peft>=0.14.0 trl>=0.12.0 \
    datasets pdfplumber python-docx spacy nltk scikit-learn \
    fastapi httpx uvicorn accelerate
python -m spacy download es_core_news_sm

# 3. Poner los datos en sus carpetas:
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
| 7 | `build-dataset` | Formatea + split 80/10/10 → `dataset/train.jsonl` etc. | < 1 min |
| 8 | `mlx-convert` | Cuantiza Gemma 4 E2B a 4-bit (**una sola vez**, ~15 min, ~2.8 GB en disco) | ~15 min |
| 9 | `finetuning-train` | Entrena LoRA con MLX-LM | ~25–40 min (M5) |
| 10 | `inferencia` | Prueba los tres registros | < 1 min |
| 11 | `merge-lora` | Mergea adaptador → `memoria-merged/` | ~5 min |
| 12 | `eval-perplexity` | Mide perplexidad base vs. fine-tuneado sobre **test set** | ~5 min |
| 13 | `eval-style` | Métricas de estilo léxico-sintácticas | < 1 min |
| 14 | `eval-classifier` | Entrena clasificador BETO de autoría | ~10 min |
| 15 | `eval-blind-test` | Genera pares para test ciego humano | ~5 min |

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
# Con Ollama ya corriendo y el modelo registrado:
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
# Abrir http://127.0.0.1:8000

# O con Docker Compose (levanta Ollama + backend juntos):
docker compose up
```

---

## Decisiones de diseño importantes

### Por qué MLX-LM y no bitsandbytes

`bitsandbytes` requiere CUDA — no corre en Mac. MLX-LM usa la memoria unificada de Apple Silicon directamente, sin copias CPU↔GPU. Con Gemma 4 E2B (5.1B params), 4-bit cuantización baja los pesos de ~10.2 GB a ~2.8 GB, lo que permite training holgado en 16 GB.

### Por qué los prompts de train se muestrean de catálogos independientes

Versiones anteriores usaban las primeras 4 palabras del texto target como "topic" del prompt. Eso sigue introduciendo leakage (el modelo ve el inicio del target y puede aprender a continuar en vez de aprender estilo), y tampoco garantiza diversidad de instrucciones. La versión actual muestrea el prompt completo del catálogo `data/prompts/{register}.txt` para cada ejemplo de train. Los catálogos contienen prompts de dominios ajenos al TP (historia, biología, economía, etc.) para que E3/E4 midan estilo autorial real y no memorización del contenido del TP.

### Por qué se usa `tokenizer.apply_chat_template` en vez de string hardcodeado

Gemma tiene un chat template específico con tokens especiales de turno. Hardcodearlo puede diferir byte a byte del template oficial (especialmente entre Gemma 3 y Gemma 4). Usar `apply_chat_template` garantiza que train e inferencia usen el mismo template.

### Por qué los prompts de E3/E4 son independientes

Si el clasificador/test ciego usa las primeras N palabras del texto real como seed del generado, lo que mide es si el modelo continúa bien ese comienzo — no si aprendió el estilo. Los prompts en `data/prompts/*.txt` son independientes de los textos del test set.

### Split 80/10/10 y separación estricta del test set

El val set se usa solo para early stopping durante training. El test set se toca únicamente para las evaluaciones finales (E1–E4). Esto evita que el modelo vea señales del test set durante el proceso de entrenamiento/selección de hiperparámetros.

---

## Bugs corregidos respecto al código original

1. **Perplexidad:** el código original pasaba `labels=input_ids` sin enmascarar padding → la loss promediaba sobre tokens de padding. Fix: `labels[attention_mask == 0] = -100`.
2. **Chat template:** se construía con un f-string hardcodeado → potencial mismatch con el template oficial de Gemma 4. Fix: `tokenizer.apply_chat_template`.
3. **`padding_side` en inferencia:** quedaba en `"right"` durante training. Fix: `tokenizer.padding_side = "left"` antes de `generate()`.
4. **Argentinismos:** `if arg in text_lower` matcheaba substrings (`"re"` en `"presente"`). Fix: `re.compile(r'\barg\b')`.
5. **Filtro `real_casual`:** filtraba textos que contenían la *palabra* "casual", no los del *registro* casual. Fix: filtrar sobre el dict estructurado por registro.
6. **Backend streaming:** `json.loads(line)` sin try/except → crash silencioso con líneas vacías de Ollama. Fix: envuelto en try/except.

---

## Variables de entorno

Copiar `.env.example` a `.env` y completar:

```
AUTHOR_NAME=Nico                          # nombre exacto como aparece en WhatsApp
AUTHOR_EMAIL=nico.karagozian@gmail.com    # para filtrar emails enviados
MODEL_ID=google/gemma-4-E2B-it
MLX_MODEL_PATH=./models/gemma4-e2b-4bit
ADAPTER_PATH=./memoria-lora
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=memoria
```

---

## Criterios de éxito de la evaluación

| Evaluación | Métrica | Umbral mínimo | Umbral ideal |
|------------|---------|---------------|--------------|
| E1 Perplexidad | Mejora relativa vs. modelo base | ≥ 20% | ≥ 35% |
| E2 Estilo | Diferencia promedio en métricas | < 20% | < 10% |
| E3 Clasificador BETO | Accuracy del clasificador | < 75% | < 60% |
| E4 Test ciego humano | % jueces que identifican el real | < 65% | < 55% |

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
- Si no: `ollama create memoria -f Modelfile` (requiere que `memoria-q4.gguf` exista)

**Backend no conecta:**
- `curl http://127.0.0.1:8000/health` → debe devolver `{"status":"ok","models":[...]}`
- Si Ollama no está corriendo: `ollama serve` en otra terminal
