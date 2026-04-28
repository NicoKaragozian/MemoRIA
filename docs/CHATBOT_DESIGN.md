# Diseño — Chatbot conversacional (rediseño de scope)

**Branch:** `feature/chatbot-conversacional`
**Estado:** En desarrollo — primer commit con decisiones de diseño

---

## Cambio de scope

### Objetivo original
Generar fragmentos sueltos en el estilo del usuario dado un prompt genérico, distinguiendo tres registros (casual / email profesional / académico).

### Objetivo nuevo
Asistente que, dada una **conversación** (chat 1:1 o grupal) y un **mensaje recibido**, genera la **respuesta que el usuario hubiera escrito** en ese contexto, manteniendo el estilo propio de esa conversación específica.

> **Nota sobre nomenclatura:** "el usuario" se refiere a la persona que entrena el modelo sobre sus propios datos. Su nombre exacto se configura en `.env` con `AUTHOR_NAME`. En los ejemplos de este documento usamos `[Usuario]` como placeholder.

### Por qué el cambio
- El uso real apuntado es asistir a la escritura de respuestas a mensajes que llegan, no producir texto desconectado.
- El modelo previo entrenado con prompts genéricos no responde a preguntas — fue diseñado para generar fragmentos en estilo, lo que en producción resulta poco útil para el usuario final.
- Aprender estilo a partir de pares (contexto, respuesta) condiciona mejor al modelo y mejora la utilidad práctica.

---

## Decisiones de diseño

### Datos
- **WhatsApp** en esta iteración. Email profesional y académico se incorporan en iteraciones siguientes (ver "Ideas para más adelante").
- **Chats 1:1 y grupales** ambos incluidos. En grupos, el modelo aprende del contexto multi-autor.
- **Anonimización de PII** se aplica al **contenido** de los mensajes (teléfonos, emails, URLs, DNIs, CBUs, nombres propios mencionados dentro del texto). Los **nombres de los autores se preservan sin anonimizar**, porque son la señal que le permite al modelo distinguir interlocutores y modular el estilo. Como el modelo no se publica (uso personal), la exposición de nombres de contactos en los pesos del modelo es aceptable.

### Segmentación de conversaciones
- **Turno** = uno o más mensajes consecutivos del mismo autor sin interrupción de otro autor en el medio.
- **Conversación** = secuencia de mensajes donde el gap entre **mensajes consecutivos** es **< 6 horas**. Cuando dos mensajes consecutivos tienen un gap mayor, se cierra la conversación y arranca una nueva.
- **Mensajes de sistema** (multimedia omitido, audio omitido, "Messages and calls are end-to-end encrypted", etc.) se filtran completamente — no aparecen como contexto ni como target, y no rompen turnos.

### Pares de entrenamiento
Cada turno del usuario que **no sea el primero** de una conversación genera un par:

| Campo | Valor |
|-------|-------|
| `chat_name` | nombre del chat (1:1) o del grupo |
| `is_group` | `true` / `false` |
| `participants` | lista de autores del chat (excluyendo al usuario) |
| `context` | últimos **20 mensajes** anteriores en la conversación (cualquier autor, incluyendo al usuario), cada uno con su autor |
| `target` | turno del usuario (mensajes consecutivos unidos) |

### Filtros
- Descartar pares cuyo `target` tenga **< 30 caracteres** (descarta "Dale", "Sí", "Ok" — no aportan señal de estilo). Esos mensajes igual cuentan como contexto válido para pares siguientes.
- Descartar el primer turno de cada conversación (no tiene mensaje recibido).

### Estilo por interlocutor
El modelo recibe el `chat_name` en el input (ej. el nombre del contacto o el nombre del grupo). Esto le permite modular el estilo según con quién/dónde está hablando.

### Formato del input al modelo (chat template)

```
[Chat: NombreGrupo (con Persona1 y Persona2)]

Persona1: hoy comí lo más rico
Persona2: pasa la receta!!
Persona1: te paso por insta
[Usuario]: yo tmb quiero
Persona1: les paso a las dos
Persona2: gracias amor

[Tu próximo mensaje:]
```

`[Usuario]` se sustituye en runtime por el `AUTHOR_NAME` configurado en `.env`. El target a generar es el siguiente turno completo del usuario.

---

## Pipeline de entrenamiento

El modelo se entrena en dos etapas conceptuales: una etapa supervisada inicial sobre los chats del usuario (SFT), y una etapa posterior de aprendizaje por preferencias sobre el feedback que el usuario va generando al usar la app (DPO).

### Vista general

```
                    ┌─────────────────────────────────────────────┐
                    │  Etapa 0 — Preparación de datos             │
data/raw/*.txt ────►│  parse_whatsapp.py + build_dataset.py       │
                    │  Output: data/dataset/{train,val,test}.jsonl│
                    └────────────────────┬────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │  Etapa 1 — SFT con LoRA sobre Gemma 3 4B    │
                    │  finetune_mlx.sh                             │
                    │  Output: memoria-lora/                       │
                    │  Métricas: train/val loss, perplexity vs    │
                    │  base, smoke test, style metrics            │
                    └────────────────────┬────────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │  Etapa 2 — Deploy + recolección             │
                    │  merge_mlx.sh + export_gguf.sh + UI         │
                    │  Output: data/feedback/feedback.jsonl        │
                    │  Métricas: tasa de elección por seed,       │
                    │  diversidad inter-opciones                   │
                    └────────────────────┬────────────────────────┘
                                         ▼ (cuando hay ~50-100 elecciones)
                    ┌─────────────────────────────────────────────┐
                    │  Etapa 3 — DPO sobre preferencias           │
                    │  Output: memoria-dpo-lora/                   │
                    │  Métricas: A/B vs SFT en eval set fijo,     │
                    │  reward margin, KL vs SFT base               │
                    └────────────────────┬────────────────────────┘
                                         ▼
                                Loop iterativo
                       (deploy → feedback → DPO → eval)
```

### Etapa 0 — Preparación de datos

**Status:** implementado y probado.

```bash
# AUTHOR_NAME debe estar en .env (cómo aparece el usuario en sus chats)
export AUTHOR_NAME="Tu Nombre"
export HF_TOKEN=$(cat ~/.huggingface/token)

python -m scripts.build_dataset
```

**Outputs:**

- `data/processed/whatsapp_pairs.jsonl` — pares (contexto, target) por archivo de chat
- `data/dataset/train.jsonl` — 80% de los pares con chat template de Gemma aplicado
- `data/dataset/val.jsonl` y `valid.jsonl` — 10% para early stopping
- `data/dataset/test.jsonl` — 10% reservado para evaluación (no se toca durante training)
- `data/dataset/manifest.json` — seed, hashes, counts por chat, versiones de sklearn/transformers

**Métricas que aplican en esta etapa:**

| Métrica | Cómo se mide | Cuándo |
|---------|--------------|--------|
| Cantidad de pares por chat | `manifest.json:train_by_chat` | Cada vez que se regenera el dataset |
| Distribución de tokens por ejemplo | tokenize cada ejemplo y graficar | Sanity check antes de entrenar |
| Pares descartados por > MAX_TOKEN_LEN | log de `build_dataset.py` | Durante regeneración |
| % de turnos del usuario filtrados por `min_target_chars` | comparar `parse_whatsapp` raw vs filtrado | Sanity check |

### Etapa 1 — Supervised Fine-Tuning (SFT)

**Status:** implementado. Resultado actual con calidad baja, listo para iterar.

**Qué es:** entrenamiento supervisado clásico. El modelo aprende a maximizar la probabilidad de la respuesta real del usuario dado el contexto. El "ground truth" es lo que el usuario efectivamente escribió en su chat.

**Cómo:**

1. El parser construye pares `(contexto, respuesta_real_del_usuario)` desde los chats de WhatsApp.
2. `build_dataset.py` los formatea con el chat template de Gemma 3:
   - `user`: header del chat + contexto de los últimos N mensajes
   - `assistant`: el turno real del usuario (target)
3. Se entrena con **MLX-LM LoRA** sobre Gemma 3 4B cuantizado a 4-bit (Apple Silicon).

```bash
# Cuantizar el modelo base (una sola vez, ~3-15 min)
# Esto lo hace finetune_mlx.sh la primera vez.

# Entrenar
bash scripts/finetune_mlx.sh 2000   # 2000 iters
```

**Hiperparámetros usados en esta iteración:**

| Param | Valor | Notas |
|-------|-------|-------|
| Modelo base | `google/gemma-3-4b-it` | Versión instruct |
| Cuantización | 4-bit (Q4) | MLX, group size 64 |
| Método | LoRA | Solo el adapter se entrena |
| LoRA layers | 16 | Default de `mlx_lm.lora` |
| Iters | 2000 | Subió de 1000 → 2000 entre iteraciones |
| Learning rate | `2e-4` | Default |
| Batch size | 1 | Limitado por 16 GB unified RAM |
| Save every | 200 | Snapshots intermedios para A/B contra checkpoints |
| `MAX_TOKEN_LEN` | 2048 | Filtro al armar dataset |

**Métricas que aplican en esta etapa (en orden de costo):**

| # | Métrica | Qué mide | Cuándo correr | Status |
|---|---------|----------|---------------|--------|
| 1 | **Train loss** | Convergencia durante training | Continuo | ✅ activo (MLX-LM lo loggea) |
| 2 | **Val loss** | Generalización; señal de overfitting | Cada 200 iters | ✅ activo (default `--save-every`) |
| 3 | **Perplexity comparativa** | Modelo SFT vs modelo base sobre `test.jsonl` | Después de cada training | ⚠️ heredado de iteración 0 (`eval/perplexity.py`); funciona pero hay que adaptarlo al nuevo formato |
| 4 | **Style metrics** | Largo, vocabulario, emojis, signos: respuesta generada vs target real | Después de cada training | ⚠️ heredado (`eval/style_metrics.py`); reusable casi sin cambios |
| 5 | **Smoke test** | Inspección manual de N muestras del test set | Después de cada training | ✅ activo (manual). Documentar en `eval/smoke_<timestamp>.md` cada vez. |
| 6 | **LLM-as-judge** | Claude/Sonnet puntúa "¿esto suena al usuario?" sobre 100 ejemplos del test | Por cambio de hiperparámetros | ❌ pendiente (ver `OBSERVABILIDAD_EVALS.md`) |
| 7 | **Test ciego humano** | Panel humano elige cuál es la real entre (real, generada) | Por release | ❌ pendiente |

**Resultados al cierre de iteración 1:**

- Train loss final: 2.4
- Val loss final: 2.9
- Smoke test: respuestas incoherentes con frecuencia, repetición de tokens del anonimizador, ocasionalmente alucina prefijo de autor.

**Diagnóstico:** loss alto + smoke test pobre indica que **el modelo no convergió bien**. Hipótesis y caminos para mejorar listados en "Estado al cierre de iteración 1" (más adelante).

### Etapa 2 — Deploy + recolección de preferencias

**Status:** implementado.

```bash
# Mergear LoRA en el modelo base
bash scripts/merge_mlx.sh

# Exportar a GGUF y registrar en Ollama como "memoria"
bash scripts/export_gguf.sh

# Levantar el backend
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Cómo funciona la recolección:**

- Para cada mensaje que el usuario quiere responder, la UI lanza **3 generaciones en paralelo** con seeds distintos.
- El usuario elige una de las 3.
- El backend persiste la elección en `data/feedback/feedback.jsonl` con: `chat_name`, `is_group`, `participants`, `received_message`, `options[3]`, `chosen_idx`, `seeds`, `timestamp`.

**Métricas que aplican en esta etapa:**

| Métrica | Qué mide | Cuándo |
|---------|----------|--------|
| **Tasa de elección por opción** | ¿La opción 1 siempre gana? Indicaría sesgo del decoder o seed degenerado | Reportable a partir de ~30 elecciones |
| **Diversidad inter-opciones** | Distancia (BLEU/embedding) entre las 3 opciones; baja diversidad = fallo del sampling | Continuo |
| **Tokens / latencia / costo** | Performance operativa | Continuo |
| **Cobertura por chat** | ¿Hay chats sin elecciones? El DPO posterior va a estar sesgado a los chats activos | Reportable cuando se acumule feedback |

### Etapa 3 — Preference fine-tuning con DPO

**Status:** diseño. Ejecutable cuando se acumulen ~50-100 elecciones.

**Por qué hace falta:** SFT enseña al modelo a imitar respuestas pasadas, pero no a diferenciar **cuál de varias respuestas plausibles es mejor** en un mismo contexto. Cuando el usuario elige 1 de las 3 opciones, le da al modelo una señal nueva: "en este contexto, prefiero esto sobre estas otras dos". Esa señal es lo que se usa para una segunda etapa de entrenamiento llamada **preference fine-tuning**.

**Cómo se va a hacer:**

1. **Construir el dataset de preferencias** (`scripts/build_dpo_dataset.py`, pendiente):
   Cada elección produce 2 pares de preferencia (la elegida vs cada una de las 2 descartadas):
   ```json
   {"prompt": "<chat template formateado>", "chosen": "<respuesta elegida>", "rejected": "<respuesta descartada>"}
   ```
2. **Aplicar DPO sobre el modelo SFT actual.** MLX-LM no tiene DPO nativo al momento de este doc; se evalúan dos caminos:
   - **`trl` (HuggingFace) con `DPOTrainer`**: requiere bajar al modelo no cuantizado en CPU/MPS; más lento en M1 pero la implementación es estándar.
   - **Implementación manual con `mlx`**: la fórmula de DPO es directa; ~100 líneas de código si se sigue el paper. Más trabajo, pero corre en MLX.
3. **Output:** `memoria-dpo-lora/` (un nuevo adaptador LoRA encima del SFT, no reemplaza).

**Hiperparámetros típicos para DPO:**

| Param | Valor sugerido | Notas |
|-------|----------------|-------|
| `beta` | 0.1 | Fuerza con la que el modelo se aleja del SFT base. Más alto = menos drift, más conservador. |
| Learning rate | `5e-7` a `1e-6` | Mucho más bajo que SFT |
| Iters | depende del tamaño | Regla práctica: ~3 epochs sobre los pares de preferencia |
| Reference model | el SFT actual | DPO compara el modelo entrenado vs el de referencia |

**Métricas que aplican en esta etapa:**

| # | Métrica | Qué mide | Cómo se calcula |
|---|---------|----------|-----------------|
| 1 | **DPO loss** | Convergencia | log durante training |
| 2 | **Reward margin** (chosen − rejected) | Cuán "lejos" pone el modelo a la elegida vs la rechazada | Promedio sobre val set de preferencias |
| 3 | **KL divergence vs SFT base** | Cuánto se alejó del modelo de referencia | Métrica intrínseca de DPO |
| 4 | **Win rate A/B** | En el eval set fijo, ¿qué porcentaje de respuestas del DPO prefieren los evaluadores sobre las del SFT? | LLM-as-judge o humano sobre N pares (DPO_response, SFT_response) ciegos |
| 5 | **Drop de calidad de estilo** | ¿El DPO arruinó el estilo aprendido por SFT? | Style metrics post-DPO comparado con post-SFT |

**Por qué DPO y no RLHF:** RLHF entrena primero un reward model sobre las preferencias y después aplica PPO sobre el LM con ese reward. Es más complejo, más caro, y más inestable. DPO logra resultados equivalentes con un único entrenamiento estilo SFT, sin reward model intermedio. Para un proyecto personal con preferencias acumuladas en el tiempo, DPO es la opción correcta.

**Alternativas que se podrían explorar en iteraciones futuras:**

- **KTO (Kahneman-Tversky Optimization):** no requiere pares; alcanza con etiquetas binarias "buena/mala". Útil si se simplifica la UI a 1 opción + thumbs up/down.
- **IPO (Identity Preference Optimization):** variante de DPO más estable cuando hay ruido en las preferencias.
- **Online DPO:** reentrenamiento incremental cada N elecciones nuevas, sin re-correr todo el dataset desde cero.

### Loop de mejora continua

Una vez que existen las dos etapas, el ciclo de mejora se vuelve:

```
1. Versión actual del modelo está deployada (ej. memoria-v3)
2. Usuario interactúa con la UI → genera más feedback
3. Cada N elecciones nuevas, reentreno DPO sobre el SFT base con TODO el feedback acumulado
4. Antes de promover el modelo nuevo a "memoria-v(N+1)":
   - Métricas determinísticas no deberían empeorar (tokens, latencia)
   - Style metrics no deberían divergir del usuario real
   - Win rate vs versión anterior > 50% en LLM-as-judge sobre eval set fijo
   - (eventual) win rate humano > 50%
5. Si pasa, promover. Si no, descartar y revisar hyperparams o datos.
```

**Eval set fijo (importante):** un subset de **100-200 mensajes recibidos reales** congelados al inicio del proyecto, que se usa como benchmark estable para comparar versiones. No debe contaminarse con feedback nuevo (es el "test de regresión" del modelo).

### Vista consolidada — Métricas de evaluación por etapa

| Etapa | Activa hoy | Pendiente |
|-------|------------|-----------|
| **0. Datos** | counts del manifest, distribución de tokens | — |
| **1. SFT** | train loss, val loss, smoke test manual | perplexity comparativa, style metrics, LLM-as-judge, test ciego humano |
| **2. Recolección** | logs del backend, feedback.jsonl | tasa de elección, diversidad inter-opciones, cobertura por chat, costos/latencia |
| **3. DPO** | — (todo) | DPO loss, reward margin, KL vs SFT, win rate A/B, drop de style metrics |
| **Loop** | — | promoción condicionada por métricas, eval set fijo congelado |

El diseño completo de la capa de evals (Langfuse, taxonomía, triangulación, etc.) vive en [OBSERVABILIDAD_EVALS.md](OBSERVABILIDAD_EVALS.md).

---

## Parámetros (parametrizables vía `.env`)

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `CONTEXT_MESSAGES` | `20` | Cantidad de mensajes anteriores incluidos como contexto |
| `CONVERSATION_GAP_HOURS` | `6` | Gap de tiempo (en horas) que cierra una conversación |
| `MIN_TARGET_CHARS` | `30` | Largo mínimo del turno del usuario para considerarse target de entrenamiento |

---

## Estimación de volumen

Estimación basada en los chats del usuario que está entrenando en esta iteración. Los nombres de chats se anonimizan en este documento; los volúmenes son los reales. Los volúmenes van a variar para cada usuario según sus propios chats.

| Chat | Tipo | Mensajes del usuario | Pares estimados |
|------|------|----------------------|-----------------|
| Grupo A | Grupo (3 personas) | 2527 | ~1500 |
| Grupo B | Grupo (4 personas) | 2118 | ~1300 |
| Grupo C | Grupo (6 personas) | 387 | ~200 |
| Contacto A | 1:1 | 307 | ~180 |
| **Total** | | **5339** | **~3200** |

---

## Plan de trabajo

- [x] Crear branch + design doc inicial
- [x] Reescribir `scripts/parse_whatsapp.py` para extraer pares conversacionales
- [x] Adaptar `scripts/build_dataset.py` al nuevo formato
- [x] Definir nuevo chat template y tokenización en `build_dataset.py`
- [x] Re-entrenar con MLX-LM (LoRA) sobre el nuevo dataset
- [ ] Adaptar UI (`backend/static/`): textareas para chat name + contexto + mensaje recibido
- [ ] Adaptar backend `/generate` para aceptar el nuevo input
- [ ] Verificar end-to-end con conversación real
- [ ] Iterar sobre la calidad del modelo (ver "Estado al cierre de iteración 1")

---

## Estado al cierre de iteración 1 (2026-04-27)

### Lo que quedó funcionando

Pipeline end-to-end completo desde chats `.txt` hasta modelo entrenado:

1. **Parser conversacional** ([scripts/parse_whatsapp.py](../scripts/parse_whatsapp.py)) — 14 tests pasando, extrae pares (contexto + target) preservando autores y filtrando mensajes de sistema.
2. **Build dataset** ([scripts/build_dataset.py](../scripts/build_dataset.py)) — 11 tests pasando, formatea pares al chat template de Gemma, dedupea y estratifica por chat.
3. **Reentrenamiento** completo: 2000 iters / `lr=2e-4` / LoRA 16 layers.

### Volumen real

| Chat | Pares |
|------|-------|
| Grupo B (4 personas) | 1539 |
| Grupo A (3 personas) | 511 |
| Grupo C (6 personas) | 273 |
| Contacto A (1:1) | 197 |
| **Total parseado** | **2520** |
| Post-dedup | 2509 |
| Descartados por > MAX_TOKEN_LEN | 2 |
| **Train / Val / Test** | **2005 / 251 / 251** |

### Resultado del entrenamiento

| Métrica | Iteración 0 (1000 iters, formato viejo) | Iteración 1 (2000 iters, formato conversacional) |
|---------|-----------------------------------------|--------------------------------------------------|
| Train loss final | 3.2 | **2.4** |
| Val loss final | 3.5 | **2.9** |
| Tiempo de entrenamiento | 7 min | ~71 min |

### Smoke test sobre el test set

El modelo lee el contexto y genera algo en formato chat (no copia el último autor cuando hay sampling con temperatura), pero **la calidad del contenido es muy baja**: respuestas incoherentes, repeticiones de tokens del anonimizador (`<PER>`, `<LOC>`), a veces empieza a generar otro turno con prefijo de autor.

### Causas probables (a investigar en próximas iteraciones)

1. **Loss alto** (val 2.9) — el modelo no convergió bien.
2. **Gemma 3 4B-it tiene un prior muy "instruct"** — el LoRA chico (16 layers, rank por defecto) no tiene fuerza para reorientar el modelo a chat informal de WhatsApp.
3. **2000 ejemplos es poco** para una task generativa abierta donde cada respuesta es única.
4. **Anonimización agresiva** — `<PER>` / `<LOC>` aparecen frecuentemente y el modelo los aprende como noise pattern.

### Caminos para mejorar (cualquiera en branch separada)

| Opción | Esfuerzo | Hipótesis |
|--------|----------|-----------|
| Más iters (3000-5000) y/o más layers (32) | ~3 hs | Mejor convergencia |
| Reentrenar sobre `gemma-3-4b` base (no `-it`) | ~4 hs | Menos prior "instruct" que combatir |
| Sumar email_prof y académico | depende de datos | Más material, mejor estilo en general |
| Filtrar más agresivamente targets con muchos `<PER>`/`<LOC>` | ~1 hs | Reducir noise pattern |
| Bajar `min_target_chars` o cambiar segmentación | ~1 hs | Más datos, ver tradeoff |

### Próximos pasos concretos para esta branch

1. **Adaptar UI** ([backend/static/](../backend/static/)): textareas para chat name + contexto + mensaje recibido + botón "Generar"
2. **Adaptar backend** `/generate`: nueva firma con `chat_name`, `is_group`, `context[]`, construir el prompt con la misma lógica que `_format_user_prompt` de `build_dataset.py`
3. Re-exportar a GGUF y registrar en Ollama como `memoria-v2` (mantener `memoria` viejo para comparar)
4. Probar end-to-end en `127.0.0.1:8000` con conversación real
5. Una vez funcionando: abrir branches separadas para iterar el modelo (ver tabla arriba) y para observabilidad/evals (ver [OBSERVABILIDAD_EVALS.md](OBSERVABILIDAD_EVALS.md))

---

## Ideas para más adelante (post-MVP)

- **Feedback loop de contexto en la UI:** si la respuesta no es buena, botón "más contexto" que regenera con 40 → 60 → 80 mensajes hasta el límite de tokens.
- **Identificación implícita de respuestas a mensajes específicos** (no al último): WhatsApp tiene reply explícito que rara vez está en el export, pero se podría heurizar.
- **Filtrado de mensajes muy cortos del contexto** ("ja", "ok", "✅") si introducen demasiado ruido.
- **Sumar email_prof y académico** para extender el caso de uso a esos registros (planeado para iteraciones siguientes).
- **Selector de modelo en la UI** para comparar `memoria` vs. modelos base.

---

## Bugs detectados durante el pipeline original

Documentados acá para tratarlos en branch separada (`fix/pipeline-bugs`):

1. `scripts/merge_mlx.sh` usaba `--de-quantize` (correcto: `--dequantize` sin guión). **Ya fixeado en esta branch** porque era bloqueante.
2. `scripts/export_gguf.sh` chequea `[ ! -d llama.cpp ]` pero la carpeta vacía hace que se saltee el clone.
3. `scripts/export_gguf.sh` no copia `tokenizer.model` a `memoria-merged/` antes del convert, lo que rompe la conversión a GGUF (Gemma 3 cae en path BPE en vez de SentencePiece).
4. `slowapi` falta en `requirements.txt` principal (solo está en `backend/requirements-backend.txt`).
