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
