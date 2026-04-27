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
El modelo recibe el `chat_name` en el input (ej. `"Mechi Muino"`, `"International girlies"`). Esto le permite modular el estilo según con quién/dónde está hablando.

### Formato del input al modelo (chat template)

```
[Chat: International girlies (con Delfi y Luna)]

Delfi: hoy comí lo más rico
Luna: pasa la receta!!
Delfi: te paso por insta
[Usuario]: yo tmb quiero
Delfi: les paso a las dos
Luna: gracias amor

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

Estimación basada en los chats actuales del usuario que está entrenando ahora (`AUTHOR_NAME=Clara Kearney`). Los volúmenes van a variar para cada usuario según sus propios chats.

| Chat | Tipo | Mensajes del usuario | Pares estimados |
|------|------|----------------------|-----------------|
| International girlies | Grupo (3 personas) | 2527 | ~1500 |
| TFEC BK | Grupo | 2118 | ~1300 |
| Maestria IA mesa chica | Grupo | 387 | ~200 |
| Mechi Muino | 1:1 | 307 | ~180 |
| **Total** | | **5339** | **~3200** |

---

## Plan de trabajo

- [x] Crear branch + design doc inicial
- [ ] Reescribir `scripts/parse_whatsapp.py` para extraer pares conversacionales
- [ ] Adaptar `scripts/build_dataset.py` al nuevo formato
- [ ] Definir nuevo chat template y tokenización en `build_dataset.py`
- [ ] Re-entrenar con MLX-LM (LoRA) sobre el nuevo dataset
- [ ] Adaptar UI (`backend/static/`): textareas para chat name + contexto + mensaje recibido
- [ ] Adaptar backend `/generate` para aceptar el nuevo input
- [ ] Verificar end-to-end con conversación real

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
