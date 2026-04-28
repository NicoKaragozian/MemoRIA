# Diseño — Capa de observabilidad y evals para MemoRIA

**Estado:** Propuesta inicial (no implementada).
**Branch sugerida:** `feature/observability-evals` (independiente de `feature/chatbot-conversacional`).

---

## Motivación

MemoRIA es un chatbot que responde imitando la forma de comunicarse del usuario: tono, vocabulario, criterios, decisiones. La calidad del producto depende casi enteramente de una métrica difícil de medir objetivamente: **¿qué tan parecido al usuario responde?**

Hoy las llamadas al modelo se ejecutan sin un sistema sistemático de logging, trazabilidad ni evaluación de calidad. Esto funciona en etapa de prototipo, pero no es sostenible si se quiere iterar en serio sobre la fidelidad del chatbot a la voz del usuario, sumar contextos nuevos, o eventualmente abrirlo a más usuarios.

Sin observabilidad, cada cambio (al prompt, al modelo, al contexto inyectado, a la lógica de recuperación de memoria) se hace a ciegas. No se sabe si el modelo está respondiendo más como el usuario o menos como el usuario después del cambio.

### Problemática

1. **No hay trazabilidad del razonamiento del chatbot.** Cuando una respuesta no suena al usuario, no hay forma de investigar qué pasó: qué fragmentos del contexto recuperó, qué prompt estaba activo, qué modelo respondió, qué contexto tenía la conversación.
2. **No hay forma de medir fidelidad de voz.** Toda evaluación hoy es subjetiva: se lee una respuesta y se decide si suena o no. Sin métrica, no hay mejora sistemática.
3. **Cada cambio al sistema es a ciegas.** Cambios al prompt, al modelo, a hiperparámetros — no hay forma sistemática de comparar versión nueva vs. anterior.
4. **No hay datos para iterar.** Para que el modelo mejore con el tiempo se necesita un dataset etiquetado de pregunta + respuesta + evaluación de fidelidad. Hoy esos datos no se capturan.
5. **No hay control de costos ni performance.** No se mide tokens, costos ni latencia.

---

## Definición operacional de "sonar a mí"

> **Esta sección es la más importante del documento.** Si las dimensiones de "sonar a mí" están mal definidas, las evaluaciones no van a servir para mejorar — van a ser ruido coordinado.

### Por qué esto va antes que el tooling

"Sonar a mí" no es una dimensión única. Puede ser vocabulario, estructura de oraciones, qué cosas se elige enfatizar, qué cosas se omiten, cuándo se usa humor, cuándo se es directo y cuándo se da vuelta. Antes de implementar evals automáticos hay que definir las dimensiones — sino el rubric sale del aire y los scores no se anclan a nada real.

### Workshop de taxonomía (Fase -1)

Output: `docs/taxonomy_v1.yml` con dimensiones definidas por el usuario en sus propias palabras.

**Procedimiento:**
1. Tomar 30-50 mensajes propios reales de distintos chats (1:1 cercano, grupo, profesional, familia).
2. Anotar cada uno respondiendo: **¿qué hace que este mensaje suene a mí?** Sin grilla preexistente; libre.
3. Destilar las anotaciones en una primera lista de dimensiones.

### Posibles ejes para inspirar (no para imponer)

- **Forma**: largo, fragmentación, signos, emojis, mayúsculas para énfasis.
- **Voz**: cuándo se da vuelta, cuándo se es directo, cuándo se abre emocionalmente.
- **Léxico**: palabras propias (`"divinoo"`, `"joya"`, `"dale re"`) y palabras que **no** se usan.
- **Apertura**: contar más, contar menos, devolver pregunta.
- **Humor**: irónico, autodesburlado, juegos de palabras, ninguno.
- **Modulación por interlocutor**: ¿el "yo con Mechi" es distinto del "yo con mi jefe"?

### Versionado

La taxonomía evoluciona. `taxonomy_v1.yml` → `v2.yml` cuando se refina. Cada eval queda asociado a la versión usada.

### Sub-datasets por dimensión

Una vez definida la taxonomía, se pueden marcar sub-conjuntos del propio test set donde una dimensión específica se manifiesta fuerte (ej. "mensajes donde uso humor irónico") y entrenar/evaluar con foco.

---

## El dataset de evals como autorretrato

Punto sutil pero relevante: el usuario va a estar puntuando cómo se ve a sí mismo respondiendo. Eso introduce un sesgo de auto-percepción y, al mismo tiempo, es interesante de explorar como objeto de estudio.

### Tres mecanismos para mitigar (y aprovechar) el sesgo

**(a) Triangulación de evaluadores.**
El mismo eval set es puntuado por:
- El usuario.
- 1 o 2 personas que conocen bien al usuario (familia, amigos cercanos).
- Un LLM-as-judge (Claude Opus o Sonnet).

Los **deltas entre las tres fuentes son los datos más interesantes**. Si en "humor" el usuario puntúa 4 y una amiga puntúa 2, ahí hay algo que aprender — no necesariamente que el modelo esté mal.

**(b) Journal en paralelo a los scores.**
Junto al rating numérico, un campo libre opcional: "qué noté en esta respuesta". Después se puede minar ese journal para ver patrones — capaz se descubre que se es sistemáticamente más duro en mensajes profesionales que casuales, etc.

**(c) Auto-prediction calibration.**
Antes de evaluar una respuesta, el usuario anota "creo que esto es un X de 5". Después puntúa de verdad. La distancia entre las dos mide qué tan calibrada es la auto-percepción del modelo — pero también, indirectamente, de la propia voz.

---

## Stack recomendado

**Langfuse self-hosted** (open source, Docker Compose):
- SDK Python para wrappear llamadas al modelo.
- UI web para explorar traces.
- Sistema de datasets y evals integrado.
- Versionado de prompts.
- Cálculo de costos.
- **Privacidad**: todo queda local. Único external dep es Postgres (en el mismo compose).

**Alternativa "casera"**: SQLite + Streamlit. Más privacidad absoluta, mucho más trabajo. Recomendación: empezar con Langfuse.

---

## Schema del trace (lo que se captura por respuesta)

```jsonc
{
  "trace_id": "uuid",
  "session_id": "uuid_conversacion",   // mismo para toda una conversación
  "timestamp": "iso8601",
  "user_id": "Clara Kearney",          // del .env (AUTHOR_NAME)

  "input": {
    "user_message": "...",
    "chat_name": "Mechi Muino",
    "is_group": false,
    "conversation_history": [{"author": "...", "text": "..."}]
  },

  "spans": [                            // pasos del pipeline, cada uno trackeado
    {
      "name": "classify_intent",        // si en algún momento hay clasificación
      "input": {...}, "output": {...},
      "model": "...", "tokens_in": N, "tokens_out": N, "latency_ms": N
    },
    {
      "name": "retrieve_memory",        // si se agrega RAG sobre memorias propias
      "query": "...",
      "results": [{"chunk": "...", "score": 0.87}],
      "latency_ms": N
    },
    {
      "name": "generate_response",
      "prompt_version": "v3",
      "model": "memoria-v2",
      "params": {"temperature": 0.8, "top_p": 0.9},
      "tokens_in": N, "tokens_out": N, "latency_ms": N
    }
  ],

  "output": "respuesta final",
  "total_tokens": N,
  "total_latency_ms": N,
  "cost_usd": 0.0,                      // calculado como equivalente cloud aunque corra local

  "evals": {                            // se llenan asincrónicamente
    "voice_fidelity_per_dimension": {
      "tono": 4.2,
      "vocabulario": 3.8,
      "humor": 4.5,
      ...                                // sale de la taxonomía
    },
    "style_metrics": {                   // métricas computacionales (ya existen)
      "avg_msg_len": 23, "emoji_ratio": 0.12, "exclamation_ratio": 0.3
    },
    "human_evaluator_scores": [          // triangulación
      {"evaluator": "self", "scores": {...}, "journal": "..."},
      {"evaluator": "amiga_1", "scores": {...}, "journal": "..."}
    ]
  }
}
```

---

## Pipeline de evals (3 niveles)

| Nivel | Qué mide | Cuándo corre | Costo |
|-------|----------|--------------|-------|
| **1. Determinístico** | Tokens, latencia, costo, patologías (respuesta vacía, prefijo de autor alucinado, loops de tokens) | Cada generación | Gratis |
| **2. LLM-as-judge** | Cada dimensión de la taxonomía puntuada 1-5 por Claude, comparando vs. respuesta real | Sample diario + por cada cambio (modelo / prompt / config) sobre un set fijo de 100 ejemplos | ~$0.5 por corrida (Opus) o ~$0.05 (Sonnet) |
| **3. Test ciego humano + triangulación** | El usuario, 1-2 personas que lo conocen, y el LLM puntúan el mismo set. Comparación de deltas. | Por release importante | Tiempo del usuario y de evaluadores externos |

Las **métricas de estilo computacionales** del proyecto ([eval/style_metrics.py](../eval/style_metrics.py)) se reutilizan dentro del Nivel 1.

---

## Plan por fases

| Fase | Tiempo | Qué entrega |
|------|--------|-------------|
| **-1. Workshop de taxonomía** | 1-2 hs (sesión sola del usuario, con apoyo del agent) | `docs/taxonomy_v1.yml` con dimensiones de "sonar a mí" |
| **0. Setup Langfuse** | 1-2 hs | Docker compose + SDK en `backend/main.py`, primer trace funcionando |
| **1. Schema completo + capture en producción** | 2-3 hs | Cada `/generate` se loggea con todo el contexto del schema |
| **2. Eval Nivel 1 + 2** | 3-4 hs | Script `eval/run_evals.py` que corre evals contra una versión del modelo, usando taxonomía v1 |
| **2.5 Triangulación** | ongoing | Setup mínimo para que 1-2 evaluadores externos puntúen el mismo set |
| **3. Dashboard custom** (opcional) | 2-3 hs | Vistas específicas si Langfuse UI no alcanza |
| **4. Eval Nivel 3** | iterativo | Sistema de panel ciego con journal y auto-prediction calibration |

---

## Decisiones abiertas

1. **Langfuse vs casero.** Recomendación: Langfuse self-hosted.
2. **LLM juez para Nivel 2.** Claude Opus (mejor calidad, ~$0.5/corrida) vs Sonnet (~10× más barato, calidad muy alta igual). Requiere API key de Anthropic.
3. **Persistencia.** Langfuse usa Postgres (queda local en Docker). Alternativa más liviana: SQLite con solución casera.
4. **Eval set.** Sugerencia: 100 ejemplos fijos del test set + 50 muestreados aleatoriamente cada corrida.
5. **Trackeo de costos.** Aunque todo es local, calcular costo equivalente como si fuera cloud (útil para benchmarking y para decisiones de migración futura).

---

## Cómo encaja con otros features

Esta capa **no compite** con `feature/chatbot-conversacional` — son ortogonales. Sugerencia de orden:

1. Cerrar `feature/chatbot-conversacional` (UI + backend adaptados, modelo end-to-end aunque imperfecto).
2. Iterar la calidad del modelo en branches separadas (ver "Caminos para mejorar" en [CHATBOT_DESIGN.md](CHATBOT_DESIGN.md)).
3. Abrir `feature/observability-evals` empezando por la **Fase -1 (taxonomía)** — esa fase no requiere ningún cambio de código y rinde valor inmediato como ejercicio de diseño.
