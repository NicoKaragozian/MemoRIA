import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.config import (
    AUTHOR_NAME, CORS_ORIGINS, MAX_CONCURRENT_STREAMS, MODEL_NAME,
    OLLAMA_TAGS, OLLAMA_TEMPERATURE, OLLAMA_TIMEOUT, OLLAMA_TOP_P, OLLAMA_URL,
    RATE_LIMIT_GENERATE, RATE_LIMIT_HEALTH,
)
from scripts.build_dataset import _format_user_prompt

PAIRS_FILE    = Path("data/processed/whatsapp_pairs.jsonl")
FEEDBACK_DIR  = Path("data/feedback")
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memoria")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="MemoRIA API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'"
        return response


app.add_middleware(_SecurityHeadersMiddleware)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Tokens especiales de Gemma que no deben llegar inyectados por el usuario.
_FORBIDDEN = re.compile(
    r'<start_of_turn>|<end_of_turn>|<bos>|<eos>|<\|[^|]+\|>',
    re.IGNORECASE,
)

# Patrones que el modelo puede generar como output cuando no aprendió a parar
# bien (subentrenamiento). Si aparecen, cortamos la respuesta ahí: el output
# útil es solo lo que vino antes.
_OUTPUT_STOP = re.compile(
    r'\[Tu próximo mensaje:\]'
    r'|\[Chat con\b'
    r'|\[Chat:'
    r'|\[CASUAL\]|\[EMAIL-PROF\]|\[ACADÉMICO\]'
    r'|<start_of_turn>|<end_of_turn>|<bos>|<eos>'
    # Tokens del anonimizador. Después del reentrenamiento sin anonimización
    # de nombres propios, <PER>/<LOC>/<ORG> ya no deberían aparecer en outputs
    # — los sacamos. Los tokens de PII real (números, emails, etc.) siguen
    # presentes en el dataset, así que el modelo todavía puede emitirlos.
    r'|<EMAIL>|<URL>|<PHONE>|<HANDLE>|<COORDS>'
    r'|<DNI>|<ID>|<CBU>|<IBAN>|<NUM>',
    re.IGNORECASE,
)

# Frases de mensajes de sistema de WhatsApp que el modelo aprendió como noise
# y a veces emite. Se eliminan del output.
_OUTPUT_NOISE = re.compile(
    r'(?:audio|image|video|sticker|GIF|document) omitted'
    r'|<Multimedia omitido>|imagen omitida|audio omitido|video omitido'
    r'|sticker omitido|Contact card omitted',
    re.IGNORECASE,
)

# Buffer de hold: nunca enviamos al frontend los últimos N caracteres del
# output, así si están construyendo un patrón stop podemos cortarlos antes
# de que el cliente los vea. N debe ser >= longitud del patrón más largo.
_HOLD_CHARS = 64

def _make_dynamic_stop(chat_name: str, participants: list[str]) -> Optional[re.Pattern]:
    """
    Construye un regex que matchea 'Author:' al inicio de línea para:
    - cada participante del chat,
    - el nombre del chat (en grupos puede aparecer como autor de mensajes
      de sistema mal capturados),
    - el AUTHOR_NAME del usuario (a veces el modelo aluciona su propio
      nombre como prefijo, porque en el dataset las respuestas del usuario
      aparecen identificadas con su nombre en el contexto).
    Si el modelo empieza a generar un turno con prefijo de autor, cortamos
    ahí — significa que se confundió y empezó a alucinar otro mensaje.
    """
    candidates = (participants or []) + [chat_name, AUTHOR_NAME]
    names = [n.strip() for n in candidates if n and n.strip()]
    if not names:
        return None
    escaped = sorted({re.escape(n) for n in names}, key=len, reverse=True)
    # Match al inicio de una línea (^ con MULTILINE) seguido del nombre y ":"
    return re.compile(
        r'(?m)(?:^|\n)\s*(?:' + '|'.join(escaped) + r')\s*:',
    )

_stream_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _stream_semaphore
    if _stream_semaphore is None:
        _stream_semaphore = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)
    return _stream_semaphore


def _check_text(text: str, field: str) -> None:
    if _FORBIDDEN.search(text):
        raise HTTPException(
            status_code=400,
            detail=f"El campo '{field}' contiene tokens especiales no permitidos.",
        )


class ContextMessage(BaseModel):
    author: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    text:   Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class GenerateRequest(BaseModel):
    chat_name:    Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    is_group:     bool
    participants: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]] = Field(default_factory=list)
    context:      list[ContextMessage] = Field(default_factory=list, max_length=200)
    stream:       bool = True
    max_tokens:   int  = Field(default=300, ge=1, le=2000)
    seed:         int | None = None


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/generate")
@limiter.limit(RATE_LIMIT_GENERATE)
async def generate(req: GenerateRequest, request: Request):
    # Validar que ningún campo tenga tokens especiales inyectados.
    _check_text(req.chat_name, "chat_name")
    for p in req.participants:
        _check_text(p, "participants")
    for m in req.context:
        _check_text(m.author, "context.author")
        _check_text(m.text,   "context.text")

    pair = {
        "chat_name":    req.chat_name,
        "is_group":     req.is_group,
        "participants": req.participants,
        "context":      [{"author": m.author, "text": m.text} for m in req.context],
    }
    user_content = _format_user_prompt(pair)

    options: dict = {
        "num_predict": req.max_tokens,
        "temperature": OLLAMA_TEMPERATURE,
        "top_p":       OLLAMA_TOP_P,
    }
    if req.seed is not None:
        options["seed"] = req.seed

    payload = {
        "model":   MODEL_NAME,
        "prompt":  user_content,    # Ollama aplica el chat template del Modelfile.
        "stream":  req.stream,
        "options": options,
    }

    # Construir el stop dinámico para este request: el modelo no debería
    # generar prefijos de autor de los participantes del chat.
    dynamic_stop = _make_dynamic_stop(req.chat_name, req.participants)

    def find_stop(text: str) -> Optional[re.Match]:
        """Busca cualquier patrón stop (estático o dinámico) en el texto."""
        m = _OUTPUT_STOP.search(text)
        if dynamic_stop is not None:
            md = dynamic_stop.search(text)
            if md and (m is None or md.start() < m.start()):
                m = md
        return m

    def emit_clean(slice_text: str) -> str:
        """Aplica filtros de output (noise / mensajes de sistema)."""
        clean = _OUTPUT_NOISE.sub('', slice_text)
        return f"data: {json.dumps({'token': clean})}\n\n" if clean else ""

    if req.stream:
        async def stream_tokens():
            full_text = ""    # acumulado de Ollama
            sent_until = 0    # índice hasta donde ya se envió al frontend

            async with _get_semaphore():
                try:
                    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                            done_payload = None
                            stopped_early = False

                            async for line in response.aiter_lines():
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                except json.JSONDecodeError:
                                    logger.warning("Non-JSON line from Ollama: %s", line[:80])
                                    continue

                                full_text += data.get("response", "")

                                # Si aparece un patrón "stop", cortamos ahí: enviamos
                                # solo lo previo al patrón y terminamos el stream.
                                m = find_stop(full_text)
                                if m:
                                    full_text = full_text[:m.start()].rstrip()
                                    new_text = full_text[sent_until:]
                                    chunk = emit_clean(new_text)
                                    if chunk:
                                        yield chunk
                                    sent_until = len(full_text)
                                    stopped_early = True
                                    eval_count = data.get("eval_count", 0)
                                    eval_duration = data.get("eval_duration", 1) or 1
                                    tps = round(eval_count / (eval_duration / 1e9), 1)
                                    yield f"data: {json.dumps({'done': True, 'eval_count': eval_count, 'tokens_per_sec': tps})}\n\n"
                                    yield "data: [DONE]\n\n"
                                    break

                                # Solo enviamos hasta los últimos _HOLD_CHARS
                                # caracteres del buffer (por si están armando
                                # un patrón stop incompleto).
                                safe_until = max(0, len(full_text) - _HOLD_CHARS)
                                if safe_until > sent_until:
                                    chunk = emit_clean(full_text[sent_until:safe_until])
                                    if chunk:
                                        yield chunk
                                    sent_until = safe_until

                                if data.get("done"):
                                    done_payload = data
                                    break

                            # Si terminó por done normal (no stop), drenamos lo
                            # que queda en el buffer de hold.
                            if not stopped_early and done_payload is not None:
                                tail = full_text[sent_until:]
                                chunk = emit_clean(tail)
                                if chunk:
                                    yield chunk
                                eval_count = done_payload.get("eval_count", 0)
                                eval_duration = done_payload.get("eval_duration", 1) or 1
                                tps = round(eval_count / (eval_duration / 1e9), 1)
                                yield f"data: {json.dumps({'done': True, 'eval_count': eval_count, 'tokens_per_sec': tps})}\n\n"
                                yield "data: [DONE]\n\n"
                except httpx.TimeoutException:
                    logger.exception("Ollama timeout (stream)")
                    yield f"data: {json.dumps({'error': 'upstream_timeout'})}\n\n"
                except Exception:
                    logger.exception("Streaming error")
                    yield f"data: {json.dumps({'error': 'internal'})}\n\n"

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            raw = response.json().get("response", "")
            # Misma sanitización que el path stream.
            m = find_stop(raw)
            if m:
                raw = raw[:m.start()].rstrip()
            cleaned = _OUTPUT_NOISE.sub('', raw).strip()
            return {"text": cleaned}
    except httpx.TimeoutException:
        logger.exception("Ollama timeout (non-stream)")
        raise HTTPException(status_code=504, detail="upstream_timeout")
    except Exception:
        logger.exception("Non-stream generate error")
        raise HTTPException(status_code=500, detail="internal")


@app.get("/chats")
async def list_chats():
    """Devuelve la lista de chats parseados disponibles para usar en la UI."""
    if not PAIRS_FILE.exists():
        return {"chats": [], "reason": "no_pairs_file"}
    seen: dict[str, dict] = {}
    with open(PAIRS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = pair.get("chat_name")
            if not name or name in seen:
                continue
            seen[name] = {
                "chat_name":    name,
                "is_group":     bool(pair.get("is_group")),
                "participants": pair.get("participants", []),
            }
    chats = sorted(seen.values(), key=lambda c: (c["is_group"], c["chat_name"].lower()))
    return {"chats": chats}


class FeedbackRequest(BaseModel):
    chat_name:        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    is_group:         bool
    participants:     list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]] = Field(default_factory=list)
    received_message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    options:          list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=0, max_length=4000)]]
    chosen_idx:       int = Field(ge=0)
    seeds:            list[int] = Field(default_factory=list)


@app.post("/feedback")
async def save_feedback(req: FeedbackRequest):
    """Guarda el feedback (qué opción eligió el usuario) para reentrenamiento futuro."""
    if req.chosen_idx >= len(req.options):
        raise HTTPException(status_code=400, detail="chosen_idx fuera de rango")

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **req.model_dump(),
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"ok": True, "saved_to": str(FEEDBACK_FILE)}


@app.get("/health")
@limiter.limit(RATE_LIMIT_HEALTH)
async def health(request: Request):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r      = await client.get(OLLAMA_TAGS)
            models = [m["name"] for m in r.json().get("models", [])]
            model_loaded = any(MODEL_NAME in m for m in models)
            result = {"status": "ok" if model_loaded else "degraded", "models": models}
            if not model_loaded:
                result["reason"] = "model_not_loaded"
            return result
    except Exception:
        logger.exception("Health check error")
        return {"status": "error", "detail": "internal"}
