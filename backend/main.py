import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Annotated, Literal

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
    CORS_ORIGINS, MAX_CONCURRENT_STREAMS, MODEL_NAME,
    OLLAMA_TAGS, OLLAMA_TEMPERATURE, OLLAMA_TIMEOUT, OLLAMA_TOP_P, OLLAMA_URL,
    RATE_LIMIT_GENERATE, RATE_LIMIT_HEALTH,
)

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

REGISTER_TAGS = {
    "casual":       "[CASUAL]",
    "professional": "[EMAIL-PROF]",
    "academic":     "[ACADÉMICO]",
}

# Tokens especiales de Gemma y literales de registro que no deben inyectarse
_FORBIDDEN = re.compile(
    r'<start_of_turn>|<end_of_turn>|<bos>|<eos>|<\|[^|]+\|>'
    r'|\[CASUAL\]|\[EMAIL-PROF\]|\[ACADÉMICO\]',
    re.IGNORECASE,
)

_stream_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _stream_semaphore
    if _stream_semaphore is None:
        _stream_semaphore = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)
    return _stream_semaphore


def _sanitize_prompt(text: str) -> str:
    if _FORBIDDEN.search(text):
        raise HTTPException(
            status_code=400,
            detail="El prompt contiene tokens especiales no permitidos.",
        )
    return text


class GenerateRequest(BaseModel):
    prompt:     Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    register:   Literal["casual", "professional", "academic"] = "casual"
    stream:     bool    = True
    max_tokens: int     = Field(default=500, ge=1, le=2000)
    seed:       int | None = None


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/generate")
@limiter.limit(RATE_LIMIT_GENERATE)
async def generate(req: GenerateRequest, request: Request):
    prompt_clean = _sanitize_prompt(req.prompt)
    tag          = REGISTER_TAGS.get(req.register, "[CASUAL]")
    full_prompt  = f"{tag} {prompt_clean}"

    options: dict = {
        "num_predict": req.max_tokens,
        "temperature": OLLAMA_TEMPERATURE,
        "top_p":       OLLAMA_TOP_P,
    }
    if req.seed is not None:
        options["seed"] = req.seed

    payload = {
        "model":   MODEL_NAME,
        "prompt":  full_prompt,
        "stream":  req.stream,
        "options": options,
    }

    if req.stream:
        async def stream_tokens():
            async with _get_semaphore():
                try:
                    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                        async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                            async for line in response.aiter_lines():
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                except json.JSONDecodeError:
                                    logger.warning("Non-JSON line from Ollama: %s", line[:80])
                                    continue
                                yield f"data: {json.dumps({'token': data.get('response', '')})}\n\n"
                                if data.get("done"):
                                    eval_count    = data.get("eval_count", 0)
                                    eval_duration = data.get("eval_duration", 1) or 1
                                    tps = round(eval_count / (eval_duration / 1e9), 1)
                                    yield f"data: {json.dumps({'done': True, 'eval_count': eval_count, 'tokens_per_sec': tps})}\n\n"
                                    yield "data: [DONE]\n\n"
                                    break
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
            return {"text": response.json().get("response", "")}
    except httpx.TimeoutException:
        logger.exception("Ollama timeout (non-stream)")
        raise HTTPException(status_code=504, detail="upstream_timeout")
    except Exception:
        logger.exception("Non-stream generate error")
        raise HTTPException(status_code=500, detail="internal")


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
