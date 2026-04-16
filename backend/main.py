import json
import logging
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import CORS_ORIGINS, MODEL_NAME, OLLAMA_TAGS, OLLAMA_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memoria")

app = FastAPI(title="MemoRIA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

REGISTER_TAGS = {
    "casual":       "[CASUAL]",
    "professional": "[EMAIL-PROF]",
    "academic":     "[ACADÉMICO]",
}


class GenerateRequest(BaseModel):
    prompt:     str            = Field(..., min_length=1, max_length=2000)
    register:   Literal["casual", "professional", "academic"] = "casual"
    stream:     bool           = True
    max_tokens: int            = Field(default=500, ge=1, le=2000)


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/generate")
async def generate(req: GenerateRequest):
    tag         = REGISTER_TAGS.get(req.register, "[CASUAL]")
    full_prompt = f"{tag} {req.prompt}"

    payload = {
        "model":   MODEL_NAME,
        "prompt":  full_prompt,
        "stream":  req.stream,
        "options": {
            "num_predict": req.max_tokens,
            "temperature": 0.8,
            "top_p":       0.9,
        },
    }

    if req.stream:
        async def stream_tokens():
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                logger.warning("Non-JSON line from Ollama: %s", line[:80])
                                continue
                            token = data.get("response", "")
                            yield f"data: {json.dumps({'token': token})}\n\n"
                            if data.get("done"):
                                yield "data: [DONE]\n\n"
                                break
            except Exception as e:
                logger.error("Streaming error: %s", e)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        return {"text": response.json().get("response", "")}


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r      = await client.get(OLLAMA_TAGS)
            models = [m["name"] for m in r.json().get("models", [])]
            return {"status": "ok", "models": models}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
