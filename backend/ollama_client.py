"""
Utilidades de cliente Ollama reutilizables en tests (httpx.MockTransport).
main.py contiene la lógica de streaming completa; este módulo expone helpers.
"""
from __future__ import annotations
import json
from typing import AsyncGenerator

import httpx


async def stream_generate(
    ollama_url: str,
    payload: dict,
    timeout: int = 120,
    client: httpx.AsyncClient | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Genera tokens desde Ollama en streaming via /api/generate (legado).
    Yields dicts con claves: 'response', 'done', 'eval_count', 'eval_duration'.
    Acepta un client inyectado para facilitar tests con MockTransport.
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        async with client.stream("POST", ollama_url, json=payload) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield data
                if data.get("done"):
                    break
    finally:
        if own_client:
            await client.aclose()


async def chat_stream(
    ollama_url: str,
    messages: list[dict],
    model: str,
    options: dict | None = None,
    timeout: int = 120,
    client: httpx.AsyncClient | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Genera tokens desde Ollama en streaming via /api/chat.
    Yields dicts con claves: 'message' (dict con 'content'), 'done', 'eval_count', 'eval_duration'.
    Acepta un client inyectado para facilitar tests con MockTransport.
    """
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   True,
    }
    if options:
        payload["options"] = options

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        async with client.stream("POST", ollama_url, json=payload) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield data
                if data.get("done"):
                    break
    finally:
        if own_client:
            await client.aclose()


async def check_model_loaded(tags_url: str, model_name: str, timeout: int = 5) -> bool:
    """Devuelve True si model_name aparece en la lista de modelos de Ollama."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(tags_url)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(model_name in m for m in models)
