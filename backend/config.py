import os

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434/api/generate")
OLLAMA_TAGS  = os.getenv("OLLAMA_TAGS",  "http://localhost:11434/api/tags")
MODEL_NAME   = os.getenv("OLLAMA_MODEL", "memoria")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
