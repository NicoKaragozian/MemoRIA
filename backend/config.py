import os

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434/api/chat")
OLLAMA_TAGS  = os.getenv("OLLAMA_TAGS",  "http://localhost:11434/api/tags")
MODEL_NAME   = os.getenv("OLLAMA_MODEL", "memoria")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")

OLLAMA_TEMPERATURE     = float(os.getenv("OLLAMA_TEMPERATURE",    "0.8"))
OLLAMA_TOP_P           = float(os.getenv("OLLAMA_TOP_P",           "0.9"))
OLLAMA_TIMEOUT         = int(os.getenv("OLLAMA_TIMEOUT",           "120"))
MAX_CONCURRENT_STREAMS = int(os.getenv("MAX_CONCURRENT_STREAMS",   "3"))
RATE_LIMIT_GENERATE    = os.getenv("RATE_LIMIT_GENERATE", "10/minute")
RATE_LIMIT_HEALTH      = os.getenv("RATE_LIMIT_HEALTH",   "30/minute")
