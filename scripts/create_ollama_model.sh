#!/usr/bin/env bash
# Registra el modelo en Ollama para uso local (fuera de Docker).
# Requiere que memoria-q4.gguf exista en la raíz del repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
GGUF_PATH="$REPO_ROOT/memoria-q4.gguf"
MODELFILE="$REPO_ROOT/Modelfile"

if [ ! -f "$GGUF_PATH" ]; then
  echo "ERROR: $GGUF_PATH no existe. Corré primero export_gguf.sh." >&2
  exit 1
fi

TMPFILE="$(mktemp /tmp/Modelfile.local.XXXXXX)"
trap 'rm -f "$TMPFILE"' EXIT

# Reemplaza el path absoluto de Docker por la ruta local real
sed "s|FROM /models/memoria.gguf|FROM $GGUF_PATH|" "$MODELFILE" > "$TMPFILE"

echo "Creando modelo 'memoria' en Ollama desde $GGUF_PATH ..."
ollama create memoria -f "$TMPFILE"
echo "Listo. Verificá con: ollama list"
