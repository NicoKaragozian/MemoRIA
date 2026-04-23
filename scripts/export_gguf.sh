#!/usr/bin/env bash
# Convierte el modelo mergeado a GGUF Q4_K_M y lo registra en Ollama
# Requiere llama.cpp clonado en ./llama.cpp
# Uso: bash scripts/export_gguf.sh

set -euo pipefail

MERGED="./memoria-merged"
GGUF="./memoria-q4.gguf"
LLAMA_CPP="./llama.cpp"

# Clonar llama.cpp si no existe
if [ ! -d "$LLAMA_CPP" ]; then
  echo "Clonando llama.cpp..."
  git clone https://github.com/ggerganov/llama.cpp --depth=1 "$LLAMA_CPP"
  pip install -q -r "$LLAMA_CPP/requirements.txt"
fi

# Convertir a GGUF Q4_K_M
echo "Convirtiendo a GGUF Q4_K_M..."
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" \
  --outfile "$GGUF" \
  --outtype q4_k_m

echo "GGUF generado: $GGUF"

# Registrar en Ollama con la ruta local real
echo "Registrando en Ollama como 'memoria'..."
bash scripts/create_ollama_model.sh

echo "Listo. Probar con:"
echo "  ollama run memoria \"[CASUAL] Contame algo de tu finde\""
