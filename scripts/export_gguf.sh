#!/usr/bin/env bash
# Convierte el modelo mergeado a GGUF Q4_K_M y lo registra en Ollama
# Requiere llama.cpp clonado en ./llama.cpp
# Uso: bash scripts/export_gguf.sh

set -e

MERGED="./memoria-merged"
GGUF="./memoria-q4.gguf"
LLAMA_CPP="./llama.cpp"
HF_CACHE="$HOME/.cache/huggingface/hub/models--google--gemma-3-4b-it"

# Clonar llama.cpp si no existe (o está vacío)
if [ ! -d "$LLAMA_CPP" ] || [ -z "$(ls -A "$LLAMA_CPP" 2>/dev/null)" ]; then
  echo "Clonando llama.cpp..."
  rm -rf "$LLAMA_CPP"
  git clone https://github.com/ggerganov/llama.cpp --depth=1 "$LLAMA_CPP"
  pip install -q -r "$LLAMA_CPP/requirements.txt"
fi

# Asegurar tokenizer.model en memoria-merged (Gemma 3 lo necesita para
# que convert_hf_to_gguf.py use SentencePiece y no caiga en el path BPE)
if [ ! -f "$MERGED/tokenizer.model" ]; then
  TOK=$(find "$HF_CACHE" -name "tokenizer.model" 2>/dev/null | head -1)
  if [ -n "$TOK" ]; then
    echo "Copiando tokenizer.model desde el cache de HuggingFace..."
    cp "$TOK" "$MERGED/tokenizer.model"
  else
    echo "WARNING: tokenizer.model no encontrado en $HF_CACHE — la conversión puede fallar." >&2
  fi
fi

# Convertir a GGUF
echo "Convirtiendo a GGUF Q8_0..."
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" \
  --outfile "$GGUF" \
  --outtype q8_0

echo "✓ GGUF generado: $GGUF"

# Registrar en Ollama con la ruta local real
echo "Registrando en Ollama como 'memoria'..."
bash scripts/create_ollama_model.sh

echo "✓ Listo. Probar con:"
echo "  ollama run memoria \"[EMAIL-PROF] Escribí un email sobre el proyecto MemoRIA\""
