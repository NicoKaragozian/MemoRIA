#!/usr/bin/env bash
# Mergea el adaptador LoRA con el modelo base usando MLX-LM
# Uso: bash scripts/merge_mlx.sh

set -e

MLX_MODEL="./models/gemma3-4b-4bit"
ADAPTER="./memoria-lora"
MERGED="./memoria-merged"

echo "Mergeando adaptador LoRA con modelo base..."
mlx_lm.fuse \
  --model "$MLX_MODEL" \
  --adapter-path "$ADAPTER" \
  --save-path "$MERGED" \
  --dequantize

echo "✓ Modelo mergeado en $MERGED (bf16, listo para convertir a GGUF)"
