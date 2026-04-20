#!/usr/bin/env bash
# Fine-tuning con MLX-LM en Mac Apple Silicon (path principal para M5 16 GB)
# Uso: bash scripts/finetune_mlx.sh [iters]

set -euo pipefail

ITERS=${1:-1000}
MLX_MODEL="./models/gemma3-4b-4bit"
ADAPTER="./memoria-lora"
DATA="data/dataset"

# Paso 1: cuantizar el modelo base si no existe
if [ ! -d "$MLX_MODEL" ]; then
  echo "⬇ Descargando y cuantizando Gemma 4 E2B a 4-bit..."
  mlx_lm.convert \
    --hf-path google/gemma-3-4b-it \
    --mlx-path "$MLX_MODEL" \
    --quantize --q-bits 4 --q-group-size 64
  echo "✓ Modelo cuantizado en $MLX_MODEL"
fi

# Paso 2: fine-tuning LoRA
echo "Iniciando fine-tuning ($ITERS iters)..."
mlx_lm.lora \
  --model "$MLX_MODEL" \
  --train \
  --data "$DATA" \
  --iters "$ITERS" \
  --batch-size 1 \
  --num-layers 16 \
  --learning-rate 2e-4 \
  --adapter-path "$ADAPTER" \
  --grad-checkpoint \
  --save-every 200

echo "✓ Adaptador LoRA guardado en $ADAPTER"
