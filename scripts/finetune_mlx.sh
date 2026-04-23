#!/usr/bin/env bash
# Fine-tuning con MLX-LM en Mac Apple Silicon (M5 16 GB)
# Modelo: Qwen 3 4B Instruct | Formato dataset: chat | Loss: solo sobre assistant
# Uso: bash scripts/finetune_mlx.sh [iters]
#      bash scripts/finetune_mlx.sh 50   ← dry-run para verificar pipeline

set -euo pipefail

ITERS=${1:-900}
MLX_MODEL="./models/qwen3-4b-4bit"
ADAPTER="./memoria-lora"
DATA="data/dataset"

# Paso 1: cuantizar el modelo base si no existe
if [ ! -d "$MLX_MODEL" ]; then
  echo "Descargando y cuantizando Qwen3-4B-Instruct-2507 a 4-bit..."
  python3 -m mlx_lm convert \
    --hf-path Qwen/Qwen3-4B-Instruct-2507 \
    --mlx-path "$MLX_MODEL" \
    --quantize --q-bits 4 --q-group-size 64
  echo "Modelo cuantizado en $MLX_MODEL"
fi

# Paso 2: fine-tuning LoRA con formato chat y mask-prompt
echo "Iniciando fine-tuning ($ITERS iters) ..."
python3 -m mlx_lm lora \
  --model "$MLX_MODEL" \
  --train \
  --data "$DATA" \
  --fine-tune-type lora \
  --mask-prompt \
  --num-layers -1 \
  --batch-size 1 \
  --learning-rate 5e-5 \
  --iters "$ITERS" \
  --val-batches 25 \
  --steps-per-eval 75 \
  --save-every 150 \
  --grad-checkpoint \
  --adapter-path "$ADAPTER" \
  --lora-parameters '{"rank": 16, "alpha": 32, "dropout": 0.05, "scale": 10.0}'

echo "Adaptador LoRA guardado en $ADAPTER"
