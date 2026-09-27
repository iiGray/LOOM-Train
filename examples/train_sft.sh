#!/usr/bin/env bash
set -euo pipefail

# Start the Ray cluster once: bash linktrain/ray_cluster.sh head
export PYTHONWARNINGS="ignore"
export NCCL_NVLS_ENABLE=0
export PYTHONUNBUFFERED=1

exec python "$(dirname "$0")/../linktrain/scripts/ray_launch.py" \
    --execution=supervised \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_port=29500 \
    "$@" \
    -m linktrain.scripts.train_sft \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct/ \
    --dataset-paths /data/datas/LongMiT \
    --data-cache-dir ./datasetfiles \
    --val-split eval \
    --prompt-key chat_template \
    --response-key golden \
    --train-samples 30 \
    --val-samples 3 \
    --max-data-length 128000 \
    --packing-length 128000 \
    --logtype 'tensorboard' \
    --lr 2e-6 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --val-interval 40 \
    --val-batch-size 2 \
    --ckpt-interval 2 \
    --weight-interval 4 \
    --enable-micro-bar true \
    --save-dir ./test
