export PYTHONWARNINGS="ignore"
export NCCL_NVLS_ENABLE=0
export PYTHONUNBUFFERED=1

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_port=29500 \
    -m loomtrain.scripts.train_dpo \
    --model-path meta-llama/Llama-3.1-8B-Instruct/ \
    --dataset-paths /path/to/dataset \
    --train-samples -1 \
    --val-samples 10 \
    --max-data-length 128000 \
    --packing-length 4000 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --val-interval 40 \
    --val-batch-size 2 \
    --ckpt-interval 2 \
    --weight-interval 4 \
    --prompt-key chat_template \
    --chosen-key chosen \
    --rejected-key rejects \
    --num-rejects 1 \
    --lr 2e-6 \
    --beta 2.5 \
    --nll-loss-weight 0.05 \
    --enable-micro-bar true \
    --save-dir ./test-dpo 