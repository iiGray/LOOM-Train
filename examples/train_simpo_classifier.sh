export PYTHONWARNINGS="ignore"
export NCCL_NVLS_ENABLE=0
export PYTHONUNBUFFERED=1

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_port=29500 \
    -m loomtrain.scripts.train_simpo \
    --model-path /data/preference_dataset \
    --dataset-paths meta-llama/Meta-Llama-3.1-8B-Instruct/ \
    --train-samples 800 \
    --val-samples 10 \
    --max-data-length 128000 \
    --packing-length 128000 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --prompt-key chat_template \
    --chosen-key chosen \
    --rejected-key rejects \
    --num-rejects 1 \
    --lr 2e-6 \
    --gamma 0.5 \
    --beta 2.5 \
    --enable-micro-bar true \
    --save-dir ./test 