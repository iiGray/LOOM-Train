# deepspeed --module scripts.train_sft \
#     --model-path /data/hf_models/Meta-Llama-3.1-8B-Instruct/ \
#     --dataset-paths /data/datas/Skywork-Reward-80k-Formatted \
#     --prompt-key chat_template \
#     --response-key golden \
#     --sample-counts 80 \
#     --sample-counts-eval 10 \
#     --max-data-length 128000 \
#     --max-packing-length 0 \
#     --micro-batch-size 1 \
#     --global-batch-size 2 \
#     --save-dir ./ \
#     --save-name Test-Loom-Llama3.1 \
#     --data-cache-dir ./datasetfiles \
#     --tensorboard-logdir ./tensorboard \


# deepspeed --module scripts.train_sft \
#     --model-path /data/hf_models/Meta-Llama-3.1-8B-Instruct/ \
#     --dataset-paths /data/datas/Skywork-Reward-80k-Formatted \
#     --prompt-key chat_template \
#     --response-key golden \
#     --sample-counts 80 \
#     --sample-counts-eval 10 \
#     --max-data-length 128000 \
#     --max-packing-length 0 \
#     --micro-batch-size 1 \
#     --global-batch-size 2 \
#     --save-dir ./ \
#     --save-name Test-Loom-Llama3.1 \
#     --data-cache-dir ./datasetfiles \
#     --tensorboard-logdir ./tensorboard \

export PYTHONWARNINGS="ignore"
export NCCL_NVLS_ENABLE=0
export PYTHONUNBUFFERED=1

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_port=29500 \
    -m loomtrain.scripts.train_sft \
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
    --global-batch-size 1 \
    --val-interval 40 \
    --val-batch-size 2 \
    --ckpt-interval 2 \
    --weight-interval 4 \
    --enable-micro-bar true \
    --save-dir ./test