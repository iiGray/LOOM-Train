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


export NCCL_NVLS_ENABLE=0

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_port=29500 \
    -m scripts.train_sft \
    --model-path /data/hf_models/Meta-Llama-3.1-8B-Instruct/ \
    --dataset-paths /data/lcm_lab/jbb/datas/Skywork-Reward-80k-Formatted \
    --prompt-key chat_template \
    --response-key chosen \
    --train-samples 80 \
    --val-samples 10 \
    --max-data-length 128000 \
    --packing-length 0 \
    --micro-batch-size 1 \
    --logtype '' \
    --global-batch-size 2 \
    --save-dir ./Test-Loom-Llama3.1