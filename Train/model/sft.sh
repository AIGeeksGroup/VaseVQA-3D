#!/bin/bash
set -x

# 列出所有可用的conda环境
conda info --envs

# 激活正确的conda环境 
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swift_env

# 设置代理（如需要）
# export http_proxy=http://your_proxy:port
# export https_proxy=http://your_proxy:port

# 确认swift命令是否可用
which swift

# 指定data路径
export DATASET_PATH='./data/train.jsonl'

# 设置多卡训练参数（如需要）
# export CUDA_VISIBLE_DEVICES=0,1
# export NPROC_PER_NODE=2  # 使用的GPU数量
# export MASTER_PORT=29501

echo "开始训练..."
# 显存资源：24GiB
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift sft \
    --model './models/Qwen2.5-VL-7B-Instruct' \
    --dataset './data/video_training_dataset.json' \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output/7B \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4