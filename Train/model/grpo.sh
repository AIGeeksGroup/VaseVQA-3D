#!/bin/bash
set -x

# 列出所有可用的conda环境
conda info --envs

# 激活正确的conda环境 
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swift_env  # 修改为你的环境名称

# 确认swift命令是否可用
which swift

# 指定data路径
export DATASET_PATH='./data/train.jsonl'
export TRACK_SAMPLES_COUNT=1

# 设置多卡训练参数
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1  # 使用的GPU数量
export MASTER_PORT=29501

swift rlhf \
    --rlhf_type grpo \
    --model './output/checkpoint-merged' \
    --external_plugins ./plugin/plugin.py \
    --reward_funcs external_vase_acc \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset './data/grpo_video_dataset.json' \
    --max_completion_length 256 \
    --num_train_epochs 1 \
    --target_modules all-linear \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 1000 \
    --save_steps 300 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir output/GRPO_CLEVR_COUNTDOWN \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 1\
    --num_generations 4 \
    --temperature 1.0 \
    --log_completions true \
    --report_to tensorboard \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 \
    --system ./prompt.txt \
    --gradient_checkpointing false \
    --ddp_find_unused_parameters false
