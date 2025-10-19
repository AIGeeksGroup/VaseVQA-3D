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

swift export \
    --adapters ./output/checkpoint-600 \
    --merge_lora true