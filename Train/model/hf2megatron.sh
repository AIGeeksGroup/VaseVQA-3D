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

CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model './models/Qwen2.5-VL-3B-Instruct' \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir output/Qwen2.5-VL-3B-Instruct-mcore \
    --test_convert_precision true