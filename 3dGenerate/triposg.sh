#!/bin/bash

echo "activating..."
# use your env path
source env/bin/activate

echo "enveriment activating..."
# 设置PyTorch兼容性
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 设置CUDA兼容性
export CUDA_LAUNCH_BLOCKING=1

# 限制并行处理
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 设置Open3D优化
export OPEN3D_CPU_RENDERING=1

echo "开始批处理图像（强力修复版本 + 按文件名命名输出）..."
echo "此版本会处理指定文件夹中的所有图片，并按原始文件名命名输出文件"
echo "输出文件格式: 原文件名_segmented.png, 原文件名_mesh.glb, 原文件名_multiview.png, 原文件名_textured_4k.glb"
python triposg_batch_threaded.py "$1" 