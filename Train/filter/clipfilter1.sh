#!/bin/bash

# CLIP 花瓶图像过滤器 - 一键运行脚本
# 用途: 自动过滤花瓶数据集，去除低质量图像

# 默认路径
INPUT_DIR="./images3/1"
OUTPUT_DIR="./filtered_vases"

echo "📍 数据集路径:"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 错误: 输入目录不存在!"
    echo "请检查路径: $INPUT_DIR"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "clipfilter1.py" ]; then
    echo "❌ 错误: 找不到 clipfilter1.py 脚本!"
    echo "请确保脚本文件在当前目录中"
    exit 1
fi

# 检查依赖包
echo "🔍 检查依赖包..."
python3 -c "import torch, transformers, PIL, numpy, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  缺少依赖包，正在安装..."
    pip install torch torchvision transformers Pillow numpy tqdm
    if [ $? -ne 0 ]; then
        echo "❌ 依赖包安装失败!"
        exit 1
    fi
    echo "✅ 依赖包安装完成!"
fi

# 显示菜单
echo ""
echo "请选择操作模式:"
echo "1) 🔍 仅分析数据集质量分布 (推荐首次使用)"
echo "2) 🎯 自动过滤 (先分析后过滤)"
echo "3) ⚙️  自定义阈值过滤"
echo "4) 💻 强制使用CPU模式"
echo "5) 🎮 使用GPU 1"
echo "6) 📊 快速分析 (100个样本)"
echo ""

read -p "请输入选项 (1-6): " choice

case $choice in
        1)
        echo "🔍 开始分析数据集质量分布..."
        python3 clipfilter1.py --analyze_only --input_dir "$INPUT_DIR" --auto_select_device
        ;;
    2)
        echo "🎯 开始自动过滤..."
        python3 clipfilter1.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --model_cache_dir "./model" --auto_select_device
        ;;
    3)
        read -p "请输入阈值 (例如: 0.15): " threshold
        echo "⚙️  使用自定义阈值 $threshold 进行过滤..."
        python3 clipfilter1.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --threshold "$threshold" --model_cache_dir "./model" --auto_select_device
        ;;
    4)
        echo "💻 使用CPU模式进行过滤..."
        python3 clipfilter1.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --device cpu --model_cache_dir "./model"
        ;;
    5)
        echo "🎮 使用GPU 1进行过滤..."
        python3 clipfilter1.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --device cuda:1 --model_cache_dir "./model"
        ;;
    6)
        echo "📊 快速分析 (100个样本)..."
        python3 clipfilter1.py --analyze_only --input_dir "$INPUT_DIR" --sample_size 100 --auto_select_device
        ;;
    *)
        echo "❌ 无效选项，退出"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "✅ 操作完成!"

if [ "$choice" = "2" ] || [ "$choice" = "3" ] || [ "$choice" = "4" ] || [ "$choice" = "5" ]; then
if [ "$choice" = "2" ] || [ "$choice" = "3" ] || [ "$choice" = "4" ]; then
    if [ -d "$OUTPUT_DIR" ]; then
        echo "📁 结果目录: $OUTPUT_DIR"
        echo "📊 统计信息:"

        if [ -d "$OUTPUT_DIR/accepted" ]; then
            accepted_count=$(find "$OUTPUT_DIR/accepted" -type f | wc -l)
            echo "   ✅ 接受的图像: $accepted_count 个"
        fi

        if [ -d "$OUTPUT_DIR/rejected" ]; then
            rejected_count=$(find "$OUTPUT_DIR/rejected" -type f | wc -l)
            echo "   ❌ 拒绝的图像: $rejected_count 个"
        fi

        if [ -f "$OUTPUT_DIR/filtering_report.json" ]; then
            echo "   📋 详细报告: $OUTPUT_DIR/filtering_report.json"
        fi
    fi
fi
