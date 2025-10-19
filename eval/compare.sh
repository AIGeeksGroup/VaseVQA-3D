#!/bin/bash

# 批量评估脚本
# 用于批量处理文件夹下的所有模型结果文件

# 设置路径
CAPTION_DIR="./data/captions"
GROUND_TRUTH="./data/groundTruth.json"
SCRIPT_PATH="compare.py"

# 检查脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ 评估脚本不存在: $SCRIPT_PATH"
    exit 1
fi

# 检查目录是否存在
if [ ! -d "$CAPTION_DIR" ]; then
    echo "❌ 目录不存在: $CAPTION_DIR"
    exit 1
fi

echo "🚀 开始批量评估模型结果..."
echo "📁 处理目录: $CAPTION_DIR"
echo "="*80

# 计数器
total_files=0
success_count=0
failed_count=0

# 创建结果汇总文件
summary_file="batch_evaluation_summary.json"
echo "[" > "$summary_file"

# 遍历所有以image_开头的JSON文件
for file in "$CAPTION_DIR"/image_*.json; do
    # 检查文件是否存在（避免通配符没有匹配到文件的情况）
    if [ ! -f "$file" ]; then
        continue
    fi

    # 跳过groundTruth.json文件
    if [[ "$file" == *"groundTruth.json" ]]; then
        continue
    fi

    total_files=$((total_files + 1))
    filename=$(basename "$file")
    model_name=${filename#image_}
    model_name=${model_name%.json}

    echo ""
    echo "📊 处理模型 [$total_files]: $model_name"
    echo "📄 文件: $filename"
    echo "-"*60

    # 执行评估
    if python "$SCRIPT_PATH" --generated "$file" --ground_truth "$GROUND_TRUTH"; then
        success_count=$((success_count + 1))
        echo "✅ $model_name 评估完成"

        # 检查是否生成了结果文件
        result_file="compare_${model_name}.json"
        if [ -f "$result_file" ]; then
            # 添加到汇总文件（除了第一个文件外，前面加逗号）
            if [ $success_count -gt 1 ]; then
                echo "," >> "$summary_file"
            fi
            cat "$result_file" >> "$summary_file"
        fi
    else
        failed_count=$((failed_count + 1))
        echo "❌ $model_name 评估失败"
    fi
done

# 完成汇总文件
echo "]" >> "$summary_file"

echo ""
echo "="*80
echo "🎉 批量评估完成！"
echo "📈 统计结果:"
echo "   总文件数: $total_files"
echo "   成功: $success_count"
echo "   失败: $failed_count"
echo "   成功率: $(( success_count * 100 / total_files ))%"
echo ""
echo "📋 生成的文件:"
echo "   - 各模型详细结果: compare_*.json"
echo "   - 各模型评估报告: compare_*.txt"
echo "   - 批量评估汇总: $summary_file"
echo ""

# 生成简化的对比表格
echo "📊 生成模型对比表格..."
python3 << 'EOF'
import json
import os
from typing import List, Dict

def generate_comparison_table():
    """生成模型对比表格"""

    # 收集所有compare_*.json文件
    compare_files = [f for f in os.listdir('.') if f.startswith('compare_') and f.endswith('.json')]

    if not compare_files:
        print("❌ 未找到评估结果文件")
        return

    results = []

    for file in compare_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            model_name = data.get('metadata', {}).get('model_name', file.replace('compare_', '').replace('.json', ''))

            result = {
                'model': model_name,
                'clip_score': data.get('clip_score', 0),
                'fid_score': data.get('fid_score', 0),
                'semantic_similarity': data.get('semantic_similarity', 0),
                'r_at_1': data.get('r_at_1', 0),
                'r_at_5': data.get('r_at_5', 0),
                'r_at_10': data.get('r_at_10', 0),
                'lexical_similarity': data.get('lexical_similarity', 0),
                'overall_score': data.get('overall_score', 0),
                'total_pairs': data.get('metadata', {}).get('total_pairs', 0)
            }

            results.append(result)

        except Exception as e:
            print(f"❌ 处理文件 {file} 时出错: {e}")

    if not results:
        print("❌ 没有有效的评估结果")
        return

    # 按overall_score排序
    results.sort(key=lambda x: x['overall_score'], reverse=True)

    # 生成Markdown表格
    with open('model_comparison_table.md', 'w', encoding='utf-8') as f:
        f.write("# 模型评估对比表格\n\n")
        f.write("## 综合排名\n\n")

        # 表头
        f.write("| 排名 | 模型名称 | 整体评分 | CLIP Score | 语义相似度 | R@10 | R@5 | R@1 | 词汇重叠 | FID Score | 样本数 |\n")
        f.write("|------|----------|----------|------------|------------|------|-----|-----|----------|-----------|--------|\n")

        # 数据行
        for i, result in enumerate(results, 1):
            f.write(f"| {i} | {result['model']} | {result['overall_score']:.4f} | {result['clip_score']:.4f} | {result['semantic_similarity']:.4f} | {result['r_at_10']:.2%} | {result['r_at_5']:.2%} | {result['r_at_1']:.2%} | {result['lexical_similarity']:.4f} | {result['fid_score']:.4f} | {result['total_pairs']} |\n")

        f.write("\n## 指标说明\n\n")
        f.write("- **整体评分**: 综合所有指标的加权平均分 (0-1，越高越好)\n")
        f.write("- **CLIP Score**: CLIP模型文本相似度 (0-1，越高越好)\n")
        f.write("- **语义相似度**: 基于CLIP的语义相似度 (0-1，越高越好)\n")
        f.write("- **R@K**: Top-K检索准确率 (百分比，越高越好)\n")
        f.write("- **词汇重叠**: Jaccard相似度 (0-1，越高越好)\n")
        f.write("- **FID Score**: Fréchet Inception Distance (越低越好)\n")
        f.write("- **样本数**: 参与评估的样本对数量\n")

    # 生成CSV文件
    with open('model_comparison_table.csv', 'w', encoding='utf-8') as f:
        f.write("排名,模型名称,整体评分,CLIP_Score,语义相似度,R@10,R@5,R@1,词汇重叠,FID_Score,样本数\n")

        for i, result in enumerate(results, 1):
            f.write(f"{i},{result['model']},{result['overall_score']:.4f},{result['clip_score']:.4f},{result['semantic_similarity']:.4f},{result['r_at_10']:.4f},{result['r_at_5']:.4f},{result['r_at_1']:.4f},{result['lexical_similarity']:.4f},{result['fid_score']:.4f},{result['total_pairs']}\n")

    print(f"✅ 生成对比表格完成:")
    print(f"   - Markdown格式: model_comparison_table.md")
    print(f"   - CSV格式: model_comparison_table.csv")
    print(f"   - 共对比 {len(results)} 个模型")

    # 显示前5名
    print(f"\n🏆 Top 5 模型:")
    for i, result in enumerate(results[:5], 1):
        print(f"   {i}. {result['model']}: {result['overall_score']:.4f}")

if __name__ == "__main__":
    generate_comparison_table()
EOF

echo "🎯 批量评估和对比分析完成！"
echo "📁 查看结果文件以获取详细信息。"
