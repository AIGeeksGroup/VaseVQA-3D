#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caption评估脚本
比较生成的caption和ground truth，计算多个评估指标
包括：FID Score, CLIP Score, R-Precision, 语义相似度等
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from scipy import linalg

# 使用本地模型路径
CLIP_MODEL_PATH = './models/openai/clip-vit-base-patch32'
SENTENCE_MODEL_PATH = './models/sentence-transformers/all-mpnet-base-v2'

class CaptionEvaluator:
    """Caption评估器"""

    def __init__(self):
        """初始化评估器"""
        print("正在初始化评估器...")

        # 初始化CLIP模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

        try:
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
            print("✅ CLIP模型加载成功")
        except Exception as e:
            print(f"❌ CLIP模型加载失败: {e}")
            self.clip_model = None
            self.clip_processor = None

        # 初始化句子嵌入模型
        try:
            self.sentence_model = SentenceTransformer(SENTENCE_MODEL_PATH)
            print("✅ 句子嵌入模型加载成功")
        except Exception as e:
            print(f"❌ 句子嵌入模型加载失败: {e}")
            self.sentence_model = None

        print("✅ 评估器初始化完成")

    def load_json_data(self, file_path: str) -> List[Dict]:
        """加载JSON数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 成功加载 {file_path}，包含 {len(data)} 条数据")
            return data
        except Exception as e:
            print(f"❌ 加载 {file_path} 失败: {e}")
            return []

    def extract_image_id(self, image_name: str) -> str:
        """从图片名提取ID"""
        # 提取UUID部分作为唯一标识
        if '_' in image_name:
            parts = image_name.split('_')
            if len(parts) >= 3:
                return f"{parts[0]}_{parts[1]}"
        return image_name.replace('.jpg', '')

    def match_data(self, generated_data: List[Dict], ground_truth_data: List[Dict]) -> List[Tuple[str, str, str]]:
        """匹配生成数据和ground truth数据"""
        print("正在匹配数据...")

        # 创建ground truth的映射
        gt_map = {}
        for item in ground_truth_data:
            if 'images' in item and len(item['images']) > 0:
                image_name = item['images'][0]
                image_id = self.extract_image_id(image_name)
                gt_map[image_id] = item['caption']

        # 匹配数据
        matched_pairs = []
        skipped_count = 0

        for item in generated_data:
            if 'images' in item and len(item['images']) > 0:
                image_name = item['images'][0]
                image_id = self.extract_image_id(image_name)

                if image_id in gt_map:
                    generated_caption = item['caption']
                    ground_truth_caption = gt_map[image_id]
                    matched_pairs.append((image_id, generated_caption, ground_truth_caption))
                else:
                    skipped_count += 1

        print(f"✅ 匹配完成: {len(matched_pairs)} 对数据匹配成功，{skipped_count} 条数据跳过")
        return matched_pairs

    def calculate_clip_score(self, generated_captions: List[str], ground_truth_captions: List[str]) -> float:
        """计算CLIP Score"""
        if self.clip_model is None or self.clip_processor is None:
            print("❌ CLIP模型未加载，跳过CLIP Score计算")
            return 0.0

        print("正在计算CLIP Score...")

        try:
            # 分批处理，避免内存问题
            batch_size = 32
            all_similarities = []

            for i in range(0, len(generated_captions), batch_size):
                end_idx = min(i + batch_size, len(generated_captions))
                batch_gen = generated_captions[i:end_idx]
                batch_gt = ground_truth_captions[i:end_idx]

                # 编码文本
                generated_inputs = self.clip_processor(text=batch_gen, return_tensors="pt", padding=True, truncation=True).to(self.device)
                gt_inputs = self.clip_processor(text=batch_gt, return_tensors="pt", padding=True, truncation=True).to(self.device)

                with torch.no_grad():
                    generated_features = self.clip_model.get_text_features(**generated_inputs)
                    gt_features = self.clip_model.get_text_features(**gt_inputs)

                    # 归一化
                    generated_features = generated_features / generated_features.norm(dim=-1, keepdim=True)
                    gt_features = gt_features / gt_features.norm(dim=-1, keepdim=True)

                    # 计算余弦相似度
                    similarities = torch.sum(generated_features * gt_features, dim=-1)
                    all_similarities.extend(similarities.cpu().numpy())

            clip_score = np.mean(all_similarities)
            print(f"✅ CLIP Score: {clip_score:.4f}")
            return float(clip_score)

        except Exception as e:
            print(f"❌ CLIP Score计算失败: {e}")
            return 0.0

    def calculate_simple_similarity(self, generated_captions: List[str], ground_truth_captions: List[str]) -> float:
        """计算简单的词汇重叠相似度"""
        print("正在计算词汇重叠相似度...")

        try:
            similarities = []

            for gen_cap, gt_cap in zip(generated_captions, ground_truth_captions):
                # 转换为小写并分词
                gen_words = set(gen_cap.lower().split())
                gt_words = set(gt_cap.lower().split())

                # 计算Jaccard相似度
                intersection = len(gen_words & gt_words)
                union = len(gen_words | gt_words)

                if union == 0:
                    similarity = 0.0
                else:
                    similarity = intersection / union

                similarities.append(similarity)

            avg_similarity = np.mean(similarities)
            print(f"✅ 词汇重叠相似度: {avg_similarity:.4f}")
            return float(avg_similarity)

        except Exception as e:
            print(f"❌ 词汇重叠相似度计算失败: {e}")
            return 0.0

    def calculate_fid_score(self, generated_captions: List[str], ground_truth_captions: List[str]) -> float:
        """计算基于句子嵌入的FID Score"""
        if self.sentence_model is None:
            print("❌ 句子嵌入模型未加载，跳过FID Score计算")
            return 0.0

        print("正在计算FID Score...")

        try:
            # 使用sentence transformer获取文本嵌入
            generated_embeddings = self.sentence_model.encode(generated_captions)
            gt_embeddings = self.sentence_model.encode(ground_truth_captions)

            # 计算均值和协方差
            mu1, sigma1 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
            mu2, sigma2 = gt_embeddings.mean(axis=0), np.cov(gt_embeddings, rowvar=False)

            # 计算FID
            diff = mu1 - mu2

            # 添加小的正则化项以避免数值不稳定
            eps = 1e-6
            sigma1 += eps * np.eye(sigma1.shape[0])
            sigma2 += eps * np.eye(sigma2.shape[0])

            # 使用scipy计算矩阵平方根
            try:
                covmean = linalg.sqrtm(sigma1.dot(sigma2))

                # 检查是否为复数，如果是则取实部
                if np.iscomplexobj(covmean):
                    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                        print("警告: 协方差矩阵平方根包含显著虚部")
                    covmean = covmean.real

                # 计算FID
                fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

            except Exception as e:
                print(f"警告: 矩阵平方根计算失败，使用简化版本: {e}")
                # 简化版本：使用迹的几何平均
                fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.sqrt(np.trace(sigma1) * np.trace(sigma2))

            print(f"✅ FID Score: {fid:.4f}")
            return float(fid)

        except Exception as e:
            print(f"❌ FID Score计算失败: {e}")
            return 0.0

    def calculate_semantic_similarity(self, generated_captions: List[str], ground_truth_captions: List[str]) -> float:
        """计算基于CLIP的语义相似度"""
        if self.clip_model is None or self.clip_processor is None:
            print("❌ CLIP模型未加载，跳过语义相似度计算")
            return 0.0

        print("正在计算语义相似度...")

        try:
            # 使用CLIP计算语义相似度
            similarities = []
            batch_size = 32

            for i in range(0, len(generated_captions), batch_size):
                end_idx = min(i + batch_size, len(generated_captions))
                batch_gen = generated_captions[i:end_idx]
                batch_gt = ground_truth_captions[i:end_idx]

                gen_inputs = self.clip_processor(text=batch_gen, return_tensors="pt", padding=True, truncation=True).to(self.device)
                gt_inputs = self.clip_processor(text=batch_gt, return_tensors="pt", padding=True, truncation=True).to(self.device)

                with torch.no_grad():
                    gen_features = self.clip_model.get_text_features(**gen_inputs)
                    gt_features = self.clip_model.get_text_features(**gt_inputs)

                    # 归一化
                    gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)
                    gt_features = gt_features / gt_features.norm(dim=-1, keepdim=True)

                    # 计算成对相似度
                    batch_similarities = torch.sum(gen_features * gt_features, dim=-1)
                    similarities.extend(batch_similarities.cpu().numpy())

            avg_similarity = np.mean(similarities)
            print(f"✅ 语义相似度: {avg_similarity:.4f}")
            return float(avg_similarity)

        except Exception as e:
            print(f"❌ 语义相似度计算失败: {e}")
            return 0.0

    def calculate_r_precision(self, generated_captions: List[str], ground_truth_captions: List[str]) -> Dict[str, float]:
        """计算R-Precision (R@1, R@5, R@10)"""
        if self.clip_model is None or self.clip_processor is None:
            print("❌ CLIP模型未加载，跳过R-Precision计算")
            return {"r_at_1": 0.0, "r_at_5": 0.0, "r_at_10": 0.0}

        print("正在计算R-Precision...")

        try:
            # 使用CLIP获取所有特征
            def get_all_features(captions):
                all_features = []
                batch_size = 32

                for i in range(0, len(captions), batch_size):
                    end_idx = min(i + batch_size, len(captions))
                    batch_captions = captions[i:end_idx]

                    inputs = self.clip_processor(text=batch_captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

                    with torch.no_grad():
                        features = self.clip_model.get_text_features(**inputs)
                        # 归一化
                        features = features / features.norm(dim=-1, keepdim=True)
                        all_features.append(features.cpu().numpy())

                return np.vstack(all_features)

            gen_features = get_all_features(generated_captions)
            gt_features = get_all_features(ground_truth_captions)

            # 计算相似度矩阵
            similarity_matrix = np.dot(gen_features, gt_features.T)

            r_at_1 = 0
            r_at_5 = 0
            r_at_10 = 0

            for i in range(len(generated_captions)):
                # 获取第i个生成caption与所有ground truth的相似度
                similarities = similarity_matrix[i]
                # 排序获取最相似的索引
                sorted_indices = np.argsort(similarities)[::-1]

                # 检查正确答案是否在top-k中
                if i in sorted_indices[:1]:
                    r_at_1 += 1
                if i in sorted_indices[:5]:
                    r_at_5 += 1
                if i in sorted_indices[:10]:
                    r_at_10 += 1

            # 计算比例
            total = len(generated_captions)
            r_at_1_score = r_at_1 / total
            r_at_5_score = r_at_5 / total
            r_at_10_score = r_at_10 / total

            results = {
                "r_at_1": r_at_1_score,
                "r_at_5": r_at_5_score,
                "r_at_10": r_at_10_score
            }

            print(f"✅ R-Precision: R@1={r_at_1_score:.4f}, R@5={r_at_5_score:.4f}, R@10={r_at_10_score:.4f}")
            return results

        except Exception as e:
            print(f"❌ R-Precision计算失败: {e}")
            return {"r_at_1": 0.0, "r_at_5": 0.0, "r_at_10": 0.0}


    def evaluate(self, generated_file: str, ground_truth_file: str) -> Dict[str, Any]:
        """执行完整评估"""
        print("="*60)
        print("开始Caption评估")
        print("="*60)

        # 加载数据
        generated_data = self.load_json_data(generated_file)
        ground_truth_data = self.load_json_data(ground_truth_file)

        if not generated_data or not ground_truth_data:
            print("❌ 数据加载失败")
            return {}

        # 匹配数据
        matched_pairs = self.match_data(generated_data, ground_truth_data)

        if not matched_pairs:
            print("❌ 没有匹配的数据")
            return {}

        # 提取caption列表
        image_ids = [pair[0] for pair in matched_pairs]
        generated_captions = [pair[1] for pair in matched_pairs]
        ground_truth_captions = [pair[2] for pair in matched_pairs]

        print(f"\n开始评估 {len(matched_pairs)} 对caption...")

        # 计算各项指标
        results = {}

        # CLIP Score
        results["clip_score"] = self.calculate_clip_score(generated_captions, ground_truth_captions)

        # FID Score
        results["fid_score"] = self.calculate_fid_score(generated_captions, ground_truth_captions)

        # 语义相似度
        results["semantic_similarity"] = self.calculate_semantic_similarity(generated_captions, ground_truth_captions)

        # R-Precision
        r_precision_results = self.calculate_r_precision(generated_captions, ground_truth_captions)
        results.update(r_precision_results)

        # 词汇重叠相似度
        results["lexical_similarity"] = self.calculate_simple_similarity(generated_captions, ground_truth_captions)

        # 计算加权综合评分
        results["overall_score"] = (
            results.get("clip_score", 0) * 0.25 +
            results.get("semantic_similarity", 0) * 0.25 +
            results.get("r_at_10", 0) * 0.2 +
            results.get("lexical_similarity", 0) * 0.15 +
            results.get("r_at_5", 0) * 0.1 +
            results.get("r_at_1", 0) * 0.05
        )

        # 添加元数据
        results["metadata"] = {
            "total_pairs": len(matched_pairs),
            "generated_file": generated_file,
            "ground_truth_file": ground_truth_file,
            "model_name": self.extract_model_name(generated_file)
        }

        return results

    def extract_model_name(self, file_path: str) -> str:
        """从文件路径提取模型名称"""
        filename = os.path.basename(file_path)
        if filename.startswith("image_"):
            return filename.replace("image_", "").replace(".json", "")
        return filename.replace(".json", "")

    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成评估报告"""
        if not results:
            return "评估失败，无结果数据。"

        model_name = results.get("metadata", {}).get("model_name", "Unknown")
        total_pairs = results.get("metadata", {}).get("total_pairs", 0)

        report = f"""
# Caption评估报告 - {model_name}

## 基本信息
- **模型名称**: {model_name}
- **评估样本数**: {total_pairs}
- **生成文件**: {results.get("metadata", {}).get("generated_file", "N/A")}
- **Ground Truth文件**: {results.get("metadata", {}).get("ground_truth_file", "N/A")}

## 评估指标

### 1. 核心指标
- **CLIP Score**: {results.get("clip_score", 0):.4f}
  - 衡量生成caption与ground truth在CLIP嵌入空间中的相似度
  - 范围: 0-1，越高越好

- **FID Score**: {results.get("fid_score", 0):.4f}
  - 基于CLIP特征的Fréchet Inception Distance
  - 范围: 0+，越低越好

- **语义相似度**: {results.get("semantic_similarity", 0):.4f}
  - 基于CLIP的语义相似度
  - 范围: 0-1，越高越好

### 2. R-Precision结果
- **R@1**: {results.get("r_at_1", 0):.2%}
  - Top-1检索准确率
- **R@5**: {results.get("r_at_5", 0):.2%}
  - Top-5检索准确率
- **R@10**: {results.get("r_at_10", 0):.2%}
  - Top-10检索准确率

### 3. 词汇匹配
- **词汇重叠相似度**: {results.get("lexical_similarity", 0):.4f}
  - 基于Jaccard相似度的词汇重叠程度
  - 范围: 0-1，越高越好

### 4. 综合评分
- **整体评分**: {results.get("overall_score", 0):.4f}
  - 综合考虑所有指标的加权平均分
  - 权重: CLIP Score (25%) + 语义相似度 (25%) + R@10 (20%) + 词汇重叠 (15%) + R@5 (10%) + R@1 (5%)
  - 范围: 0-1，越高越好

## 评估总结

该模型在caption生成任务上的表现：

1. **CLIP相似度**: {"优秀" if results.get("clip_score", 0) > 0.8 else "良好" if results.get("clip_score", 0) > 0.6 else "一般" if results.get("clip_score", 0) > 0.4 else "较差"}
2. **语义理解**: {"优秀" if results.get("semantic_similarity", 0) > 0.8 else "良好" if results.get("semantic_similarity", 0) > 0.6 else "一般" if results.get("semantic_similarity", 0) > 0.4 else "较差"}
3. **检索性能**: {"优秀" if results.get("r_at_10", 0) > 0.7 else "良好" if results.get("r_at_10", 0) > 0.5 else "一般" if results.get("r_at_10", 0) > 0.3 else "较差"}
4. **词汇匹配**: {"优秀" if results.get("lexical_similarity", 0) > 0.6 else "良好" if results.get("lexical_similarity", 0) > 0.4 else "一般" if results.get("lexical_similarity", 0) > 0.2 else "较差"}
5. **整体评分**: {"优秀" if results.get("overall_score", 0) > 0.8 else "良好" if results.get("overall_score", 0) > 0.6 else "一般" if results.get("overall_score", 0) > 0.4 else "较差"}

---
*报告生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Caption评估脚本')
    parser.add_argument('--generated', type=str, required=True,
                       help='生成的caption JSON文件路径')
    parser.add_argument('--ground_truth', type=str, default='./data/groundTruth.json',
                       help='Ground truth文件路径 (默认: ./data/groundTruth.json)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出报告文件路径 (可选)')

    args = parser.parse_args()

    # 使用参数指定的ground truth路径
    ground_truth_file = args.ground_truth

    # 检查文件是否存在
    if not os.path.exists(args.generated):
        print(f"❌ 生成文件不存在: {args.generated}")
        return

    if not os.path.exists(ground_truth_file):
        print(f"❌ Ground truth文件不存在: {ground_truth_file}")
        return

    # 创建评估器
    evaluator = CaptionEvaluator()

    # 执行评估
    results = evaluator.evaluate(args.generated, ground_truth_file)

    if not results:
        print("❌ 评估失败")
        return

    # 生成报告
    report = evaluator.generate_report(results)

    # 输出结果
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)
    print(report)

    # 保存详细结果到JSON
    model_name = evaluator.extract_model_name(args.generated)

    if args.output:
        output_json = args.output
        output_txt = args.output.replace('.json', '.txt')
    else:
        output_json = f"compare_{model_name}.json"
        output_txt = f"compare_{model_name}.txt"

    # 保存JSON结果
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存文本报告
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✅ 详细结果已保存到: {output_json}")
    print(f"✅ 评估报告已保存到: {output_txt}")

if __name__ == "__main__":
    main()
