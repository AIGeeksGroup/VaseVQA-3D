#!/usr/bin/env python3
"""
基于CLIP得分过滤花瓶图像的最佳视角
每个花瓶只保留CLIP得分最高的一个视角
"""

import os
import shutil
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time
import re
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel

class VaseViewFilter:
    def __init__(self, device=None):
        """
        初始化CLIP模型用于图像质量评估
        """
        # 设备选择
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"使用设备: {self.device}")
        if self.device.startswith("cuda"):
            print(f"GPU信息: {torch.cuda.get_device_name(self.device)}")

        # 加载CLIP模型
        print("正在加载CLIP模型...")
        try:
            # 本地模型路径
            local_model_path = "./openai/clip-vit-base-patch32"

            if os.path.exists(local_model_path):
                print(f"使用本地模型: {local_model_path}")
                self.model = CLIPModel.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                    trust_remote_code=True
                ).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
            else:
                print("使用在线模型: openai/clip-vit-base-patch32")
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            print("✅ CLIP模型加载完成!")

        except Exception as e:
            print(f"❌ CLIP模型加载失败: {e}")
            raise

        # 定义用于花瓶质量评估的文本提示
        self.quality_prompts = [
            "a complete intact vase viewed from the front",
            "a whole undamaged vase in frontal view",
            "a perfect ceramic vase facing forward",
            "an intact decorative vase front view",
            "a complete flower vase shown frontally",
            "a full unbroken vase from front angle",
            "an entire undamaged pottery vase facing camera",
            "a pristine whole vase in front perspective"
        ]

        # 编码文本提示
        self.encode_text_prompts()

    def encode_text_prompts(self):
        """编码文本提示"""
        print("编码文本提示...")
        text_inputs = self.processor(text=self.quality_prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            # 取平均值作为质量特征
            self.quality_features = text_features.mean(dim=0, keepdim=True)
            self.quality_features /= self.quality_features.norm(dim=-1, keepdim=True)

        print("文本提示编码完成!")

    def calculate_clip_score(self, image_path):
        """
        计算图像的CLIP质量得分
        """
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            image_inputs = self.processor(images=image, return_tensors="pt")
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

            # 编码图像
            with torch.no_grad():
                image_features = self.model.get_image_features(**image_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # 计算与质量提示的相似度
                similarity = torch.cosine_similarity(image_features, self.quality_features).item()

                return similarity

        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            return -1.0  # 返回低分数

    def parse_filename(self, filename):
        """
        解析文件名，提取花瓶ID和视角编号
        格式: UUID_视角编号_ac001001.jpg
        """
        # 使用正则表达式解析文件名
        pattern = r'^([A-F0-9-]+)_(\d+)_ac001001\.jpg$'
        match = re.match(pattern, filename, re.IGNORECASE)

        if match:
            vase_id = match.group(1)
            view_number = int(match.group(2))
            return vase_id, view_number
        else:
            print(f"警告: 无法解析文件名 {filename}")
            return None, None

    def group_images_by_vase(self, input_dir):
        """
        按花瓶ID分组图像文件
        """
        input_path = Path(input_dir)

        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp',
                          '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.WEBP'}

        # 按花瓶ID分组
        vase_groups = defaultdict(list)

        for file_path in input_path.iterdir():
            if file_path.is_file() and file_path.suffix in image_extensions:
                vase_id, view_number = self.parse_filename(file_path.name)

                if vase_id and view_number is not None:
                    vase_groups[vase_id].append({
                        'file_path': file_path,
                        'view_number': view_number,
                        'filename': file_path.name
                    })

        print(f"找到 {len(vase_groups)} 个不同的花瓶")

        # 统计视角数量分布
        view_counts = defaultdict(int)
        for vase_id, views in vase_groups.items():
            view_counts[len(views)] += 1

        print("视角数量分布:")
        for count, num_vases in sorted(view_counts.items()):
            print(f"  {count} 个视角: {num_vases} 个花瓶")

        return vase_groups, view_counts

    def filter_best_views(self, input_dir, output_dir):
        """
        为每个花瓶选择CLIP得分最高的视角
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # 检查输入目录
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        # 创建输出目录
        best_views_dir = output_path / "best_views"
        best_views_dir.mkdir(parents=True, exist_ok=True)

        # 按花瓶分组
        vase_groups, view_counts = self.group_images_by_vase(input_dir)

        if not vase_groups:
            print("没有找到任何有效的花瓶图像文件!")
            return None

        # 存储结果
        results = []
        total_original_images = sum(len(views) for views in vase_groups.values())
        processed_vases = 0

        print(f"开始处理 {len(vase_groups)} 个花瓶的 {total_original_images} 张图像...")

        start_time = time.time()

        # 处理每个花瓶
        for vase_id, views in tqdm(vase_groups.items(), desc="处理花瓶"):
            if len(views) == 1:
                # 只有一个视角，直接复制
                view = views[0]
                best_view = {
                    'vase_id': vase_id,
                    'selected_view': view['view_number'],
                    'selected_filename': view['filename'],
                    'clip_score': self.calculate_clip_score(view['file_path']),
                    'total_views': 1,
                    'reason': 'only_one_view'
                }

                # 复制文件
                try:
                    shutil.copy2(view['file_path'], best_views_dir / view['filename'])
                except Exception as e:
                    print(f"复制文件 {view['filename']} 时出错: {e}")
                    continue

            else:
                # 多个视角，计算CLIP得分并选择最佳的
                view_scores = []

                for view in views:
                    clip_score = self.calculate_clip_score(view['file_path'])
                    view_scores.append({
                        'view': view,
                        'clip_score': clip_score
                    })

                # 选择得分最高的视角
                best_view_data = max(view_scores, key=lambda x: x['clip_score'])
                best_view_info = best_view_data['view']

                best_view = {
                    'vase_id': vase_id,
                    'selected_view': best_view_info['view_number'],
                    'selected_filename': best_view_info['filename'],
                    'clip_score': best_view_data['clip_score'],
                    'total_views': len(views),
                    'reason': 'best_clip_score',
                    'all_view_scores': [
                        {
                            'view_number': vs['view']['view_number'],
                            'filename': vs['view']['filename'],
                            'clip_score': vs['clip_score']
                        }
                        for vs in view_scores
                    ]
                }

                # 复制最佳视角的文件
                try:
                    shutil.copy2(best_view_info['file_path'], best_views_dir / best_view_info['filename'])
                except Exception as e:
                    print(f"复制文件 {best_view_info['filename']} 时出错: {e}")
                    continue

            results.append(best_view)
            processed_vases += 1

            # 每处理100个花瓶显示一次进度
            if processed_vases % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / processed_vases
                remaining = (len(vase_groups) - processed_vases) * avg_time
                print(f"已处理 {processed_vases}/{len(vase_groups)} 个花瓶, 预计剩余时间: {remaining/60:.1f}分钟")

        # 生成报告
        processing_time = time.time() - start_time

        report = {
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "processing_time_seconds": processing_time,
            "total_vases": len(vase_groups),
            "total_original_images": total_original_images,
            "total_selected_images": len(results),
            "reduction_rate": 1 - (len(results) / total_original_images) if total_original_images > 0 else 0,
            "average_clip_score": np.mean([r['clip_score'] for r in results]),
            "results": results
        }

        # 保存详细报告
        report_file = output_path / "best_views_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 保存简化的统计报告
        stats_report = {
            "summary": {
                "total_vases": len(vase_groups),
                "original_images": total_original_images,
                "selected_images": len(results),
                "reduction_rate": f"{report['reduction_rate']*100:.1f}%",
                "average_clip_score": f"{report['average_clip_score']:.4f}",
                "processing_time_minutes": f"{processing_time/60:.1f}"
            },
            "view_distribution": {
                str(count): num_vases
                for count, num_vases in sorted(view_counts.items())
            },
            "clip_score_stats": {
                "min": float(np.min([r['clip_score'] for r in results])),
                "max": float(np.max([r['clip_score'] for r in results])),
                "mean": float(np.mean([r['clip_score'] for r in results])),
                "std": float(np.std([r['clip_score'] for r in results]))
            }
        }

        stats_file = output_path / "filtering_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_report, f, indent=2, ensure_ascii=False)

        # 打印统计信息
        print(f"\n" + "="*60)
        print(f"过滤完成!")
        print(f"处理的花瓶数量: {len(vase_groups)}")
        print(f"原始图像数量: {total_original_images}")
        print(f"选择的图像数量: {len(results)}")
        print(f"数据减少率: {report['reduction_rate']*100:.1f}%")
        print(f"平均CLIP得分: {report['average_clip_score']:.4f}")
        print(f"处理时间: {processing_time/60:.1f}分钟")
        print(f"最佳视角保存在: {best_views_dir}")
        print(f"详细报告: {report_file}")
        print(f"统计报告: {stats_file}")
        print("="*60)

        return report

def main():
    import argparse

    # 默认路径
    default_input_dir = "./filtered_vases/accepted"
    default_output_dir = "./filtered_vases"

    parser = argparse.ArgumentParser(description="基于CLIP得分过滤花瓶图像的最佳视角")
    parser.add_argument("--input_dir", type=str, default=default_input_dir,
                       help=f"输入图像目录路径 (默认: {default_input_dir})")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                       help=f"输出目录路径 (默认: {default_output_dir})")
    parser.add_argument("--device", type=str, default=None,
                       help="指定设备 (cuda:0, cuda:1, cpu 等)")

    args = parser.parse_args()

    print("="*60)
    print("基于CLIP得分的花瓶最佳视角过滤器")
    print("="*60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")

    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 '{args.input_dir}' 不存在")
        return

    # 设备选择
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # 初始化过滤器
        filter_tool = VaseViewFilter(device=device)

        # 执行过滤
        report = filter_tool.filter_best_views(args.input_dir, args.output_dir)

        if report:
            print(f"\n✅ 过滤完成! 最佳视角已保存到: {args.output_dir}/best_views/")

    except Exception as e:
        print(f"❌ 过滤过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
