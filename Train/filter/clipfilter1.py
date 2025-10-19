#!/usr/bin/env python3

import os
import shutil
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import time
from transformers import CLIPProcessor, CLIPModel

class ServerVaseImageFilter:
    def __init__(self, device=None, model_cache_dir=None):
        """
        初始化CLIP模型用于图像质量过滤

        Args:
            device: 指定设备 (cuda/cpu)
            model_cache_dir: 自定义模型缓存目录
        """
        # 设备选择逻辑
        if device:
            if device.startswith("cuda:"):
                # 指定特定GPU
                gpu_id = device.split(":")[1]
                if torch.cuda.is_available() and int(gpu_id) < torch.cuda.device_count():
                    self.device = device
                else:
                    print(f"⚠️  GPU {gpu_id} 不可用，使用默认设备")
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device == "cuda":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"使用设备: {self.device}")
        if self.device.startswith("cuda"):
            print(f"GPU信息: {torch.cuda.get_device_name(self.device)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f} GB")

        # 设置自定义缓存目录
        if model_cache_dir:
            self.setup_custom_cache(model_cache_dir)

        # 加载CLIP模型
        print("正在加载CLIP模型...")
        try:
            # 本地模型路径
            local_model_path = "./openai/clip-vit-base-patch32"

            # 尝试的模型路径和名称（按优先级排序）
            model_sources = [
                {"path": local_model_path, "name": "本地模型", "local_only": True},
                {"path": "openai/clip-vit-base-patch32", "name": "HuggingFace Hub", "local_only": False},
                {"path": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "name": "LAION模型", "local_only": False},
            ]

            model_loaded = False
            for source in model_sources:
                try:
                    print(f"尝试加载模型: {source['name']} ({source['path']})")

                    # 根据是否为本地模型设置参数
                    if source['local_only']:
                        # 检查本地路径是否存在
                        if not os.path.exists(source['path']):
                            print(f"❌ 本地模型路径不存在: {source['path']}")
                            continue

                        self.model = CLIPModel.from_pretrained(
                            source['path'],
                            local_files_only=True,
                            trust_remote_code=True
                        ).to(self.device)
                        self.processor = CLIPProcessor.from_pretrained(
                            source['path'],
                            local_files_only=True,
                            trust_remote_code=True
                        )
                    else:
                        # 从HuggingFace Hub下载
                        cache_dir = model_cache_dir if model_cache_dir else None
                        self.model = CLIPModel.from_pretrained(
                            source['path'],
                            cache_dir=cache_dir,
                            local_files_only=False,
                            trust_remote_code=True
                        ).to(self.device)
                        self.processor = CLIPProcessor.from_pretrained(
                            source['path'],
                            cache_dir=cache_dir,
                            local_files_only=False,
                            trust_remote_code=True
                        )

                    print(f"✅ CLIP模型加载完成! 使用: {source['name']}")
                    model_loaded = True
                    break

                except Exception as e:
                    print(f"❌ {source['name']} 加载失败: {e}")
                    continue

            if not model_loaded:
                raise Exception("所有模型源都加载失败")

        except Exception as e:
            print(f"CLIP模型加载失败: {e}")
            raise

        # 定义用于质量评估的文本提示
        self.quality_prompts = {
            "high_quality": [
                "a complete intact vase viewed from the front",
                "a whole undamaged vase in frontal view",
                "a perfect ceramic vase facing forward",
                "an intact decorative vase front view",
                "a complete flower vase shown frontally",
                "a full unbroken vase from front angle",
                "an entire undamaged pottery vase facing camera",
                "a pristine whole vase in front perspective"
            ],
            "low_quality": [
                "broken vase fragments and pieces",
                "shattered ceramic vase debris",
                "cracked and damaged vase parts",
                "incomplete vase with missing pieces",
                "fragmented broken pottery",
                "ceramic vase shards and chips",
                "damaged vase with cracks and breaks",
                "partial vase pieces and fragments"
            ]
        }

        # 编码文本提示
        self.encode_text_prompts()

    def setup_custom_cache(self, cache_dir):
        """设置自定义缓存目录"""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        print(f"设置自定义缓存目录: {cache_dir}")

        # 设置环境变量
        os.environ['CLIP_CACHE_DIR'] = str(cache_path)
        os.environ['XDG_CACHE_HOME'] = str(cache_path.parent)

        # 创建clip子目录
        clip_cache_dir = cache_path / "clip"
        clip_cache_dir.mkdir(exist_ok=True)

        # 如果原缓存目录存在且有文件，尝试移动
        default_cache = Path.home() / ".cache" / "clip"
        if default_cache.exists():
            print(f"检测到原缓存目录: {default_cache}")
            try:
                # 移动现有文件到新位置
                for file in default_cache.glob("*"):
                    if file.is_file():
                        target = clip_cache_dir / file.name
                        if not target.exists():
                            shutil.move(str(file), str(target))
                            print(f"移动缓存文件: {file.name}")
            except Exception as e:
                print(f"移动缓存文件时出错: {e}")

        print(f"CLIP模型将缓存到: {clip_cache_dir}")

    def encode_text_prompts(self):
        """编码所有文本提示"""
        print("编码文本提示...")

        # 编码高质量提示
        high_quality_inputs = self.processor(text=self.quality_prompts["high_quality"], return_tensors="pt", padding=True)
        high_quality_inputs = {k: v.to(self.device) for k, v in high_quality_inputs.items()}

        with torch.no_grad():
            high_quality_outputs = self.model.get_text_features(**high_quality_inputs)
            self.high_quality_features = high_quality_outputs.mean(dim=0, keepdim=True)
            self.high_quality_features /= self.high_quality_features.norm(dim=-1, keepdim=True)

        # 编码低质量提示
        low_quality_inputs = self.processor(text=self.quality_prompts["low_quality"], return_tensors="pt", padding=True)
        low_quality_inputs = {k: v.to(self.device) for k, v in low_quality_inputs.items()}

        with torch.no_grad():
            low_quality_outputs = self.model.get_text_features(**low_quality_inputs)
            self.low_quality_features = low_quality_outputs.mean(dim=0, keepdim=True)
            self.low_quality_features /= self.low_quality_features.norm(dim=-1, keepdim=True)

        print("文本提示编码完成!")

    def calculate_quality_score(self, image_path):
        """
        计算图像的质量分数
        返回: (quality_score, high_quality_sim, low_quality_sim)
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

                # 计算与高质量和低质量提示的相似度
                high_quality_sim = torch.cosine_similarity(image_features, self.high_quality_features).item()
                low_quality_sim = torch.cosine_similarity(image_features, self.low_quality_features).item()

                # 质量分数 = 高质量相似度 - 低质量相似度
                quality_score = high_quality_sim - low_quality_sim

                return quality_score, high_quality_sim, low_quality_sim

        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            return -1.0, 0.0, 1.0  # 返回低质量分数

    def get_image_files(self, input_dir):
        """获取目录中的所有图像文件"""
        input_path = Path(input_dir)

        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.WEBP'}

        # 获取所有图像文件
        image_files = []
        for file_path in input_path.iterdir():
            if file_path.is_file() and file_path.suffix in image_extensions:
                image_files.append(file_path)

        return image_files

    def filter_images(self, input_dir, output_dir, threshold=0.1, save_rejected=True):
        """
        过滤图像数据集

        Args:
            input_dir: 输入图像目录
            output_dir: 输出目录
            threshold: 质量阈值，高于此值的图像被保留
            save_rejected: 是否保存被拒绝的图像
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # 检查输入目录是否存在
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        # 创建输出目录
        accepted_dir = output_path / "accepted"
        rejected_dir = output_path / "rejected"
        accepted_dir.mkdir(parents=True, exist_ok=True)
        if save_rejected:
            rejected_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图像文件
        image_files = self.get_image_files(input_dir)

        print(f"在目录 {input_dir} 中找到 {len(image_files)} 个图像文件")

        if len(image_files) == 0:
            print("警告: 没有找到任何图像文件!")
            return None

        # 存储结果
        results = []
        accepted_count = 0
        rejected_count = 0

        # 记录开始时间
        start_time = time.time()

        # 处理每个图像
        for i, image_file in enumerate(tqdm(image_files, desc="处理图像")):
            quality_score, high_sim, low_sim = self.calculate_quality_score(image_file)

            result = {
                "filename": image_file.name,
                "quality_score": quality_score,
                "high_quality_similarity": high_sim,
                "low_quality_similarity": low_sim,
                "accepted": quality_score >= threshold
            }
            results.append(result)

            # 复制文件到相应目录
            try:
                if quality_score >= threshold:
                    shutil.copy2(image_file, accepted_dir / image_file.name)
                    accepted_count += 1
                else:
                    if save_rejected:
                        shutil.copy2(image_file, rejected_dir / image_file.name)
                    rejected_count += 1
            except Exception as e:
                print(f"复制文件 {image_file.name} 时出错: {e}")

            # 每处理100个文件显示一次进度
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = (len(image_files) - i - 1) * avg_time
                print(f"已处理 {i+1}/{len(image_files)} 个文件, 预计剩余时间: {remaining/60:.1f}分钟")

        # 保存结果报告
        report = {
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "total_images": len(image_files),
            "accepted_images": accepted_count,
            "rejected_images": rejected_count,
            "acceptance_rate": accepted_count / len(image_files) if image_files else 0,
            "threshold": threshold,
            "processing_time_seconds": time.time() - start_time,
            "results": results
        }

        report_file = output_path / "filtering_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印统计信息
        print(f"\n" + "="*60)
        print(f"过滤完成!")
        print(f"总图像数: {len(image_files)}")
        print(f"接受的图像: {accepted_count} ({accepted_count/len(image_files)*100:.1f}%)")
        print(f"拒绝的图像: {rejected_count} ({rejected_count/len(image_files)*100:.1f}%)")
        print(f"质量阈值: {threshold}")
        print(f"处理时间: {(time.time() - start_time)/60:.1f}分钟")
        print(f"结果保存在: {output_path}")
        print(f"详细报告: {report_file}")
        print("="*60)

        return report

    def analyze_quality_distribution(self, input_dir, sample_size=None):
        """
        分析数据集的质量分布，帮助确定合适的阈值
        """
        input_path = Path(input_dir)

        # 检查输入目录是否存在
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        # 获取所有图像文件
        image_files = self.get_image_files(input_dir)

        print(f"在目录 {input_dir} 中找到 {len(image_files)} 个图像文件")

        if len(image_files) == 0:
            print("警告: 没有找到任何图像文件!")
            return None, None

        # 如果指定了样本大小，随机采样
        if sample_size and len(image_files) > sample_size:
            import random
            image_files = random.sample(image_files, sample_size)
            print(f"随机采样 {len(image_files)} 个文件进行分析")

        quality_scores = []

        # 记录开始时间
        start_time = time.time()

        for i, image_file in enumerate(tqdm(image_files, desc="分析质量")):
            quality_score, _, _ = self.calculate_quality_score(image_file)
            quality_scores.append(quality_score)

            # 每处理50个文件显示一次进度
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = (len(image_files) - i - 1) * avg_time
                print(f"已分析 {i+1}/{len(image_files)} 个文件, 预计剩余时间: {remaining/60:.1f}分钟")

        # 计算统计信息
        quality_scores = np.array(quality_scores)
        stats = {
            "mean": float(np.mean(quality_scores)),
            "std": float(np.std(quality_scores)),
            "min": float(np.min(quality_scores)),
            "max": float(np.max(quality_scores)),
            "median": float(np.median(quality_scores)),
            "percentile_10": float(np.percentile(quality_scores, 10)),
            "percentile_25": float(np.percentile(quality_scores, 25)),
            "percentile_50": float(np.percentile(quality_scores, 50)),
            "percentile_75": float(np.percentile(quality_scores, 75)),
            "percentile_90": float(np.percentile(quality_scores, 90))
        }

        print(f"\n" + "="*60)
        print(f"质量分数统计 (基于 {len(image_files)} 个样本):")
        print(f"平均值: {stats['mean']:.3f}")
        print(f"标准差: {stats['std']:.3f}")
        print(f"最小值: {stats['min']:.3f}")
        print(f"最大值: {stats['max']:.3f}")
        print(f"中位数: {stats['median']:.3f}")
        print(f"10%分位数: {stats['percentile_10']:.3f}")
        print(f"25%分位数: {stats['percentile_25']:.3f}")
        print(f"75%分位数: {stats['percentile_75']:.3f}")
        print(f"90%分位数: {stats['percentile_90']:.3f}")

        # 建议不同的阈值策略
        print(f"\n阈值建议:")
        print(f"保守过滤 (保留90%): {stats['percentile_10']:.3f}")
        print(f"中等过滤 (保留75%): {stats['percentile_25']:.3f}")
        print(f"平衡过滤 (保留50%): {stats['percentile_50']:.3f}")
        print(f"激进过滤 (保留25%): {stats['percentile_75']:.3f}")
        print(f"严格过滤 (保留10%): {stats['percentile_90']:.3f}")
        print("="*60)

        return stats, quality_scores

def select_device():
    """交互式选择设备"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，将使用CPU")
        return "cpu"

    gpu_count = torch.cuda.device_count()
    print(f"\n🎮 检测到 {gpu_count} 个GPU:")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

    print(f"  CPU: 使用CPU模式")
    print("")

    while True:
        choice = input(f"请选择设备 (0-{gpu_count-1} 选择GPU, c 选择CPU): ").strip().lower()

        if choice == 'c':
            return "cpu"
        elif choice.isdigit():
            gpu_id = int(choice)
            if 0 <= gpu_id < gpu_count:
                return f"cuda:{gpu_id}"
            else:
                print(f"❌ 无效的GPU ID，请输入 0-{gpu_count-1}")
        else:
            print("❌ 无效输入，请输入数字或 'c'")

def main():
    # 默认路径
    default_input_dir = "./images3/1"
    default_output_dir = "./filtered_vases"

    parser = argparse.ArgumentParser(description="使用CLIP过滤花瓶图像数据集")
    parser.add_argument("--input_dir", type=str, default=default_input_dir,
                       help=f"输入图像目录路径 (默认: {default_input_dir})")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                       help=f"输出目录路径 (默认: {default_output_dir})")
    parser.add_argument("--threshold", type=float, default=None,
                       help="质量阈值 (如果不指定，会先分析数据集)")
    parser.add_argument("--analyze_only", action="store_true",
                       help="仅分析质量分布，不进行过滤")
    parser.add_argument("--sample_size", type=int, default=500,
                       help="分析时的采样大小 (默认: 500)")
    parser.add_argument("--device", type=str, default=None,
                       help="指定设备 (cuda:0, cuda:1, cpu 等)")
    parser.add_argument("--model_cache_dir", type=str,
                       default="./CLIP/model",
                       help="CLIP模型缓存目录")
    parser.add_argument("--auto_select_device", action="store_true",
                       help="交互式选择设备")

    args = parser.parse_args()

    print("="*60)
    print("CLIP 花瓶图像质量过滤器")
    print("="*60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")

    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 '{args.input_dir}' 不存在")
        print("请确认路径是否正确")
        return

    # 设备选择
    device = args.device
    if args.auto_select_device or device is None:
        device = select_device()

    # 初始化过滤器
    try:
        filter_tool = ServerVaseImageFilter(device=device, model_cache_dir=args.model_cache_dir)
    except Exception as e:
        print(f"初始化CLIP模型失败: {e}")
        print("请确保已安装必要的依赖包:")
        print("pip install torch torchvision transformers Pillow numpy tqdm")
        return

    # 如果只是分析模式
    if args.analyze_only:
        try:
            stats, scores = filter_tool.analyze_quality_distribution(
                args.input_dir, args.sample_size
            )
            if stats is None:
                return
        except Exception as e:
            print(f"分析过程中出错: {e}")
        return

    # 如果没有指定阈值，先分析数据集
    if args.threshold is None:
        print("未指定阈值，先分析数据集质量分布...")
        try:
            stats, scores = filter_tool.analyze_quality_distribution(
                args.input_dir, args.sample_size
            )
            if stats is None:
                return

            # 使用25%分位数作为默认阈值（保留75%的数据）
            threshold = stats['percentile_25']
            print(f"\n使用建议阈值: {threshold:.3f} (保留约75%的数据)")

            # 询问用户是否继续
            user_input = input("\n是否使用此阈值继续过滤? (y/n/自定义阈值): ").strip().lower()
            if user_input == 'n':
                print("过滤已取消")
                return
            elif user_input != 'y':
                try:
                    threshold = float(user_input)
                    print(f"使用自定义阈值: {threshold}")
                except ValueError:
                    print("无效输入，使用默认阈值")

        except Exception as e:
            print(f"分析过程中出错: {e}")
            return
    else:
        threshold = args.threshold

    # 执行过滤
    try:
        print(f"\n开始过滤，阈值: {threshold}")
        report = filter_tool.filter_images(
            args.input_dir,
            args.output_dir,
            threshold=threshold
        )

        if report:
            print(f"\n过滤完成! 详细报告已保存到: {args.output_dir}/filtering_report.json")

    except Exception as e:
        print(f"过滤过程中出错: {e}")

if __name__ == "__main__":
    main()
