#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用训练好的模型对图片进行分类并组织到不同文件夹
将images目录下的图片分类后保存到images3/0和images3/1文件夹
"""

import os
import shutil
from tqdm import tqdm
from fenleiqi import ImageQualityPredictor

# ==================== 全局配置 ====================
DEFAULT_GPU = 2                                    # 使用的GPU编号
MODEL_PATH = 'best_image_quality_model.pth'        # 模型路径
SOURCE_DIR = 'images'                              # 源图片目录
TARGET_DIR = 'images3'                             # 目标目录
BATCH_SIZE = 32                                    # 批处理大小
CONFIDENCE_THRESHOLD = 0.5                         # 置信度阈值
# ================================================

def get_all_images(directory):
    """获取目录下所有图片文件"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF')
    image_paths = []

    print(f"扫描目录: {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))

    print(f"找到 {len(image_paths)} 张图片")
    return image_paths

def create_target_directories(target_dir):
    """创建目标目录结构"""
    low_quality_dir = os.path.join(target_dir, '0')
    high_quality_dir = os.path.join(target_dir, '1')

    os.makedirs(low_quality_dir, exist_ok=True)
    os.makedirs(high_quality_dir, exist_ok=True)

    print(f"创建目标目录:")
    print(f"  低质量图片: {low_quality_dir}")
    print(f"  高质量图片: {high_quality_dir}")

    return low_quality_dir, high_quality_dir

def batch_predict_and_organize(predictor, image_paths, low_quality_dir, high_quality_dir, batch_size=32):
    """批量预测并组织图片"""

    # 统计变量
    total_images = len(image_paths)
    processed_count = 0
    low_quality_count = 0
    high_quality_count = 0
    error_count = 0

    print(f"\n开始处理 {total_images} 张图片...")
    print(f"批次大小: {batch_size}")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print("-" * 60)

    # 分批处理
    for i in tqdm(range(0, total_images, batch_size), desc="批次进度"):
        batch_paths = image_paths[i:i+batch_size]

        # 批量预测
        try:
            results = predictor.predict_batch(batch_paths)

            # 处理每个结果
            for result in results:
                if 'error' in result:
                    error_count += 1
                    print(f"预测失败: {os.path.basename(result['image_path'])}")
                    continue

                image_path = result['image_path']
                predicted_class = result['predicted_class']
                confidence = result['confidence']
                filename = os.path.basename(image_path)

                # 根据预测结果选择目标目录
                if predicted_class == 0:
                    target_path = os.path.join(low_quality_dir, filename)
                    low_quality_count += 1
                else:
                    target_path = os.path.join(high_quality_dir, filename)
                    high_quality_count += 1

                # 复制文件
                try:
                    shutil.copy2(image_path, target_path)
                    processed_count += 1
                except Exception as e:
                    print(f"复制失败 {filename}: {e}")
                    error_count += 1

                # 每处理100张图片显示一次进度
                if processed_count % 100 == 0:
                    print(f"已处理: {processed_count}/{total_images}, "
                          f"低质量: {low_quality_count}, 高质量: {high_quality_count}")

        except Exception as e:
            print(f"批次处理失败: {e}")
            error_count += len(batch_paths)

    return {
        'total': total_images,
        'processed': processed_count,
        'low_quality': low_quality_count,
        'high_quality': high_quality_count,
        'errors': error_count
    }

def main():
    """主函数"""
    print("=" * 60)
    print("图像质量分类器 - 批量分类和组织工具")
    print("=" * 60)

    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 {MODEL_PATH}")
        return

    # 检查源目录
    if not os.path.exists(SOURCE_DIR):
        print(f"错误: 源目录不存在 {SOURCE_DIR}")
        return

    # 获取所有图片
    image_paths = get_all_images(SOURCE_DIR)
    if not image_paths:
        print("错误: 源目录中没有找到图片文件")
        return

    # 创建目标目录
    low_quality_dir, high_quality_dir = create_target_directories(TARGET_DIR)

    # 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    try:
        predictor = ImageQualityPredictor(MODEL_PATH, device=f'cuda:{DEFAULT_GPU}')
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 批量预测和组织
    stats = batch_predict_and_organize(
        predictor, image_paths, low_quality_dir, high_quality_dir, BATCH_SIZE
    )

    # 输出最终统计
    print("\n" + "=" * 60)
    print("分类完成！最终统计:")
    print("=" * 60)
    print(f"总图片数: {stats['total']}")
    print(f"成功处理: {stats['processed']}")
    print(f"处理失败: {stats['errors']}")
    print(f"低质量图片: {stats['low_quality']} (保存至 {TARGET_DIR}/0/)")
    print(f"高质量图片: {stats['high_quality']} (保存至 {TARGET_DIR}/1/)")

    # 计算比例
    if stats['processed'] > 0:
        low_ratio = stats['low_quality'] / stats['processed'] * 100
        high_ratio = stats['high_quality'] / stats['processed'] * 100
        print(f"\n质量分布:")
        print(f"低质量比例: {low_ratio:.1f}%")
        print(f"高质量比例: {high_ratio:.1f}%")

    # 验证结果
    actual_low = len([f for f in os.listdir(low_quality_dir) if os.path.isfile(os.path.join(low_quality_dir, f))])
    actual_high = len([f for f in os.listdir(high_quality_dir) if os.path.isfile(os.path.join(high_quality_dir, f))])

    print(f"\n验证结果:")
    print(f"{TARGET_DIR}/0/ 实际文件数: {actual_low}")
    print(f"{TARGET_DIR}/1/ 实际文件数: {actual_high}")
    print(f"总计: {actual_low + actual_high}")

    print("\n分类任务完成！")

if __name__ == '__main__':
    main()
