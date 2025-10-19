
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多视角图片描述生成器 - Qwen2.5-VL版本
基于Qwen2.5-VL本地模型，处理6个面的图片生成综合caption
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any

from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def extract_image_name_from_folder(folder_path: str) -> str:
    """
    从文件夹路径提取图片名
    例如: /path/to/0A1F555B-F4F2-4074-9559-D96D3FA0C97C_2_ac001001_textured_2k.glb_shaded
    返回: 0A1F555B-F4F2-4074-9559-D96D3FA0C97C_2_ac001001.jpg
    """
    folder_name = os.path.basename(folder_path)

    # 使用正则表达式匹配UUID格式的文件名模式
    # 匹配格式: XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX_数字_ac001001
    pattern = r'^([A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}_\d+_ac001001)'

    match = re.match(pattern, folder_name)
    if match:
        base_name = match.group(1)
        return f"{base_name}.jpg"
    else:
        # 如果不匹配预期格式，尝试其他方式
        # 去掉常见的后缀
        suffixes_to_remove = ['_textured_4k.glb_shaded', '_textured_2k.glb_shaded', '_textured_2k', '.glb_shaded', '_shaded']

        for suffix in suffixes_to_remove:
            if folder_name.endswith(suffix):
                base_name = folder_name[:-len(suffix)]
                return f"{base_name}.jpg"

        # 如果都不匹配，直接使用文件夹名
        return f"{folder_name}.jpg"

def find_multiview_images(model_dir: str):
    """查找模型目录中的6个视角图片"""
    view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
    view_images = {}

    for view in view_names:
        image_path = os.path.join(model_dir, f"{view}.jpg")
        if os.path.exists(image_path):
            view_images[view] = image_path
        else:
            print(f"⚠️  缺少视角图片: {image_path}")

    return view_images

def generate_multiview_caption(model_dir: str, model, processor):
    """为单个模型的多视角图片生成综合描述"""

    print(f"\n=== 处理模型: {os.path.basename(model_dir)} ===")

    # 查找6个视角的图片
    view_images = find_multiview_images(model_dir)

    if len(view_images) == 0:
        print(f"❌ 未找到任何视角图片")
        return None

    print(f"✅ 找到 {len(view_images)} 个视角图片: {list(view_images.keys())}")

    # 构建消息内容
    content_parts = []

    # 添加所有图片
    for view, image_path in view_images.items():
        content_parts.append({
            "type": "image",
            "image": image_path
        })

    # 添加文字提示
    prompt_text = f"""
请分析这个古希腊花瓶模型的多视角图像。我提供了以下 {len(view_images)} 个视角的图片：
{', '.join(view_images.keys())}

请生成一个简洁准确的caption描述，格式类似于：
"Athenian black-figure lekythos, c. 525–475 BCE, depicting Herakles and the boar; Marathon, Attica."

要求：
1. 包含器型名称（如lekythos, amphora, hydria等）
2. 包含制作技法（如black-figure, red-figure）
3. 包含大致年代
4. 简要描述主要装饰内容
5. 如果能识别，包含可能的出土地点

请只返回caption描述，不要包含其他分析内容。
"""

    content_parts.append({
        "type": "text",
        "text": prompt_text
    })

    # 构建消息
    messages = [
        {
            "role": "user",
            "content": content_parts
        }
    ]

    try:
        print(f"\n正在调用 Qwen2.5-VL 进行多视角分析...")

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if output_text and len(output_text) > 0:
            caption = output_text[0].strip()
            print(f"✅ 生成描述成功 ({len(caption)} 字符)")
            return caption
        else:
            print(f"❌ 未获得模型响应")
            return None

    except Exception as e:
        print(f"❌ 模型调用失败: {e}")
        return None

def batch_generate_captions(input_dir: str, output_dir: str, model_path: str = './models/Qwen2.5-VL-3B-Instruct'):
    """批量处理多个模型的多视角图片"""

    print(f"多视角图片描述生成器 - Qwen2.5-VL版本")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 初始化模型
    print("正在加载 Qwen2.5-VL 模型...")
    print(f"模型路径: {model_path}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("✅ 模型加载完成")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有模型目录
    model_dirs = [d for d in os.listdir(input_dir)
                  if os.path.isdir(os.path.join(input_dir, d))]

    if not model_dirs:
        print(f"❌ 在 {input_dir} 中未找到模型目录")
        return

    print(f"找到 {len(model_dirs)} 个模型目录")

    # 所有结果保存到一个列表中
    all_results = []
    success_count = 0

    # 生成输出文件名：image_qwen2.5.json
    output_filename = "image_qwen2.5-vl-3B-rl.json"
    output_file = os.path.join(output_dir, output_filename)

    for i, model_dir_name in enumerate(model_dirs):
        model_dir_path = os.path.join(input_dir, model_dir_name)

        print(f"\n=== 处理 {i+1}/{len(model_dirs)}: {model_dir_name} ===")

        # 从文件夹名提取图片名
        image_name = extract_image_name_from_folder(model_dir_path)
        print(f"提取的图片名: {image_name}")

        try:
            # 生成描述
            caption = generate_multiview_caption(model_dir_path, model, processor)

            if caption:
                # 按照用户要求的格式保存结果
                result_data = {
                    "images": [image_name],
                    "caption": caption,
                    "data_by": "qwen2.5"
                }

                all_results.append(result_data)
                success_count += 1

                print(f"✅ 成功生成描述")
                print(f"图片名: {image_name}")
                print(f"描述: {caption}")

            else:
                print(f"❌ 描述生成失败: {model_dir_name}")

        except KeyboardInterrupt:
            print(f"\n⚠️  用户中断，正在保存已处理的结果...")
            break
        except Exception as e:
            print(f"❌ 处理 {model_dir_name} 时出错: {e}")

    # 保存所有结果到一个JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 保存处理报告
    report = {
        'total_models': len(model_dirs),
        'processed': i + 1,
        'successful': success_count,
        'failed': (i + 1) - success_count,
        'success_rate': f"{success_count/(i+1)*100:.1f}%",
        'model_used': "qwen2.5",
        'output_file': output_file,
        'total_results': len(all_results)
    }

    report_file = os.path.join(output_dir, "qwen2.5_caption_generation_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n" + "="*60)
    print(f"批量描述生成完成！")
    print(f"总模型数: {len(model_dirs)}")
    print(f"已处理: {i + 1}")
    print(f"成功: {success_count}")
    print(f"失败: {(i + 1) - success_count}")
    print(f"成功率: {success_count/(i+1)*100:.1f}%")
    print(f"结果保存到: {output_file}")
    print(f"总共生成了 {len(all_results)} 条描述")
    print(f"详细报告: {report_file}")
    print("="*60)

def main():
    """主函数"""

    parser = argparse.ArgumentParser(description='多视角图片描述生成器 - Qwen2.5-VL版本')
    parser.add_argument('--input_dir', type=str,
                       default='./data/multiview_images',
                       help='包含多视角图片的输入目录')
    parser.add_argument('--output_dir', type=str,
                       default='./data/captions',
                       help='描述输出目录')
    parser.add_argument('--model_path', type=str,
                       default='./models/Qwen2.5-VL-3B-Instruct',
                       help='Qwen2.5-VL模型路径')
    parser.add_argument('--single_model', type=str, default=None,
                       help='处理单个模型目录')

    args = parser.parse_args()

    if args.single_model:
        # 处理单个模型
        if not os.path.exists(args.single_model):
            print(f"❌ 模型目录不存在: {args.single_model}")
            return

        os.makedirs(args.output_dir, exist_ok=True)

        # 初始化模型
        print("正在加载 Qwen2.5-VL 模型...")
        model_path = args.model_path

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
        print("✅ 模型加载完成")

        caption = generate_multiview_caption(args.single_model, model, processor)

        if caption:
            model_dir_name = os.path.basename(args.single_model)
            image_name = extract_image_name_from_folder(args.single_model)

            # 生成输出文件名
            output_filename = "image_qwen2.5-vl-3B-sft.json"
            output_file = os.path.join(args.output_dir, output_filename)

            # 按照用户要求的格式保存结果
            result_data = [{
                "images": [image_name],
                "caption": caption,
                "data_by": "qwen2.5"
            }]

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            print(f"\n✅ 单模型处理完成: {output_file}")
            print(f"图片名: {image_name}")
            print(f"描述: {caption}")
        else:
            print(f"❌ 单模型处理失败")
    else:
        # 批量处理
        batch_generate_captions(args.input_dir, args.output_dir, args.model_path)

if __name__ == "__main__":
    main()
