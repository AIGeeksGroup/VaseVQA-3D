# VaseVQA-3D Training Module

This directory contains the training-related code for the VaseVQA-3D project, mainly divided into two sub-modules: **Data Filtering (Filter)** and **Model Training (Model)**.

## 📁 目录结构

```
Train/
├── filter/                    # 数据过滤模块
│   ├── classifier.py         # 图像质量分类器
│   ├── clipfilter1.py        # CLIP过滤器（质量过滤）
│   ├── clipfilter1.sh        # CLIP过滤器启动脚本
│   ├── clipfilter2.py        # CLIP过滤器（视角选择）
│   └── model.py              # ResNet50分类模型
├── model/                     # 模型训练模块
│   ├── sft.sh                # 监督微调脚本
│   ├── grpo.sh               # GRPO强化学习训练脚本
│   ├── merge.sh              # LoRA权重合并脚本
│   ├── hf2megatron.sh        # HuggingFace转Megatron脚本
│   └── requirements.txt      # Python依赖包
└── README.md                  # 本文档
```

---

## 🔍 数据过滤模块 (Filter)

数据过滤模块用于从原始数据集中筛选高质量图像，包括基于ResNet50的质量分类和基于CLIP的视角选择。

### 1. ResNet50 图像质量分类器

#### 文件说明
- **`model.py`**: ResNet50二分类模型定义和训练代码
- **`classifier.py`**: 使用训练好的模型进行批量分类

#### 功能特点
- 使用预训练的 ResNet50 作为骨干网络
- 二分类任务：0（低质量）/ 1（高质量）
- 支持批量处理和GPU加速
- 自动数据增强和验证

#### 使用方法

**训练模型：**
```bash
cd filter
python model.py
```

配置参数（在 `model.py` 中修改）：
```python
DEFAULT_GPU = 2          # 使用的GPU编号
BATCH_SIZE = 256         # 批次大小
NUM_EPOCHS = 200         # 训练轮数
DATA_DIR = 'images2'     # 数据目录
```

数据目录结构：
```
images2/
├── 0/                   # 低质量图片
│   ├── image1.jpg
│   └── ...
└── 1/                   # 高质量图片
    ├── image2.jpg
    └── ...
```

**使用分类器：**
```bash
python classifier.py
```

配置参数（在 `classifier.py` 中修改）：
```python
DEFAULT_GPU = 2                                    # GPU编号
MODEL_PATH = 'best_image_quality_model.pth'        # 模型路径
SOURCE_DIR = 'images'                              # 源图片目录
TARGET_DIR = 'images3'                             # 目标目录
BATCH_SIZE = 32                                    # 批处理大小
```

输出结果：
```
images3/
├── 0/                   # 分类为低质量的图片
└── 1/                   # 分类为高质量的图片
```

---

### 2. CLIP 图像质量过滤器

#### 文件说明
- **`clipfilter1.py`**: 基于CLIP的图像质量过滤
- **`clipfilter1.sh`**: 交互式启动脚本
- **`clipfilter2.py`**: 基于CLIP的最佳视角选择

#### clipfilter1 - 质量过滤

**功能特点：**
- 使用CLIP模型评估图像质量
- 支持质量分布分析
- 自动阈值推荐
- 批量处理和进度跟踪

**使用方法：**

方式一：使用交互式脚本（推荐）
```bash
./clipfilter1.sh
```

脚本提供以下选项：
1. 🔍 仅分析数据集质量分布
2. 🎯 自动过滤（先分析后过滤）
3. ⚙️ 自定义阈值过滤
4. 💻 强制使用CPU模式
5. 🎮 使用GPU 1
6. 📊 快速分析（100个样本）

方式二：直接运行Python脚本
```bash
# 分析质量分布
python clipfilter1.py --analyze_only --input_dir ./images3/1

# 执行过滤
python clipfilter1.py --input_dir ./images3/1 \
                      --output_dir ./filtered_vases \
                      --threshold 0.15 \
                      --model_cache_dir ./model
```

**参数说明：**
- `--input_dir`: 输入图像目录
- `--output_dir`: 输出目录
- `--threshold`: 质量阈值（可选，不指定则自动分析）
- `--analyze_only`: 仅分析质量分布
- `--sample_size`: 分析时的采样大小（默认500）
- `--device`: 指定设备（cuda:0, cuda:1, cpu等）
- `--model_cache_dir`: CLIP模型缓存目录
- `--auto_select_device`: 交互式选择设备

**输出结果：**
```
filtered_vases/
├── accepted/            # 通过质量检查的图片
├── rejected/            # 未通过质量检查的图片
└── filtering_report.json  # 详细过滤报告
```

#### clipfilter2 - 最佳视角选择

**功能特点：**
- 为每个花瓶选择CLIP得分最高的视角
- 自动识别多视角图片
- 生成详细的选择报告

**使用方法：**
```bash
python clipfilter2.py --input_dir ./filtered_vases/accepted \
                      --output_dir ./filtered_vases \
                      --device cuda:0
```

**参数说明：**
- `--input_dir`: 包含多视角图片的输入目录
- `--output_dir`: 输出目录
- `--device`: 指定设备

**输出结果：**
```
filtered_vases/
├── best_views/                    # 每个花瓶的最佳视角
├── best_views_report.json         # 详细报告
└── filtering_stats.json           # 统计信息
```

---

## 🚀 模型训练模块 (Model)

模型训练模块基于 MS-SWIFT 框架，支持监督微调（SFT）和强化学习（GRPO）训练。

### 环境配置

**1. 创建Conda环境：**
```bash
conda create -n swift_env python=3.10
conda activate swift_env
```

**2. 安装依赖：**
```bash
cd model
pip install -r requirements.txt
```

**主要依赖：**
- PyTorch 2.0+
- MS-SWIFT 2.0+
- Transformers 4.35+
- 其他深度学习工具

---

### 1. 监督微调 (SFT)

**脚本：** `sft.sh`

**功能：** 使用标注数据对视觉语言模型进行监督微调

**使用方法：**
```bash
./sft.sh
```

**主要参数：**
```bash
CUDA_VISIBLE_DEVICES=0              # 使用的GPU
MAX_PIXELS=1003520                  # 最大像素数
--model './models/Qwen2.5-VL-7B-Instruct'  # 模型路径
--dataset './data/video_training_dataset.json'  # 训练数据
--train_type lora                   # 训练类型（lora/full）
--num_train_epochs 2                # 训练轮数
--per_device_train_batch_size 1     # 批次大小
--learning_rate 1e-4                # 学习率
--lora_rank 8                       # LoRA秩
--output_dir output/7B              # 输出目录
```

**数据格式：**
```json
[
  {
    "images": ["path/to/image.jpg"],
    "caption": "描述文本",
    "conversations": [...]
  }
]
```

**输出：**
- 训练好的LoRA权重
- 训练日志和TensorBoard记录
- 定期保存的检查点

---

### 2. GRPO 强化学习训练

**脚本：** `grpo.sh`

**功能：** 使用GRPO算法进行强化学习训练，优化模型输出质量

**使用方法：**
```bash
./grpo.sh
```

**主要参数：**
```bash
--rlhf_type grpo                    # 强化学习类型
--model './output/checkpoint-merged'  # 基础模型
--external_plugins ./plugin/plugin.py  # 外部插件
--reward_funcs external_vase_acc    # 奖励函数
--dataset './data/grpo_video_dataset.json'  # 训练数据
--num_train_epochs 1                # 训练轮数
--learning_rate 1e-6                # 学习率
--num_generations 4                 # 每次生成数量
--temperature 1.0                   # 采样温度
--beta 0.001                        # KL散度系数
--system ./prompt.txt               # 系统提示
```

**奖励函数：**
需要在 `plugin/plugin.py` 中定义自定义奖励函数，例如：
```python
def external_vase_acc(responses, references):
    # 计算奖励分数
    return scores
```

---

### 3. LoRA 权重合并

**脚本：** `merge.sh`

**功能：** 将训练好的LoRA权重合并到基础模型

**使用方法：**
```bash
./merge.sh
```

**参数：**
```bash
--adapters ./output/checkpoint-600  # LoRA权重路径
--merge_lora true                   # 启用合并
```

**输出：**
- 合并后的完整模型
- 可直接用于推理或进一步训练

---

### 4. HuggingFace 转 Megatron

**脚本：** `hf2megatron.sh`

**功能：** 将HuggingFace格式模型转换为Megatron格式

**使用方法：**
```bash
./hf2megatron.sh
```

**参数：**
```bash
--model './models/Qwen2.5-VL-3B-Instruct'  # 输入模型
--to_mcore true                     # 转换为Megatron格式
--torch_dtype bfloat16              # 数据类型
--output_dir output/Qwen2.5-VL-3B-Instruct-mcore  # 输出目录
--test_convert_precision true       # 测试转换精度
```

---

## 📊 完整训练流程

### 数据准备和过滤

```bash
# 1. 使用ResNet50分类器过滤低质量图片
cd filter
python classifier.py

# 2. 使用CLIP过滤器进一步筛选
./clipfilter1.sh
# 选择选项2：自动过滤

# 3. 选择最佳视角
python clipfilter2.py --input_dir ./filtered_vases/accepted \
                      --output_dir ./filtered_vases
```

### 模型训练

```bash
cd ../model

# 1. 监督微调
./sft.sh

# 2. 合并LoRA权重
./merge.sh

# 3. GRPO强化学习（可选）
./grpo.sh

# 4. 再次合并权重
./merge.sh
```

---

## 🔧 配置说明

### GPU 配置

所有脚本都支持通过环境变量配置GPU：

```bash
# 单卡
export CUDA_VISIBLE_DEVICES=0

# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
```

### 代理配置

如需使用代理下载模型：

```bash
export http_proxy=http://your_proxy:port
export https_proxy=http://your_proxy:port
```

### 模型缓存

设置模型缓存目录：

```bash
export HF_HOME=./cache
export TRANSFORMERS_CACHE=./cache
```

---

## 📝 注意事项

1. **显存要求**
   - ResNet50训练：≥ 8GB
   - SFT训练（7B模型）：≥ 24GB
   - GRPO训练：≥ 32GB

2. **数据格式**
   - 图片格式：PNG, JPG, JPEG
   - 训练数据：JSON格式
   - 确保数据路径正确

3. **训练时间**
   - ResNet50训练：2-4小时（200 epochs）
   - SFT训练：根据数据量，通常数小时到数天
   - GRPO训练：比SFT慢2-3倍

4. **检查点管理**
   - 定期保存检查点
   - 设置 `save_total_limit` 限制检查点数量
   - 及时备份重要检查点

---

## 🐛 故障排除

### 常见问题

**1. CUDA Out of Memory**
```bash
# 减小批次大小
--per_device_train_batch_size 1

# 启用梯度检查点
--gradient_checkpointing true

# 使用更小的模型
```

**2. 模型加载失败**
```bash
# 检查模型路径
ls -la ./models/

# 检查权限
chmod -R 755 ./models/

# 重新下载模型
```

**3. 训练中断**
```bash
# 从检查点恢复
--resume_from_checkpoint ./output/checkpoint-XXX
```

---

## 📚 参考资源

- [MS-SWIFT 文档](https://github.com/modelscope/swift)
- [Qwen2.5-VL 模型](https://huggingface.co/Qwen)
- [CLIP 模型](https://github.com/openai/CLIP)
- [ResNet 论文](https://arxiv.org/abs/1512.03385)

---

**💡 提示**: 建议先在小数据集上测试完整流程，确认无误后再进行大规模训练。
