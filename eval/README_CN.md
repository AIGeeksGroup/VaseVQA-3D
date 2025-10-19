# VaseVQA-3D 评估模块

本目录包含 VaseVQA-3D 项目的评估相关代码，主要用于生成多视角图像的描述（Caption）并评估生成质量。

## 📁 目录结构

```
eval/
├── qwen.py                # Qwen2.5-VL多视角描述生成器
├── internvl.py            # InternVL多视角描述生成器
├── compare.py             # Caption评估脚本
├── compare.sh             # 批量评估脚本
└── README.md              # 本文档
```

---

## 🎯 功能概述

评估模块包含两个主要功能：

1. **Caption 生成**：使用视觉语言模型（Qwen2.5-VL 或 InternVL）为多视角图像生成描述
2. **Caption 评估**：使用多种指标评估生成的描述质量

---

## 📝 Caption 生成

### 1. Qwen2.5-VL 描述生成器

**文件：** `qwen.py`

**功能：** 基于 Qwen2.5-VL 模型，分析多视角图像并生成综合描述

#### 使用方法

**批量处理：**
```bash
python qwen.py --input_dir ./data/multiview_images \
               --output_dir ./data/captions \
               --model_path ./models/Qwen2.5-VL-3B-Instruct
```

**处理单个模型：**
```bash
python qwen.py --single_model ./data/multiview_images/model1 \
               --output_dir ./data/captions \
               --model_path ./models/Qwen2.5-VL-7B-Instruct
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` | 包含多视角图片的输入目录 | `./data/multiview_images` |
| `--output_dir` | 描述输出目录 | `./data/captions` |
| `--model_path` | Qwen2.5-VL模型路径 | `./models/Qwen2.5-VL-3B-Instruct` |
| `--single_model` | 处理单个模型目录（可选） | `None` |

#### 输入数据格式

多视角图片目录结构：
```
multiview_images/
├── UUID_1_ac001001_textured_4k.glb_shaded/
│   ├── front.jpg
│   ├── back.jpg
│   ├── left.jpg
│   ├── right.jpg
│   ├── top.jpg
│   └── bottom.jpg
├── UUID_2_ac001001_textured_4k.glb_shaded/
│   └── ...
└── ...
```

#### 输出格式

生成的 JSON 文件格式：
```json
[
  {
    "images": ["UUID_1_ac001001.jpg"],
    "caption": "Athenian black-figure lekythos, c. 525–475 BCE, depicting Herakles and the boar; Marathon, Attica.",
    "data_by": "qwen2.5"
  },
  {
    "images": ["UUID_2_ac001001.jpg"],
    "caption": "...",
    "data_by": "qwen2.5"
  }
]
```

输出文件：
- `image_qwen2.5-vl-3B-sft.json` - 生成的描述
- `qwen2.5_caption_generation_report.json` - 处理报告

#### 提示词模板

脚本使用以下提示词引导模型生成描述：

```
请分析这个古希腊花瓶模型的多视角图像。我提供了以下 6 个视角的图片：
front, back, left, right, top, bottom

请生成一个简洁准确的caption描述，格式类似于：
"Athenian black-figure lekythos, c. 525–475 BCE, depicting Herakles and the boar; Marathon, Attica."

要求：
1. 包含器型名称（如lekythos, amphora, hydria等）
2. 包含制作技法（如black-figure, red-figure）
3. 包含大致年代
4. 简要描述主要装饰内容
5. 如果能识别，包含可能的出土地点
```

---

### 2. InternVL 描述生成器

**文件：** `internvl.py`

**功能：** 基于 InternVL 模型，分析多视角图像并生成综合描述

#### 使用方法

**批量处理：**
```bash
python internvl.py --input_dir ./data/multiview_images \
                   --output_dir ./data/captions \
                   --model_path ./models/OpenGVLab/InternVL3_5-4B
```

**处理单个模型：**
```bash
python internvl.py --single_model ./data/multiview_images/model1 \
                   --output_dir ./data/captions \
                   --model_path ./models/OpenGVLab/InternVL3_5-4B
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` | 包含多视角图片的输入目录 | `./data/multiview_images` |
| `--output_dir` | 描述输出目录 | `./data/captions` |
| `--model_path` | InternVL模型路径 | `./models/OpenGVLab/InternVL3_5-4B` |
| `--single_model` | 处理单个模型目录（可选） | `None` |

#### 输出格式

```json
[
  {
    "images": ["UUID_1_ac001001.jpg"],
    "caption": "...",
    "data_by": "internvl"
  }
]
```

输出文件：
- `image_internvl.json` - 生成的描述
- `internvl_caption_generation_report.json` - 处理报告

#### 多卡推理

InternVL 支持多卡并行推理：

```bash
# 设置可见GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# 修改脚本中的 tp 参数
# tp=1 表示单卡，tp=6 表示6卡并行
```

---

## 📊 Caption 评估

### 1. 单模型评估

**文件：** `compare.py`

**功能：** 评估生成的 caption 与 ground truth 的相似度，计算多个评估指标

#### 使用方法

```bash
python compare.py --generated ./data/captions/image_qwen2.5.json \
                  --ground_truth ./data/groundTruth.json \
                  --output results/qwen_eval.json
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--generated` | 生成的caption JSON文件路径 | **必需** |
| `--ground_truth` | Ground truth文件路径 | `./data/groundTruth.json` |
| `--output` | 输出报告文件路径（可选） | 自动生成 |

#### 评估指标

脚本计算以下评估指标：

##### 1. CLIP Score
- **说明**：衡量生成caption与ground truth在CLIP嵌入空间中的相似度
- **范围**：0-1，越高越好
- **权重**：25%

##### 2. FID Score
- **说明**：基于句子嵌入的Fréchet Inception Distance
- **范围**：0+，越低越好
- **用途**：评估生成分布与真实分布的差异

##### 3. 语义相似度
- **说明**：基于CLIP的语义相似度
- **范围**：0-1，越高越好
- **权重**：25%

##### 4. R-Precision
- **R@1**：Top-1检索准确率（权重5%）
- **R@5**：Top-5检索准确率（权重10%）
- **R@10**：Top-10检索准确率（权重20%）

##### 5. 词汇重叠相似度
- **说明**：基于Jaccard相似度的词汇重叠程度
- **范围**：0-1，越高越好
- **权重**：15%

##### 6. 综合评分
- **说明**：所有指标的加权平均分
- **范围**：0-1，越高越好
- **公式**：
  ```
  Overall Score = CLIP Score × 0.25
                + Semantic Similarity × 0.25
                + R@10 × 0.20
                + Lexical Similarity × 0.15
                + R@5 × 0.10
                + R@1 × 0.05
  ```

#### 输出结果

脚本生成两个文件：

**1. JSON 结果文件** (`compare_model_name.json`)
```json
{
  "clip_score": 0.8234,
  "fid_score": 12.5678,
  "semantic_similarity": 0.7891,
  "r_at_1": 0.6500,
  "r_at_5": 0.8200,
  "r_at_10": 0.9100,
  "lexical_similarity": 0.4567,
  "overall_score": 0.7823,
  "metadata": {
    "total_pairs": 100,
    "generated_file": "...",
    "ground_truth_file": "...",
    "model_name": "qwen2.5"
  }
}
```

**2. 文本报告** (`compare_model_name.txt`)

包含详细的评估报告，格式如下：

```markdown
# Caption评估报告 - qwen2.5

## 基本信息
- 模型名称: qwen2.5
- 评估样本数: 100
- 生成文件: ./data/captions/image_qwen2.5.json
- Ground Truth文件: ./data/groundTruth.json

## 评估指标

### 1. 核心指标
- CLIP Score: 0.8234
- FID Score: 12.5678
- 语义相似度: 0.7891

### 2. R-Precision结果
- R@1: 65.00%
- R@5: 82.00%
- R@10: 91.00%

### 3. 词汇匹配
- 词汇重叠相似度: 0.4567

### 4. 综合评分
- 整体评分: 0.7823

## 评估总结
该模型在caption生成任务上的表现：
1. CLIP相似度: 优秀
2. 语义理解: 良好
3. 检索性能: 优秀
4. 词汇匹配: 一般
5. 整体评分: 良好
```

---

### 2. 批量评估

**文件：** `compare.sh`

**功能：** 批量评估文件夹下的所有模型结果文件

#### 使用方法

```bash
# 修改脚本中的路径配置
CAPTION_DIR="./data/captions"
GROUND_TRUTH="./data/groundTruth.json"

# 运行批量评估
./compare.sh
```

#### 处理流程

1. 扫描 `CAPTION_DIR` 目录下所有 `image_*.json` 文件
2. 跳过 `groundTruth.json` 文件
3. 对每个文件调用 `compare.py` 进行评估
4. 生成各模型的详细报告
5. 汇总所有结果到 `batch_evaluation_summary.json`
6. 生成模型对比表格

#### 输出结果

```
eval/
├── compare_model1.json          # 模型1详细结果
├── compare_model1.txt           # 模型1评估报告
├── compare_model2.json          # 模型2详细结果
├── compare_model2.txt           # 模型2评估报告
├── batch_evaluation_summary.json  # 批量评估汇总
└── model_comparison_table.txt   # 模型对比表格
```

**模型对比表格示例：**
```
| Model Name | CLIP Score | Semantic Sim | R@10 | Overall Score |
|------------|------------|--------------|------|---------------|
| qwen2.5    | 0.8234     | 0.7891       | 0.91 | 0.7823        |
| internvl   | 0.8156     | 0.7654       | 0.89 | 0.7612        |
| gpt4v      | 0.8567     | 0.8123       | 0.93 | 0.8234        |
```

---

## 🔧 环境配置

### 依赖安装

```bash
pip install torch torchvision
pip install transformers
pip install sentence-transformers
pip install scipy numpy
pip install lmdeploy  # for InternVL
pip install modelscope  # for Qwen
pip install qwen-vl-utils
```

### 模型下载

需要下载以下模型：

**Caption 生成：**
- Qwen2.5-VL-3B-Instruct 或 Qwen2.5-VL-7B-Instruct
- InternVL3_5-4B

**Caption 评估：**
- CLIP-ViT-Base-Patch32
- Sentence-Transformers all-mpnet-base-v2

### 目录结构

建议的目录结构：

```
eval/
├── data/
│   ├── multiview_images/    # 多视角图片输入
│   ├── captions/             # 生成的caption输出
│   └── groundTruth.json      # Ground truth数据
├── models/
│   ├── openai/
│   │   └── clip-vit-base-patch32/
│   ├── sentence-transformers/
│   │   └── all-mpnet-base-v2/
│   ├── OpenGVLab/
│   │   └── InternVL3_5-4B/
│   └── Qwen2.5-VL-3B-Instruct/
├── qwen.py
├── internvl.py
├── compare.py
└── compare.sh
```

---

## 📊 完整评估流程

### 步骤 1: 生成 Caption

```bash
# 使用 Qwen2.5-VL 生成
python qwen.py --input_dir ./data/multiview_images \
               --output_dir ./data/captions

# 使用 InternVL 生成
python internvl.py --input_dir ./data/multiview_images \
                   --output_dir ./data/captions
```

### 步骤 2: 评估单个模型

```bash
# 评估 Qwen2.5-VL 结果
python compare.py --generated ./data/captions/image_qwen2.5.json

# 评估 InternVL 结果
python compare.py --generated ./data/captions/image_internvl.json
```

### 步骤 3: 批量评估所有模型

```bash
./compare.sh
```

### 步骤 4: 分析结果

查看生成的报告文件：
```bash
# 查看详细报告
cat compare_qwen2.5.txt

# 查看对比表格
cat model_comparison_table.txt

# 查看JSON结果
cat compare_qwen2.5.json | jq
```

---

## 💡 使用技巧

### 1. 提高生成质量

**优化提示词：**
- 在脚本中修改提示词模板
- 添加更多示例和约束
- 调整生成参数（temperature, top_p等）

**使用更大的模型：**
```bash
# 使用7B模型替代3B
python qwen.py --model_path ./models/Qwen2.5-VL-7B-Instruct
```

### 2. 加速推理

**使用多卡并行：**
```bash
# InternVL 多卡推理
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 修改脚本中 tp=4
```

**批量处理：**
```bash
# 一次性处理所有图片
python qwen.py --input_dir ./data/multiview_images
```

### 3. 自定义评估

**修改评估权重：**

在 `compare.py` 中修改综合评分计算：
```python
results["overall_score"] = (
    results.get("clip_score", 0) * 0.30 +      # 调整权重
    results.get("semantic_similarity", 0) * 0.30 +
    results.get("r_at_10", 0) * 0.20 +
    results.get("lexical_similarity", 0) * 0.20
)
```

**添加新的评估指标：**

在 `CaptionEvaluator` 类中添加新方法：
```python
def calculate_custom_metric(self, generated, ground_truth):
    # 实现自定义评估逻辑
    return score
```

---

## 🐛 故障排除

### 常见问题

**1. CUDA Out of Memory**
```bash
# 使用更小的模型
python qwen.py --model_path ./models/Qwen2.5-VL-3B-Instruct

# 减少批处理大小
# 在脚本中修改 batch_size
```

**2. 模型加载失败**
```bash
# 检查模型路径
ls -la ./models/

# 检查模型文件完整性
du -sh ./models/Qwen2.5-VL-3B-Instruct/

# 重新下载模型
```

**3. 评估指标为0**
```bash
# 检查CLIP模型是否正确加载
# 查看日志输出

# 检查数据格式是否正确
cat ./data/captions/image_model.json | jq
```

**4. 图片文件名不匹配**
```bash
# 检查文件名格式
# 确保符合 UUID_数字_ac001001.jpg 格式

# 查看提取的图片名
# 在脚本中添加调试输出
```

---

## 📈 性能优化

### 生成速度

| 模型 | 单张耗时 | 100张耗时 | GPU显存 |
|------|---------|----------|---------|
| Qwen2.5-VL-3B | ~5s | ~8min | ~12GB |
| Qwen2.5-VL-7B | ~8s | ~13min | ~24GB |
| InternVL3_5-4B | ~6s | ~10min | ~16GB |

### 评估速度

| 样本数 | 耗时 | GPU显存 |
|--------|------|---------|
| 100 | ~2min | ~4GB |
| 500 | ~8min | ~4GB |
| 1000 | ~15min | ~4GB |

---

## 📚 参考资源

- [Qwen2.5-VL 文档](https://github.com/QwenLM/Qwen2-VL)
- [InternVL 文档](https://github.com/OpenGVLab/InternVL)
- [CLIP 论文](https://arxiv.org/abs/2103.00020)
- [Sentence-Transformers](https://www.sbert.net/)

---

## 📝 数据格式说明

### Ground Truth 格式

```json
[
  {
    "images": ["UUID_1_ac001001.jpg"],
    "caption": "Athenian black-figure lekythos, c. 525–475 BCE, depicting Herakles and the boar; Marathon, Attica."
  },
  {
    "images": ["UUID_2_ac001001.jpg"],
    "caption": "..."
  }
]
```

### 生成结果格式

```json
[
  {
    "images": ["UUID_1_ac001001.jpg"],
    "caption": "生成的描述文本",
    "data_by": "模型名称"
  }
]
```

---

**💡 提示**: 建议先在小数据集上测试完整流程，确认无误后再进行大规模评估。
