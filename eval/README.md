# VaseVQA-3D Evaluation Module

This directory contains the evaluation-related code for the VaseVQA-3D project, mainly used for generating multi-view image captions and evaluating generation quality.

## 📁 Directory Structure

```
eval/
├── qwen.py                # Qwen2.5-VL multi-view caption generator
├── internvl.py            # InternVL multi-view caption generator
├── compare.py             # Caption evaluation script
├── compare.sh             # Batch evaluation script
└── README.md              # This document
```

---

## 🎯 Overview

The evaluation module includes two main functions:

1. **Caption Generation**: Use vision-language models (Qwen2.5-VL or InternVL) to generate descriptions for multi-view images
2. **Caption Evaluation**: Evaluate the quality of generated descriptions using multiple metrics

---

## 📝 Caption Generation

### 1. Qwen2.5-VL Caption Generator

**File:** `qwen.py`

**Function:** Analyze multi-view images and generate comprehensive descriptions based on the Qwen2.5-VL model

#### Usage

**Batch Processing:**
```bash
python qwen.py --input_dir ./data/multiview_images \
               --output_dir ./data/captions \
               --model_path ./models/Qwen2.5-VL-3B-Instruct
```

**Process Single Model:**
```bash
python qwen.py --single_model ./data/multiview_images/model1 \
               --output_dir ./data/captions \
               --model_path ./models/Qwen2.5-VL-7B-Instruct
```

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_dir` | Input directory containing multi-view images | `./data/multiview_images` |
| `--output_dir` | Caption output directory | `./data/captions` |
| `--model_path` | Qwen2.5-VL model path | `./models/Qwen2.5-VL-3B-Instruct` |
| `--single_model` | Process single model directory (optional) | `None` |

#### Input Data Format

Multi-view image directory structure:
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

#### Output Format

Generated JSON file format:
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

Output files:
- `image_qwen2.5-vl-3B-sft.json` - Generated captions
- `qwen2.5_caption_generation_report.json` - Processing report

#### Prompt Template

The script uses the following prompt to guide the model:

```
Please analyze this ancient Greek vase model's multi-view images. I have provided 6 perspective images:
front, back, left, right, top, bottom

Please generate a concise and accurate caption description, similar to:
"Athenian black-figure lekythos, c. 525–475 BCE, depicting Herakles and the boar; Marathon, Attica."

Requirements:
1. Include vessel type name (e.g., lekythos, amphora, hydria, etc.)
2. Include production technique (e.g., black-figure, red-figure)
3. Include approximate date
4. Briefly describe main decorative content
5. If identifiable, include possible provenance
```

---

### 2. InternVL Caption Generator

**File:** `internvl.py`

**Function:** Analyze multi-view images and generate comprehensive descriptions based on the InternVL model

#### Usage

**Batch Processing:**
```bash
python internvl.py --input_dir ./data/multiview_images \
                   --output_dir ./data/captions \
                   --model_path ./models/OpenGVLab/InternVL3_5-4B
```

**Process Single Model:**
```bash
python internvl.py --single_model ./data/multiview_images/model1 \
                   --output_dir ./data/captions \
                   --model_path ./models/OpenGVLab/InternVL3_5-4B
```

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_dir` | Input directory containing multi-view images | `./data/multiview_images` |
| `--output_dir` | Caption output directory | `./data/captions` |
| `--model_path` | InternVL model path | `./models/OpenGVLab/InternVL3_5-4B` |
| `--single_model` | Process single model directory (optional) | `None` |

#### Output Format

```json
[
  {
    "images": ["UUID_1_ac001001.jpg"],
    "caption": "...",
    "data_by": "internvl"
  }
]
```

Output files:
- `image_internvl.json` - Generated captions
- `internvl_caption_generation_report.json` - Processing report

#### Multi-GPU Inference

InternVL supports multi-GPU parallel inference:

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# Modify tp parameter in script
# tp=1 for single GPU, tp=6 for 6-GPU parallel
```

---

## 📊 Caption Evaluation

### 1. Single Model Evaluation

**File:** `compare.py`

**Function:** Evaluate the similarity between generated captions and ground truth, calculating multiple evaluation metrics

#### Usage

```bash
python compare.py --generated ./data/captions/image_qwen2.5.json \
                  --ground_truth ./data/groundTruth.json \
                  --output results/qwen_eval.json
```

#### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--generated` | Generated caption JSON file path | **Required** |
| `--ground_truth` | Ground truth file path | `./data/groundTruth.json` |
| `--output` | Output report file path (optional) | Auto-generated |

#### Evaluation Metrics

The script calculates the following evaluation metrics:

##### 1. CLIP Score
- **Description**: Measures similarity between generated captions and ground truth in CLIP embedding space
- **Range**: 0-1, higher is better
- **Weight**: 25%

##### 2. FID Score
- **Description**: Fréchet Inception Distance based on sentence embeddings
- **Range**: 0+, lower is better
- **Purpose**: Evaluate difference between generated and real distributions

##### 3. Semantic Similarity
- **Description**: CLIP-based semantic similarity
- **Range**: 0-1, higher is better
- **Weight**: 25%

##### 4. R-Precision
- **R@1**: Top-1 retrieval accuracy (weight 5%)
- **R@5**: Top-5 retrieval accuracy (weight 10%)
- **R@10**: Top-10 retrieval accuracy (weight 20%)

##### 5. Lexical Similarity
- **Description**: Word overlap based on Jaccard similarity
- **Range**: 0-1, higher is better
- **Weight**: 15%

##### 6. Overall Score
- **Description**: Weighted average of all metrics
- **Range**: 0-1, higher is better
- **Formula**:
  ```
  Overall Score = CLIP Score × 0.25
                + Semantic Similarity × 0.25
                + R@10 × 0.20
                + Lexical Similarity × 0.15
                + R@5 × 0.10
                + R@1 × 0.05
  ```

#### Output Results

The script generates two files:

**1. JSON Results File** (`compare_model_name.json`)
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

**2. Text Report** (`compare_model_name.txt`)

Contains detailed evaluation report in the following format:

```markdown
# Caption Evaluation Report - qwen2.5

## Basic Information
- Model Name: qwen2.5
- Evaluation Samples: 100
- Generated File: ./data/captions/image_qwen2.5.json
- Ground Truth File: ./data/groundTruth.json

## Evaluation Metrics

### 1. Core Metrics
- CLIP Score: 0.8234
- FID Score: 12.5678
- Semantic Similarity: 0.7891

### 2. R-Precision Results
- R@1: 65.00%
- R@5: 82.00%
- R@10: 91.00%

### 3. Lexical Matching
- Lexical Similarity: 0.4567

### 4. Overall Score
- Overall Score: 0.7823

## Evaluation Summary
Model performance on caption generation task:
1. CLIP Similarity: Excellent
2. Semantic Understanding: Good
3. Retrieval Performance: Excellent
4. Lexical Matching: Fair
5. Overall Score: Good
```

---

### 2. Batch Evaluation

**File:** `compare.sh`

**Function:** Batch evaluate all model result files in a folder

#### Usage

```bash
# Modify path configuration in script
CAPTION_DIR="./data/captions"
GROUND_TRUTH="./data/groundTruth.json"

# Run batch evaluation
./compare.sh
```

#### Processing Flow

1. Scan all `image_*.json` files in `CAPTION_DIR` directory
2. Skip `groundTruth.json` file
3. Call `compare.py` for evaluation on each file
4. Generate detailed reports for each model
5. Aggregate all results to `batch_evaluation_summary.json`
6. Generate model comparison table

#### Output Results

```
eval/
├── compare_model1.json          # Model 1 detailed results
├── compare_model1.txt           # Model 1 evaluation report
├── compare_model2.json          # Model 2 detailed results
├── compare_model2.txt           # Model 2 evaluation report
├── batch_evaluation_summary.json  # Batch evaluation summary
└── model_comparison_table.txt   # Model comparison table
```

**Model Comparison Table Example:**
```
| Model Name | CLIP Score | Semantic Sim | R@10 | Overall Score |
|------------|------------|--------------|------|---------------|
| qwen2.5    | 0.8234     | 0.7891       | 0.91 | 0.7823        |
| internvl   | 0.8156     | 0.7654       | 0.89 | 0.7612        |
| gpt4v      | 0.8567     | 0.8123       | 0.93 | 0.8234        |
```

---

## 🔧 Environment Setup

### Dependency Installation

```bash
pip install torch torchvision
pip install transformers
pip install sentence-transformers
pip install scipy numpy
pip install lmdeploy  # for InternVL
pip install modelscope  # for Qwen
pip install qwen-vl-utils
```

### Model Download

Download the following models:

**Caption Generation:**
- Qwen2.5-VL-3B-Instruct or Qwen2.5-VL-7B-Instruct
- InternVL3_5-4B

**Caption Evaluation:**
- CLIP-ViT-Base-Patch32
- Sentence-Transformers all-mpnet-base-v2

### Directory Structure

Recommended directory structure:

```
eval/
├── data/
│   ├── multiview_images/    # Multi-view image input
│   ├── captions/             # Generated caption output
│   └── groundTruth.json      # Ground truth data
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

## 📊 Complete Evaluation Pipeline

### Step 1: Generate Captions

```bash
# Generate using Qwen2.5-VL
python qwen.py --input_dir ./data/multiview_images \
               --output_dir ./data/captions

# Generate using InternVL
python internvl.py --input_dir ./data/multiview_images \
                   --output_dir ./data/captions
```

### Step 2: Evaluate Single Model

```bash
# Evaluate Qwen2.5-VL results
python compare.py --generated ./data/captions/image_qwen2.5.json

# Evaluate InternVL results
python compare.py --generated ./data/captions/image_internvl.json
```

### Step 3: Batch Evaluate All Models

```bash
./compare.sh
```

### Step 4: Analyze Results

View generated report files:
```bash
# View detailed report
cat compare_qwen2.5.txt

# View comparison table
cat model_comparison_table.txt

# View JSON results
cat compare_qwen2.5.json | jq
```

---

## 💡 Usage Tips

### 1. Improve Generation Quality

**Optimize Prompts:**
- Modify prompt template in script
- Add more examples and constraints
- Adjust generation parameters (temperature, top_p, etc.)

**Use Larger Models:**
```bash
# Use 7B model instead of 3B
python qwen.py --model_path ./models/Qwen2.5-VL-7B-Instruct
```

### 2. Accelerate Inference

**Use Multi-GPU Parallel:**
```bash
# InternVL multi-GPU inference
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Modify tp=4 in script
```

**Batch Processing:**
```bash
# Process all images at once
python qwen.py --input_dir ./data/multiview_images
```

### 3. Custom Evaluation

**Modify Evaluation Weights:**

Modify overall score calculation in `compare.py`:
```python
results["overall_score"] = (
    results.get("clip_score", 0) * 0.30 +      # Adjust weights
    results.get("semantic_similarity", 0) * 0.30 +
    results.get("r_at_10", 0) * 0.20 +
    results.get("lexical_similarity", 0) * 0.20
)
```

**Add New Evaluation Metrics:**

Add new method in `CaptionEvaluator` class:
```python
def calculate_custom_metric(self, generated, ground_truth):
    # Implement custom evaluation logic
    return score
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Use smaller model
python qwen.py --model_path ./models/Qwen2.5-VL-3B-Instruct

# Reduce batch size
# Modify batch_size in script
```

**2. Model Loading Failed**
```bash
# Check model path
ls -la ./models/

# Check model file integrity
du -sh ./models/Qwen2.5-VL-3B-Instruct/

# Re-download model
```

**3. Evaluation Metrics are 0**
```bash
# Check if CLIP model loaded correctly
# View log output

# Check data format
cat ./data/captions/image_model.json | jq
```

**4. Image Filename Mismatch**
```bash
# Check filename format
# Ensure it follows UUID_number_ac001001.jpg format

# View extracted image names
# Add debug output in script
```

---

## 📈 Performance Benchmarks

### Generation Speed

| Model | Per Image | 100 Images | GPU Memory |
|-------|-----------|------------|------------|
| Qwen2.5-VL-3B | ~5s | ~8min | ~12GB |
| Qwen2.5-VL-7B | ~8s | ~13min | ~24GB |
| InternVL3_5-4B | ~6s | ~10min | ~16GB |

### Evaluation Speed

| Samples | Time | GPU Memory |
|---------|------|------------|
| 100 | ~2min | ~4GB |
| 500 | ~8min | ~4GB |
| 1000 | ~15min | ~4GB |

---

## 📚 References

- [Qwen2.5-VL Documentation](https://github.com/QwenLM/Qwen2-VL)
- [InternVL Documentation](https://github.com/OpenGVLab/InternVL)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Sentence-Transformers](https://www.sbert.net/)

---

## 📝 Data Format Specification

### Ground Truth Format

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

### Generated Results Format

```json
[
  {
    "images": ["UUID_1_ac001001.jpg"],
    "caption": "Generated description text",
    "data_by": "Model name"
  }
]
```

---

**💡 Tip**: It is recommended to test the complete pipeline on a small dataset first to ensure everything works correctly before large-scale evaluation.
