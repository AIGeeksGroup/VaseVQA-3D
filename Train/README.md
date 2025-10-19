# VaseVQA-3D Training Module

This directory contains the training-related code for the VaseVQA-3D project, mainly divided into two sub-modules: **Data Filtering** and **Model Training**.

## 📁 Directory Structure

```
Train/
├── filter/                    # Data filtering module
│   ├── classifier.py         # Image quality classifier
│   ├── clipfilter1.py        # CLIP filter (quality filtering)
│   ├── clipfilter1.sh        # CLIP filter startup script
│   ├── clipfilter2.py        # CLIP filter (view selection)
│   └── model.py              # ResNet50 classification model
├── model/                     # Model training module
│   ├── sft.sh                # Supervised fine-tuning script
│   ├── grpo.sh               # GRPO reinforcement learning training script
│   ├── merge.sh              # LoRA weight merging script
│   ├── hf2megatron.sh        # HuggingFace to Megatron conversion script
│   └── requirements.txt      # Python dependencies
└── README.md                  # This document
```

---

## 🔍 Data Filtering Module

The data filtering module is used to filter high-quality images from the raw dataset, including ResNet50-based quality classification and CLIP-based view selection.

### 1. ResNet50 Image Quality Classifier

#### File Description
- **`model.py`**: ResNet50 binary classification model definition and training code
- **`classifier.py`**: Batch classification using trained model

#### Features
- Uses pre-trained ResNet50 as backbone network
- Binary classification task: 0 (low quality) / 1 (high quality)
- Supports batch processing and GPU acceleration
- Automatic data augmentation and validation

#### Usage

**Train Model:**
```bash
cd filter
python model.py
```

Configuration parameters (modify in `model.py`):
```python
DEFAULT_GPU = 2          # GPU number to use
BATCH_SIZE = 256         # Batch size
NUM_EPOCHS = 200         # Training epochs
DATA_DIR = 'images2'     # Data directory
```

Data directory structure:
```
images2/
├── 0/                   # Low quality images
│   ├── image1.jpg
│   └── ...
└── 1/                   # High quality images
    ├── image2.jpg
    └── ...
```

**Use Classifier:**
```bash
python classifier.py
```

Configuration parameters (modify in `classifier.py`):
```python
DEFAULT_GPU = 2                                    # GPU number
MODEL_PATH = 'best_image_quality_model.pth'        # Model path
SOURCE_DIR = 'images'                              # Source image directory
TARGET_DIR = 'images3'                             # Target directory
BATCH_SIZE = 32                                    # Batch size
```

Output:
```
images3/
├── 0/                   # Images classified as low quality
└── 1/                   # Images classified as high quality
```

---

### 2. CLIP Image Quality Filter

#### File Description
- **`clipfilter1.py`**: CLIP-based image quality filtering
- **`clipfilter1.sh`**: Interactive startup script
- **`clipfilter2.py`**: CLIP-based best view selection

#### clipfilter1 - Quality Filtering

**Features:**
- Uses CLIP model to evaluate image quality
- Supports quality distribution analysis
- Automatic threshold recommendation
- Batch processing and progress tracking

**Usage:**

Method 1: Use interactive script (recommended)
```bash
./clipfilter1.sh
```

Script options:
1. 🔍 Analyze dataset quality distribution only
2. 🎯 Automatic filtering (analyze then filter)
3. ⚙️ Custom threshold filtering
4. 💻 Force CPU mode
5. 🎮 Use GPU 1
6. 📊 Quick analysis (100 samples)

Method 2: Run Python script directly
```bash
# Analyze quality distribution
python clipfilter1.py --analyze_only --input_dir ./images3/1

# Execute filtering
python clipfilter1.py --input_dir ./images3/1 \
                      --output_dir ./filtered_vases \
                      --threshold 0.15 \
                      --model_cache_dir ./model
```

**Parameters:**
- `--input_dir`: Input image directory
- `--output_dir`: Output directory
- `--threshold`: Quality threshold (optional, auto-analyze if not specified)
- `--analyze_only`: Only analyze quality distribution
- `--sample_size`: Sample size for analysis (default 500)
- `--device`: Specify device (cuda:0, cuda:1, cpu, etc.)
- `--model_cache_dir`: CLIP model cache directory
- `--auto_select_device`: Interactive device selection

**Output:**
```
filtered_vases/
├── accepted/            # Images passing quality check
├── rejected/            # Images failing quality check
└── filtering_report.json  # Detailed filtering report
```

#### clipfilter2 - Best View Selection

**Features:**
- Selects the view with highest CLIP score for each vase
- Automatically identifies multi-view images
- Generates detailed selection report

**Usage:**
```bash
python clipfilter2.py --input_dir ./filtered_vases/accepted \
                      --output_dir ./filtered_vases \
                      --device cuda:0
```

**Parameters:**
- `--input_dir`: Input directory containing multi-view images
- `--output_dir`: Output directory
- `--device`: Specify device

**Output:**
```
filtered_vases/
├── best_views/                    # Best view for each vase
├── best_views_report.json         # Detailed report
└── filtering_stats.json           # Statistics
```

---

## 🚀 Model Training Module

The model training module is based on the MS-SWIFT framework, supporting Supervised Fine-Tuning (SFT) and Reinforcement Learning (GRPO) training.

### Environment Setup

**1. Create Conda environment:**
```bash
conda create -n swift_env python=3.10
conda activate swift_env
```

**2. Install dependencies:**
```bash
cd model
pip install -r requirements.txt
```

**Main dependencies:**
- PyTorch 2.0+
- MS-SWIFT 2.0+
- Transformers 4.35+
- Other deep learning tools

---

### 1. Supervised Fine-tuning (SFT)

**Script:** `sft.sh`

**Function:** Supervised fine-tuning of vision-language models using annotated data

**Usage:**
```bash
./sft.sh
```

**Main parameters:**
```bash
CUDA_VISIBLE_DEVICES=0              # GPU to use
MAX_PIXELS=1003520                  # Maximum pixels
--model './models/Qwen2.5-VL-7B-Instruct'  # Model path
--dataset './data/video_training_dataset.json'  # Training data
--train_type lora                   # Training type (lora/full)
--num_train_epochs 2                # Training epochs
--per_device_train_batch_size 1     # Batch size
--learning_rate 1e-4                # Learning rate
--lora_rank 8                       # LoRA rank
--output_dir output/7B              # Output directory
```

**Data format:**
```json
[
  {
    "images": ["path/to/image.jpg"],
    "caption": "Description text",
    "conversations": [...]
  }
]
```

**Output:**
- Trained LoRA weights
- Training logs and TensorBoard records
- Periodically saved checkpoints

---

### 2. GRPO Reinforcement Learning Training

**Script:** `grpo.sh`

**Function:** Reinforcement learning training using GRPO algorithm to optimize model output quality

**Usage:**
```bash
./grpo.sh
```

**Main parameters:**
```bash
--rlhf_type grpo                    # Reinforcement learning type
--model './output/checkpoint-merged'  # Base model
--external_plugins ./plugin/plugin.py  # External plugins
--reward_funcs external_vase_acc    # Reward function
--dataset './data/grpo_video_dataset.json'  # Training data
--num_train_epochs 1                # Training epochs
--learning_rate 1e-6                # Learning rate
--num_generations 4                 # Number of generations per iteration
--temperature 1.0                   # Sampling temperature
--beta 0.001                        # KL divergence coefficient
--system ./prompt.txt               # System prompt
```

**Reward function:**
Define custom reward function in `plugin/plugin.py`, for example:
```python
def external_vase_acc(responses, references):
    # Calculate reward scores
    return scores
```

---

### 3. LoRA Weight Merging

**Script:** `merge.sh`

**Function:** Merge trained LoRA weights into base model

**Usage:**
```bash
./merge.sh
```

**Parameters:**
```bash
--adapters ./output/checkpoint-600  # LoRA weights path
--merge_lora true                   # Enable merging
```

**Output:**
- Merged complete model
- Can be directly used for inference or further training

---

### 4. HuggingFace to Megatron Conversion

**Script:** `hf2megatron.sh`

**Function:** Convert HuggingFace format model to Megatron format

**Usage:**
```bash
./hf2megatron.sh
```

**Parameters:**
```bash
--model './models/Qwen2.5-VL-3B-Instruct'  # Input model
--to_mcore true                     # Convert to Megatron format
--torch_dtype bfloat16              # Data type
--output_dir output/Qwen2.5-VL-3B-Instruct-mcore  # Output directory
--test_convert_precision true       # Test conversion precision
```

---

## 📊 Complete Training Pipeline

### Data Preparation and Filtering

```bash
# 1. Filter low-quality images using ResNet50 classifier
cd filter
python classifier.py

# 2. Further filtering using CLIP filter
./clipfilter1.sh
# Select option 2: Automatic filtering

# 3. Select best views
python clipfilter2.py --input_dir ./filtered_vases/accepted \
                      --output_dir ./filtered_vases
```

### Model Training

```bash
cd ../model

# 1. Supervised fine-tuning
./sft.sh

# 2. Merge LoRA weights
./merge.sh

# 3. GRPO reinforcement learning (optional)
./grpo.sh

# 4. Merge weights again
./merge.sh
```

---

## 🔧 Configuration

### GPU Configuration

All scripts support GPU configuration via environment variables:

```bash
# Single GPU
export CUDA_VISIBLE_DEVICES=0

# Multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
```

### Proxy Configuration

If you need to use a proxy to download models:

```bash
export http_proxy=http://your_proxy:port
export https_proxy=http://your_proxy:port
```

### Model Cache

Set model cache directory:

```bash
export HF_HOME=./cache
export TRANSFORMERS_CACHE=./cache
```

---

## 📝 Notes

1. **Memory Requirements**
   - ResNet50 training: ≥ 8GB
   - SFT training (7B model): ≥ 24GB
   - GRPO training: ≥ 32GB

2. **Data Format**
   - Image format: PNG, JPG, JPEG
   - Training data: JSON format
   - Ensure correct data paths

3. **Training Time**
   - ResNet50 training: 2-4 hours (200 epochs)
   - SFT training: Hours to days depending on data size
   - GRPO training: 2-3x slower than SFT

4. **Checkpoint Management**
   - Save checkpoints periodically
   - Set `save_total_limit` to limit checkpoint count
   - Backup important checkpoints promptly

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
--per_device_train_batch_size 1

# Enable gradient checkpointing
--gradient_checkpointing true

# Use smaller model
```

**2. Model Loading Failed**
```bash
# Check model path
ls -la ./models/

# Check permissions
chmod -R 755 ./models/

# Re-download model
```

**3. Training Interrupted**
```bash
# Resume from checkpoint
--resume_from_checkpoint ./output/checkpoint-XXX
```

---

## 📚 References

- [MS-SWIFT Documentation](https://github.com/modelscope/swift)
- [Qwen2.5-VL Model](https://huggingface.co/Qwen)
- [CLIP Model](https://github.com/openai/CLIP)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

---

**💡 Tip**: It is recommended to test the complete pipeline on a small dataset first to ensure everything works correctly before large-scale training.
