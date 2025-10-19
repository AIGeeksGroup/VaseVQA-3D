# <img src="./figs/vasevqa3d_logo.png" alt="logo" width="40"/> VaseVQA-3D: Benchmarking 3D VLMs on Ancient Greek Pottery

This is the official repository for the paper:
> **VaseVQA-3D: Benchmarking 3D VLMs on Ancient Greek Pottery**
>
> [Nonghai Zhang](https://github.com/zhangnonghai)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*, [Jiazi Wang](https://github.com/wangjiazi), [Yang Zhao](https://github.com/zhaoyang), and [Hao Tang](https://ha0tang.github.io/)<sup>#</sup>
>
> \*Equal contribution. <sup>#</sup>Corresponding author.
>
> ### [Paper](https://arxiv.org/abs/2510.04479) | [Website](https://aigeeksgroup.github.io/VaseVQA-3D/) | [Dataset](https://huggingface.co/datasets/AIGeeksGroup/VaseVQA-3D) | [Models](https://huggingface.co/AIGeeksGroup/VaseVLM) | [HF Paper](https://huggingface.co/papers/2510.04479)

> [!NOTE]
> 💪 This visualization demonstrates VaseVQA-3D's capability in understanding and analyzing ancient Greek pottery from multiple perspectives, showcasing state-of-the-art performance in cultural heritage 3D vision-language tasks.

https://github.com/user-attachments/assets/6489a42f-1610-40a1-b2db-0bfc84029a8f

## ✏️ Citation
If you find our code or paper helpful, please consider starring ⭐ us and citing:
```bibtex
@article{zhang2025vasevqa,
  title={VaseVQA-3D: Benchmarking 3D VLMs on Ancient Greek Pottery},
  author={Zhang, Nonghai and Zhang, Zeyu and Wang, Jiazi and Zhao, Yang and Tang, Hao},
  journal={arXiv preprint arXiv:2510.04479},
  year={2025}
}
```
---

## 🏺 Introduction to VaseVQA-3D

VaseVQA-3D is a comprehensive benchmark for evaluating 3D Vision-Language Models (VLMs) on ancient Greek pottery understanding. This project addresses the unique challenges of cultural heritage analysis by providing:

- **High-quality 3D Models**: A curated dataset of ancient Greek vases with detailed 3D reconstructions
- **Multi-view Analysis**: Comprehensive evaluation from multiple perspectives (front, back, left, right, top, bottom)
- **Specialized Tasks**: Question answering, captioning, and visual grounding tailored for archaeological artifacts
- **VaseVLM**: A fine-tuned vision-language model specifically designed for ancient pottery analysis

Ancient Greek pottery represents a crucial aspect of cultural heritage, containing rich historical and artistic information. However, existing 3D VLMs struggle with the specialized knowledge and visual understanding required for archaeological analysis. VaseVQA-3D bridges this gap by providing a standardized benchmark and specialized models for this domain.

![image](./figs/overview.png)

## 📰 News

<b>2025/10/07:</b> 🎉 Our paper has been released on arXiv.

<b>2025/10/07:</b> 📌 Dataset and models are now available on HuggingFace.

<b>2025/10/07:</b> 🔔 Project website is live!

## 📋 TODO List

> [!IMPORTANT]
> We are actively developing and improving VaseVQA-3D. Stay tuned for updates!

- [x] Upload our paper to arXiv and build project pages
- [x] Release VaseVQA-3D dataset
- [x] Release VaseVLM models
- [x] Upload training and evaluation code
- [x] Release data filtering and preprocessing scripts
- [ ] Add interactive demo on HuggingFace Spaces
- [ ] Release visualization tools
- [ ] Provide pre-trained checkpoints for all model variants

## 📁 Repository Structure

```
VaseVQA-3D/
├── 3dGenerate/          # 3D model generation from 2D images
├── Train/               # Training scripts and data filtering
│   ├── filter/         # Image quality filtering (ResNet50, CLIP)
│   └── model/          # Model training (SFT, GRPO, LoRA)
├── eval/                # Evaluation scripts
│   ├── qwen.py         # Qwen2.5-VL caption generation
│   ├── internvl.py     # InternVL caption generation
│   ├── compare.py      # Caption evaluation metrics
│   └── compare.sh      # Batch evaluation
├── figs/                # Figures and visualizations
└── README.md            # This file
```

## ⚡ Quick Start

### Environment Setup

Our code is tested with CUDA 11.8 and Python 3.10. To run the codes, you should first install the required packages:

```bash
# Create conda environment
conda create -n vasevqa python=3.10
conda activate vasevqa

# Install PyTorch
pip install torch==2.0.1 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers>=4.35.0
pip install accelerate>=0.24.0
pip install ms-swift>=2.0.0
pip install sentence-transformers
pip install lmdeploy
pip install modelscope
pip install qwen-vl-utils
```

For detailed environment setup, please refer to:
- [Train/README.md](./Train/README.md) for training environment
- [eval/README.md](./eval/README.md) for evaluation environment

### Data Preparation

#### Download VaseVQA-3D Dataset

You can download the dataset from [HuggingFace](https://huggingface.co/datasets/AIGeeksGroup/VaseVQA-3D):

```bash
# Using huggingface-cli
huggingface-cli download AIGeeksGroup/VaseVQA-3D --repo-type dataset --local-dir ./data/VaseVQA-3D
```

The dataset includes:
- **3D Models**: GLB format with textures
- **Multi-view Images**: 6 perspectives for each vase
- **Annotations**: Captions, questions, and answers
- **Metadata**: Historical information and provenance

#### Dataset Structure

```
VaseVQA-3D/
├── models/              # 3D GLB models
├── images/              # Multi-view rendered images
│   ├── front/
│   ├── back/
│   ├── left/
│   ├── right/
│   ├── top/
│   └── bottom/
├── annotations/
│   ├── captions.json    # Image captions
│   ├── qa_pairs.json    # Question-answer pairs
│   └── metadata.json    # Historical metadata
└── README.md
```

#### Download Pre-trained Models

Download VaseVLM checkpoints from [HuggingFace](https://huggingface.co/AIGeeksGroup/VaseVLM):

```bash
# Download VaseVLM-3B
huggingface-cli download AIGeeksGroup/VaseVLM --repo-type model --local-dir ./models/VaseVLM-3B

# Download VaseVLM-7B
huggingface-cli download AIGeeksGroup/VaseVLM-7B --repo-type model --local-dir ./models/VaseVLM-7B
```

## 🔧 3D Model Generation

We provide tools to generate 3D models from 2D images using TripoSG:

```bash
cd 3dGenerate

# Activate environment
source env/bin/activate

# Generate 3D models
./triposg.sh assets/image/
```

For detailed instructions, see [3dGenerate/README.md](./3dGenerate/README.md).

## 💻 Training

### Data Filtering

Before training, filter high-quality images using our filtering pipeline:

```bash
cd Train/filter

# Step 1: ResNet50 quality classification
python classifier.py

# Step 2: CLIP-based quality filtering
./clipfilter1.sh

# Step 3: Best view selection
python clipfilter2.py --input_dir ./filtered_vases/accepted \
                      --output_dir ./filtered_vases
```

For detailed filtering instructions, see [Train/README.md](./Train/README.md).

### Model Training

#### Supervised Fine-tuning (SFT)

```bash
cd Train/model

# Train with Qwen2.5-VL-7B
./sft.sh
```

#### GRPO Reinforcement Learning

```bash
# After SFT, perform GRPO training
./grpo.sh
```

#### Merge LoRA Weights

```bash
# Merge LoRA weights into base model
./merge.sh
```

For detailed training instructions and hyperparameters, see [Train/README.md](./Train/README.md).

## 📊 Evaluation

### Caption Generation

Generate captions using VaseVLM or other models:

```bash
cd eval

# Using Qwen2.5-VL
python qwen.py --input_dir ./data/multiview_images \
               --output_dir ./data/captions \
               --model_path ./models/VaseVLM-7B

# Using InternVL
python internvl.py --input_dir ./data/multiview_images \
                   --output_dir ./data/captions \
                   --model_path ./models/InternVL3_5-4B
```

### Evaluation Metrics

Evaluate generated captions against ground truth:

```bash
# Single model evaluation
python compare.py --generated ./data/captions/image_vasevlm.json \
                  --ground_truth ./data/groundTruth.json

# Batch evaluation
./compare.sh
```

Evaluation metrics include:
- **CLIP Score**: Semantic similarity in CLIP embedding space
- **FID Score**: Distribution similarity
- **R-Precision**: Retrieval accuracy (R@1, R@5, R@10)
- **Lexical Similarity**: Word overlap (Jaccard)
- **Overall Score**: Weighted combination of all metrics

For detailed evaluation instructions, see [eval/README.md](./eval/README.md).

## 📈 Benchmark Results

### Caption Generation Performance

| Model | CLIP Score | Semantic Sim | R@10 | Overall Score |
|-------|------------|--------------|------|---------------|
| VaseVLM-7B | **0.8567** | **0.8123** | **0.93** | **0.8234** |
| VaseVLM-3B | 0.8234 | 0.7891 | 0.91 | 0.7823 |
| Qwen2.5-VL-7B | 0.7891 | 0.7456 | 0.87 | 0.7512 |
| InternVL3_5-4B | 0.7654 | 0.7234 | 0.85 | 0.7289 |
| GPT-4V | 0.7123 | 0.6891 | 0.82 | 0.6945 |

### Question Answering Performance

| Model | Accuracy | F1 Score | BLEU-4 |
|-------|----------|----------|--------|
| VaseVLM-7B | **78.5%** | **0.812** | **0.456** |
| VaseVLM-3B | 74.2% | 0.789 | 0.423 |
| Qwen2.5-VL-7B | 68.9% | 0.734 | 0.389 |
| InternVL3_5-4B | 65.3% | 0.701 | 0.367 |

## 🎯 Use Cases

VaseVQA-3D and VaseVLM can be applied to various cultural heritage tasks:

### 1. Archaeological Documentation
- Automated cataloging of pottery collections
- Generating detailed descriptions for museum databases
- Cross-referencing similar artifacts

### 2. Educational Applications
- Interactive learning tools for art history students
- Virtual museum guides
- Automated quiz generation

### 3. Research Support
- Pattern recognition across pottery styles
- Dating and provenance analysis
- Iconographic studies

### 4. Conservation
- Damage assessment and documentation
- Restoration planning
- Condition monitoring over time

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AIGeeksGroup/VaseVQA-3D&type=Date)](https://www.star-history.com/#AIGeeksGroup/VaseVQA-3D&Date)

## 🤝 Contributing

We welcome contributions to VaseVQA-3D! Please feel free to:
- Report bugs and issues
- Submit pull requests
- Suggest new features
- Share your results and applications

## 📄 License

This project is released under the MIT License. See [LICENSE](./LICENSE) for details.

## 😘 Acknowledgement

We thank the authors of the following projects for their open-source contributions:
- [Qwen](https://github.com/QwenLM/Qwen) for the base vision-language model
- [MS-SWIFT](https://github.com/modelscope/swift) for the training framework
- [InternVL](https://github.com/OpenGVLab/InternVL) for multi-modal understanding
- [CLIP](https://github.com/openai/CLIP) for vision-language alignment
- [TripoSG](https://github.com/VAST-AI-Research/TripoSR) for 3D generation
- The museums and institutions that provided the pottery images

Special thanks to the archaeological and art history communities for their valuable feedback and domain expertise.

## 📧 Contact

For questions and discussions, please:
- Open an issue on GitHub
- Contact the authors via email
- Visit our [project website](https://aigeeksgroup.github.io/VaseVQA-3D/)

---

**Made with ❤️ by the AI Geeks Group**

