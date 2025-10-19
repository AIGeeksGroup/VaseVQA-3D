# TripoSG 3D Model Generation Tool

## 🎯 Overview

This is a 3D model generation tool based on TripoSG that can generate high-quality 3D GLB model files from single 2D images. The tool integrates a complete pipeline including image segmentation, 3D mesh generation, multi-view rendering, and texture mapping.

## 📁 Project Structure

```
3dGenerate/
├── triposg.py              # Main processing script (batch processing)
├── triposg.sh              # Startup script
├── requirements.txt        # Python dependencies list
├── assets/
│   └── image/
│       └── 1.png          # Example image
├── output2/                # Output directory (auto-created)
└── README.md              # This document
```

## 🔧 Environment Setup

### 1. Create Virtual Environment

```bash
# Create Python virtual environment
python -m venv env

# Activate environment
source env/bin/activate  # Linux/macOS
# or
env\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### Main Dependencies:
- **3D Processing**: trimesh, pyrender, open3d, pymeshlab
- **Deep Learning**: torch, torchvision, transformers, diffusers
- **Image Processing**: opencv-python, Pillow, scikit-image
- **Others**: CLIP, LPIPS, tqdm, etc.

### 3. Download Pre-trained Models

Download the following models to the `checkpoints/` directory:
- `RMBG-1.4` - Background removal model
- `TripoSG` - 3D generation model
- `RealESRGAN_x2plus.pth` - Texture super-resolution model
- `big-lama.pt` - Texture inpainting model

## 🚀 Usage

### Method 1: Using Shell Script (Recommended)

```bash
./triposg.sh <input_directory>
```

Examples:
```bash
# Process all images in assets/image folder
./triposg.sh assets/image/

# Process custom folder
./triposg.sh /path/to/your/images/
```

**Note**: If you encounter "triposg_batch_threaded.py not found" error, modify the last line of `triposg.sh` script, changing `triposg_batch_threaded.py` to `triposg.py`.

### Method 2: Run Python Script Directly

```bash
python triposg.py <input_directory>
```

## 📊 Processing Pipeline

The script automatically executes the following steps:

1. **Image Segmentation** - Remove background using RMBG model
2. **3D Mesh Generation** - Generate initial 3D mesh using TripoSG
3. **Multi-view Rendering** - Generate rendered images from 6 viewpoints
4. **Texture Mapping** - Generate final model with high-quality 4K textures

## 📈 Output Results

After processing, the following files will be generated in the `output2/` directory:

```
output2/
├── <original_filename>_segmented.png      # Image after background removal
├── <original_filename>_mesh.glb           # Initial 3D mesh model
├── <original_filename>_multiview.png      # Multi-view rendering
└── <original_filename>_textured_4k.glb    # Final textured 4K model
```

### Output File Description:
- `*_segmented.png`: Segmented image after background removal
- `*_mesh.glb`: Basic geometric mesh (~50,000 faces)
- `*_multiview.png`: Preview image from 6 viewpoints
- `*_textured_4k.glb`: Final high-quality textured model (recommended)

## ⚙️ Technical Parameters

- **Default Face Count**: 50,000 faces
- **Texture Resolution**: 4K (4096x4096)
- **Multi-view Count**: 6 viewpoints (front, right, back, left, top, bottom)
- **Inference Steps**: 30 steps (3D generation) / 15 steps (multi-view)
- **Device**: Auto-detect CUDA GPU, fallback to CPU if unavailable

## 💻 System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU recommended, VRAM ≥ 8GB
- **Memory**: ≥ 16GB RAM recommended
- **Storage**: ~50-200MB per model

### Software Requirements
- Python 3.8+
- CUDA 11.0+ (when using GPU)
- Linux/macOS/Windows

## ⚠️ Important Notes

1. **GPU Memory Management**
   - Script includes automatic memory cleanup mechanism
   - Deep cleanup performed every 3 images
   - Automatically switches to batch processing mode if VRAM insufficient

2. **Processing Time**
   - ~5-15 minutes per image (depends on GPU performance)
   - Recommended to test with a few images first

3. **Timeout Control**
   - Timeout limits set for each step (90-120 seconds)
   - Automatically skips and continues to next image on timeout

4. **Supported Image Formats**
   - PNG, JPG, JPEG

## 🛠️ Troubleshooting

### Common Issues

**1. Environment Activation Failed**
```bash
# Check if virtual environment exists
ls env/bin/activate

# Recreate environment
python -m venv env
```

**2. GPU Out of Memory**
```bash
# Set environment variable to limit memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**3. Model Loading Failed**
- Check if `checkpoints/` directory contains required models
- Verify model file integrity

**4. Dependency Installation Failed**
```bash
# Upgrade pip
pip install --upgrade pip

# Install problematic packages separately
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### View Processing Results

```bash
# View all generated GLB files
find ./output2 -name "*.glb"

# Count generated files
find ./output2 -name "*_textured_4k.glb" | wc -l
```

## 📊 Performance Optimization

The script includes the following optimizations:
- ✅ Automatic memory cleanup and garbage collection
- ✅ Thread timeout control
- ✅ Subprocess forced termination mechanism
- ✅ Batch processing mode (when memory insufficient)
- ✅ CUDA memory allocation optimization
- ✅ Progress display and statistics

## 📝 Usage Examples

```bash
# 1. Activate environment
source env/bin/activate

# 2. Process example images
./triposg.sh assets/image/

# 3. View results
ls -lh output2/

# 4. Batch process multiple folders
for dir in folder1 folder2 folder3; do
    ./triposg.sh $dir
done
```

## 🎉 Output Example

After processing, detailed statistics will be displayed:

```
================================================================================
Batch Processing Summary:
================================================================================
Successfully Processed: 1/1
✅ 1
   Segmented Image: output2/1_segmented.png
   Mesh File: output2/1_mesh.glb
   Multi-view Image: output2/1_multiview.png
   Final Model: output2/1_textured_4k.glb

🎉 Batch processing complete! Success rate: 1/1
```

## 📚 Related Resources

- TripoSG Model: [Related paper/repository link]
- RMBG Background Removal: [Related link]
- RealESRGAN Super-resolution: [Related link]

---

**💡 Tip**: It is recommended to test with example images in `assets/image/` first to confirm the environment is configured correctly before processing large batches of data.
