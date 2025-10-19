# TripoSG 3D模型生成工具

## 🎯 功能说明

这是一个基于 TripoSG 的 3D 模型生成工具，可以从单张 2D 图片生成高质量的 3D GLB 模型文件。该工具集成了图像分割、3D 网格生成、多视角渲染和纹理映射等完整流程。

## 📁 项目结构

```
3dGenerate/
├── triposg.py              # 主处理脚本（批量处理）
├── triposg.sh              # 启动脚本
├── requirements.txt        # Python依赖包列表
├── assets/
│   └── image/
│       └── 1.png          # 示例图片
├── output2/                # 输出目录（自动创建）
└── README.md              # 本文档
```

## 🔧 环境配置

### 1. 创建虚拟环境

```bash
# 创建Python虚拟环境
python -m venv env

# 激活环境
source env/bin/activate  # Linux/macOS
# 或
env\Scripts\activate     # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖包：
- **3D处理**: trimesh, pyrender, open3d, pymeshlab
- **深度学习**: torch, torchvision, transformers, diffusers
- **图像处理**: opencv-python, Pillow, scikit-image
- **其他**: CLIP, LPIPS, tqdm 等

### 3. 下载预训练模型

需要下载以下模型到 `checkpoints/` 目录：
- `RMBG-1.4` - 背景移除模型
- `TripoSG` - 3D生成模型
- `RealESRGAN_x2plus.pth` - 纹理超分辨率模型
- `big-lama.pt` - 纹理修复模型

## 🚀 使用方法

### 方式一：使用 Shell 脚本（推荐）

```bash
./triposg.sh <输入目录>
```

示例：
```bash
# 处理 assets/image 文件夹中的所有图片
./triposg.sh assets/image/

# 处理自定义文件夹
./triposg.sh /path/to/your/images/
```

**注意**: 如果运行时提示找不到 `triposg_batch_threaded.py`，请修改 `triposg.sh` 脚本的最后一行，将 `triposg_batch_threaded.py` 改为 `triposg.py`。

### 方式二：直接运行 Python 脚本

```bash
python triposg.py <输入目录>
```

## 📊 处理流程

脚本会自动执行以下步骤：

1. **图像分割** - 使用 RMBG 模型移除背景
2. **3D网格生成** - 使用 TripoSG 生成初始 3D 网格
3. **多视角渲染** - 生成 6 个视角的渲染图像
4. **纹理映射** - 生成高质量 4K 纹理的最终模型

## 📈 输出结果

处理完成后，在 `output2/` 目录下会生成以下文件：

```
output2/
├── <原文件名>_segmented.png      # 背景移除后的图像
├── <原文件名>_mesh.glb           # 初始3D网格模型
├── <原文件名>_multiview.png      # 多视角渲染图
└── <原文件名>_textured_4k.glb    # 最终带纹理的4K模型
```

### 输出文件说明：
- `*_segmented.png`: 去除背景后的分割图像
- `*_mesh.glb`: 基础几何网格（约50,000面）
- `*_multiview.png`: 6个视角的渲染预览图
- `*_textured_4k.glb`: 最终高质量纹理模型（推荐使用）

## ⚙️ 技术参数

- **默认面数**: 50,000 faces
- **纹理分辨率**: 4K (4096x4096)
- **多视角数量**: 6 个视角（前、右、后、左、顶、底）
- **推理步数**: 30 步（3D生成）/ 15 步（多视角）
- **设备**: 自动检测 CUDA GPU，无 GPU 则使用 CPU

## 💻 系统要求

### 硬件要求
- **GPU**: 建议 NVIDIA GPU，显存 ≥ 8GB
- **内存**: 建议 ≥ 16GB RAM
- **存储**: 每个模型约 50-200MB

### 软件要求
- Python 3.8+
- CUDA 11.0+ (使用 GPU 时)
- Linux/macOS/Windows

## ⚠️ 注意事项

1. **GPU 内存管理**
   - 脚本包含自动内存清理机制
   - 每处理 3 张图片会执行深度清理
   - 如遇显存不足，会自动切换到分批处理模式

2. **处理时间**
   - 单张图片约需 5-15 分钟（取决于 GPU 性能）
   - 建议先用少量图片测试

3. **超时控制**
   - 各步骤设有超时限制（90-120秒）
   - 超时会自动跳过并继续处理下一张

4. **支持的图片格式**
   - PNG, JPG, JPEG

## 🛠️ 故障排除

### 常见问题

**1. 环境激活失败**
```bash
# 检查虚拟环境是否存在
ls env/bin/activate

# 重新创建环境
python -m venv env
```

**2. GPU 内存不足**
```bash
# 设置环境变量限制内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**3. 模型加载失败**
- 检查 `checkpoints/` 目录是否包含所需模型
- 确认模型文件完整性

**4. 依赖包安装失败**
```bash
# 升级 pip
pip install --upgrade pip

# 单独安装问题包
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 查看处理结果

```bash
# 查看生成的所有 GLB 文件
find ./output2 -name "*.glb"

# 统计生成的文件数量
find ./output2 -name "*_textured_4k.glb" | wc -l
```

## 📊 性能优化

脚本已包含以下优化：
- ✅ 自动内存清理和垃圾回收
- ✅ 线程超时控制
- ✅ 子进程强制终止机制
- ✅ 分批处理模式（内存不足时）
- ✅ CUDA 内存分配优化
- ✅ 进度显示和统计信息

## 📝 使用示例

```bash
# 1. 激活环境
source env/bin/activate

# 2. 处理示例图片
./triposg.sh assets/image/

# 3. 查看结果
ls -lh output2/

# 4. 批量处理多个文件夹
for dir in folder1 folder2 folder3; do
    ./triposg.sh $dir
done
```

## 🎉 输出示例

处理完成后会显示详细统计：

```
================================================================================
批处理完成总结:
================================================================================
成功处理: 1/1
✅ 1
   分割图像: output2/1_segmented.png
   网格文件: output2/1_mesh.glb
   多视角图像: output2/1_multiview.png
   最终模型: output2/1_textured_4k.glb

🎉 批处理完成! 成功率: 1/1
```

## 📚 相关资源

- TripoSG 模型: [相关论文/仓库链接]
- RMBG 背景移除: [相关链接]
- RealESRGAN 超分辨率: [相关链接]

---

**💡 提示**: 建议先用 `assets/image/` 中的示例图片测试，确认环境配置正确后再处理大批量数据。
