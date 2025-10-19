#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像质量二分类器
使用ResNet50对图像质量进行0/1分类
0: 数据质量低, 1: 数据质量好
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# ==================== 全局配置 ====================
DEFAULT_GPU = 2        # 默认使用的GPU编号，可以在这里修改
BATCH_SIZE = 256        # 批次大小（深度网络需要更多显存，适当减小batch size）
NUM_EPOCHS = 200        # 训练轮数
DATA_DIR = 'images2'    # 数据目录
# ================================================

class ImageQualityNet(nn.Module):
    """基于ResNet50的图像质量二分类器"""

    def __init__(self, num_classes=2, pretrained=True):
        super(ImageQualityNet, self).__init__()

        # 加载预训练的ResNet50模型
        self.resnet = models.resnet50(pretrained=pretrained)

        # 获取ResNet50最后一层的输入特征数
        num_features = self.resnet.fc.in_features

        # 替换最后的全连接层为自定义分类器
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

class ImageQualityDataset(Dataset):
    """图像质量数据集"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 加载数据
        for label in [0, 1]:
            label_dir = os.path.join(root_dir, str(label))
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(label_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(label)

        print(f"加载数据集: {len(self.images)} 张图片")
        print(f"标签0 (低质量): {self.labels.count(0)} 张")
        print(f"标签1 (高质量): {self.labels.count(1)} 张")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # 加载图片
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class ImageQualityTrainer:
    """图像质量分类器训练器"""

    def __init__(self, model, device, train_loader, val_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 优化器和损失函数
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()

        # 记录训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def test(self):
        """测试模型"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(self, num_epochs=50, save_path='best_model.pth'):
        """训练模型"""
        best_val_acc = 0.0

        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"验证集大小: {len(self.val_loader.dataset)}")
        print("-" * 50)

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate()

            # 更新学习率
            self.scheduler.step()

            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                print(f'保存最佳模型 (Val Acc: {val_acc:.2f}%)')

        print(f'\n训练完成！最佳验证准确率: {best_val_acc:.2f}%')

        # 测试最佳模型
        self.load_model(save_path)
        test_metrics = self.test()
        print(f"\n测试集结果:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1-Score: {test_metrics['f1']:.4f}")

        return test_metrics

    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型: {model_path}")

    def plot_training_history(self, save_path='training_history.png'):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # 准确率曲线
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"训练历史图保存至: {save_path}")

class ImageQualityPredictor:
    """图像质量预测器"""

    def __init__(self, model_path, device=f'cuda:{DEFAULT_GPU}'):
        # 如果指定了具体GPU但CUDA不可用，则使用CPU
        if device.startswith('cuda') and not torch.cuda.is_available():
            print("CUDA不可用，使用CPU")
            self.device = torch.device('cpu')
        elif device.startswith('cuda:'):
            # 提取GPU编号
            gpu_id = int(device.split(':')[1])
            gpu_count = torch.cuda.device_count()

            if gpu_id >= gpu_count:
                print(f"警告: GPU {gpu_id} 不存在，只有 {gpu_count} 个GPU，使用GPU 0")
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device(device)
                print(f"使用 {device}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = ImageQualityNet(num_classes=2)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"模型加载完成，设备: {self.device}")

    def predict(self, image_path):
        """预测单张图片的质量"""
        # 加载和预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # 结果解释
        quality_label = "高质量" if predicted_class == 1 else "低质量"

        return {
            'predicted_class': predicted_class,
            'quality_label': quality_label,
            'confidence': confidence,
            'probabilities': {
                'low_quality': probabilities[0][0].item(),
                'high_quality': probabilities[0][1].item()
            }
        }

    def predict_batch(self, image_paths):
        """批量预测图片质量"""
        results = []
        for image_path in tqdm(image_paths, desc="预测中"):
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"预测失败 {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })

        return results

def create_data_loaders(data_dir, batch_size=64, num_workers=4):
    """创建数据加载器"""

    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 创建完整数据集
    full_dataset = ImageQualityDataset(data_dir, transform=None)

    # 按8:1:1分割数据集
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 为不同数据集应用不同的变换
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"数据集分割:")
    print(f"训练集: {len(train_dataset)} 张图片")
    print(f"验证集: {len(val_dataset)} 张图片")
    print(f"测试集: {len(test_dataset)} 张图片")
    print(f"批次大小: {batch_size}")

    return train_loader, val_loader, test_loader

def main():
    """主函数 - 训练模型"""
    # 使用全局配置参数
    data_dir = DATA_DIR
    batch_size = BATCH_SIZE
    num_epochs = NUM_EPOCHS
    gpu_id = DEFAULT_GPU

    # 检查CUDA可用性和设置GPU
    if not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = torch.device('cpu')
    else:
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个GPU")

        # 显示所有可用GPU信息
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # 验证GPU 2是否存在
        if gpu_id >= gpu_count:
            print(f"错误: GPU {gpu_id} 不存在，只有 {gpu_count} 个GPU (0-{gpu_count-1})")
            print(f"改用GPU 0")
            gpu_id = 0

        # 设置使用指定的GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')

        print(f"\n固定使用GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        print(f"显存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir, batch_size=batch_size, num_workers=8
    )

    # 创建模型
    model = ImageQualityNet(num_classes=2)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建训练器
    trainer = ImageQualityTrainer(model, device, train_loader, val_loader, test_loader)

    # 训练模型
    test_metrics = trainer.train(num_epochs=num_epochs, save_path='best_image_quality_model2.pth')

    # 绘制训练历史
    trainer.plot_training_history('training_history.png')

    # 保存训练结果
    results = {
        'test_metrics': test_metrics,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'train_accs': trainer.train_accs,
        'val_accs': trainer.val_accs
    }

    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("训练完成！模型和结果已保存。")

if __name__ == '__main__':
    main()
