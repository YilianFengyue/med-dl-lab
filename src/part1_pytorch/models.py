"""
模型定义 - 自编码器 + CNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import IMG_SIZE, NUM_CLASSES


class Autoencoder(nn.Module):
    """
    去噪自编码器
    结构: 1x256x256 → 编码器 → 64x64x64 → 解码器 → 1x256x256
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # 编码器: 压缩图像
        self.encoder = nn.Sequential(
            # 1x256x256 -> 32x256x256 -> 32x128x128
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            # 32x128x128 -> 64x128x128 -> 64x64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        # 解码器: 重建图像
        self.decoder = nn.Sequential(
            # 64x64x64 -> 32x64x64 -> 32x128x128
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            # 32x128x128 -> 32x128x128 -> 32x256x256
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            # 32x256x256 -> 1x256x256
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_architecture(self):
        """返回模型架构描述"""
        return """
        ┌─────────────────────────────────────────────────────┐
        │              Autoencoder Architecture               │
        ├─────────────────────────────────────────────────────┤
        │ Input:  1 × 256 × 256                               │
        ├─────────────────────────────────────────────────────┤
        │ ENCODER:                                            │
        │   Conv2d(1→32, k=3, p=1) + ReLU  → 32 × 256 × 256  │
        │   MaxPool2d(2)                    → 32 × 128 × 128  │
        │   Conv2d(32→64, k=3, p=1) + ReLU → 64 × 128 × 128  │
        │   MaxPool2d(2)                    → 64 × 64 × 64    │
        ├─────────────────────────────────────────────────────┤
        │ DECODER:                                            │
        │   Conv2d(64→32, k=3, p=1) + ReLU → 32 × 64 × 64    │
        │   Upsample(×2)                    → 32 × 128 × 128  │
        │   Conv2d(32→32, k=3, p=1) + ReLU → 32 × 128 × 128  │
        │   Upsample(×2)                    → 32 × 256 × 256  │
        │   Conv2d(32→1, k=3, p=1) + Sigmoid → 1 × 256 × 256 │
        ├─────────────────────────────────────────────────────┤
        │ Output: 1 × 256 × 256                               │
        └─────────────────────────────────────────────────────┘
        """


class CNN(nn.Module):
    """
    图像分类CNN - 增强正则化版本
    添加BatchNorm + 更强Dropout防止过拟合
    """
    def __init__(self):
        super(CNN, self).__init__()
        
        # 特征提取层 (带BatchNorm)
        self.features = nn.Sequential(
            # Block 1: 1x256x256 -> 16x128x128
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 2: 16x128x128 -> 32x64x64
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 3: 32x64x64 -> 64x32x32 (新增层)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        
        # 全局平均池化 (大幅减少参数)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, NUM_CLASSES)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def get_architecture(self):
        """返回模型架构描述"""
        return """
        ┌─────────────────────────────────────────────────────┐
        │           CNN Architecture (Enhanced)               │
        ├─────────────────────────────────────────────────────┤
        │ Input:  1 × 256 × 256                               │
        ├─────────────────────────────────────────────────────┤
        │ FEATURE EXTRACTION:                                 │
        │   Conv2d(1→16) + BN + ReLU       → 16 × 256 × 256  │
        │   MaxPool2d(2) + Dropout2d(0.25) → 16 × 128 × 128  │
        │   Conv2d(16→32) + BN + ReLU      → 32 × 128 × 128  │
        │   MaxPool2d(2) + Dropout2d(0.25) → 32 × 64 × 64    │
        │   Conv2d(32→64) + BN + ReLU      → 64 × 64 × 64    │
        │   MaxPool2d(2) + Dropout2d(0.25) → 64 × 32 × 32    │
        │   AdaptiveAvgPool2d(4,4)         → 64 × 4 × 4      │
        ├─────────────────────────────────────────────────────┤
        │ CLASSIFIER:                                         │
        │   Flatten                         → 1024            │
        │   Linear(1024→128) + BN + ReLU    → 128             │
        │   Dropout(0.5)                                      │
        │   Linear(128→32) + ReLU           → 32              │
        │   Dropout(0.3)                                      │
        │   Linear(32→3)                    → 3               │
        ├─────────────────────────────────────────────────────┤
        │ Output: 3 (Covid, Normal, Viral Pneumonia)          │
        └─────────────────────────────────────────────────────┘
        """


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    ae = Autoencoder()
    cnn = CNN()
    
    print(ae.get_architecture())
    print(f"Autoencoder 参数量: {count_parameters(ae):,}")
    
    print(cnn.get_architecture())
    print(f"CNN 参数量: {count_parameters(cnn):,}")
    
    # 测试前向传播
    x = torch.randn(4, 1, 256, 256)
    print(f"\n输入形状: {x.shape}")
    print(f"Autoencoder输出形状: {ae(x).shape}")
    print(f"CNN输出形状: {cnn(x).shape}")