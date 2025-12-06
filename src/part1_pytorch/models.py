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
    图像分类CNN
    结构: 1x256x256 → 卷积层 → 全连接层 → 3类输出
    """
    def __init__(self):
        super(CNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 全连接层
        # 经过2次池化: 256 -> 128 -> 64, 通道数32
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, NUM_CLASSES)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))  # 1x256x256 -> 16x128x128
        x = self.pool(F.relu(self.conv2(x)))  # 16x128x128 -> 32x64x64
        
        # 展平
        x = x.view(-1, 32 * 64 * 64)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 不加softmax，CrossEntropyLoss会处理
        
        return x
    
    def get_architecture(self):
        """返回模型架构描述"""
        return """
        ┌─────────────────────────────────────────────────────┐
        │                  CNN Architecture                   │
        ├─────────────────────────────────────────────────────┤
        │ Input:  1 × 256 × 256                               │
        ├─────────────────────────────────────────────────────┤
        │ FEATURE EXTRACTION:                                 │
        │   Conv2d(1→16, k=3, p=1) + ReLU  → 16 × 256 × 256  │
        │   MaxPool2d(2)                    → 16 × 128 × 128  │
        │   Conv2d(16→32, k=3, p=1) + ReLU → 32 × 128 × 128  │
        │   MaxPool2d(2)                    → 32 × 64 × 64    │
        ├─────────────────────────────────────────────────────┤
        │ CLASSIFIER:                                         │
        │   Flatten                         → 131072          │
        │   Linear(131072→128) + ReLU       → 128             │
        │   Dropout(0.5)                                      │
        │   Linear(128→32) + ReLU           → 32              │
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