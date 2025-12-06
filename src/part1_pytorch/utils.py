"""
工具函数模块
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import *


def set_seed(seed=42):
    """设置随机种子，确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 随机种子设置为: {seed}")


def get_device_info():
    """获取设备信息"""
    print("\n🖥️ 设备信息:")
    print("-" * 40)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   CUDA版本: {torch.version.cuda}")
    else:
        print("⚠️ CUDA 不可用，使用CPU训练")
    
    print(f"   PyTorch版本: {torch.__version__}")
    print("-" * 40)


def plot_data_samples(train_loader, num_samples=9):
    """可视化训练数据样本"""
    # 获取一批数据
    for data, labels in train_loader:
        break
    
    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(data))):
        ax = axes[i]
        ax.imshow(data[i].squeeze(), cmap='gray')
        ax.set_title(f'{CLASS_NAMES[labels[i]]}', fontsize=12)
        ax.axis('off')
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Training Data Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "training_samples.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 图片已保存: {FIGURE_DIR / 'training_samples.png'}")


def plot_noisy_comparison(train_loader, noise_factor=NOISE_FACTOR):
    """对比原图和加噪图"""
    from dataset import add_noise
    
    for data, labels in train_loader:
        break
    
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    
    for i in range(6):
        # 原图
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f'{CLASS_NAMES[labels[i]]}', fontsize=10)
        axes[0, i].axis('off')
        
        # 加噪图
        noisy = add_noise(data[i:i+1], noise_factor)
        axes[1, i].imshow(noisy.squeeze(), cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Noisy', fontsize=12)
    
    plt.suptitle(f'Original vs Noisy Images (noise_factor={noise_factor})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "noise_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 图片已保存: {FIGURE_DIR / 'noise_comparison.png'}")


def save_model_summary(autoencoder, cnn):
    """保存模型摘要到文件"""
    from models import count_parameters
    
    summary_path = OUTPUT_DIR / "model_summary.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型架构摘要\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("【自编码器】\n")
        f.write(autoencoder.get_architecture())
        f.write(f"\n参数量: {count_parameters(autoencoder):,}\n")
        
        f.write("\n" + "=" * 60 + "\n\n")
        
        f.write("【CNN分类器】\n")
        f.write(cnn.get_architecture())
        f.write(f"\n参数量: {count_parameters(cnn):,}\n")
    
    print(f"✅ 模型摘要已保存: {summary_path}")


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


if __name__ == "__main__":
    set_seed(42)
    get_device_info()
    
    from dataset import load_data
    train_loader, _, _, _, _ = load_data()
    
    plot_data_samples(train_loader)
    plot_noisy_comparison(train_loader)