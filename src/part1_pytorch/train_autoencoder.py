"""
自编码器训练模块
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import *
from dataset import load_data, add_noise
from models import Autoencoder


def train_autoencoder(train_loader, test_loader):
    """训练自编码器"""
    print("\n" + "=" * 60)
    print("🚀 开始训练自编码器 (Denoising Autoencoder)")
    print("=" * 60)
    
    # 初始化模型
    model = Autoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 记录训练过程
    history = {
        'train_loss': [],
        'test_loss': []
    }
    
    best_loss = float('inf')
    
    for epoch in range(AE_EPOCHS):
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{AE_EPOCHS}")
        for img, _ in pbar:
            img = img.to(DEVICE)
            noisy_img = add_noise(img)
            
            optimizer.zero_grad()
            reconstructed = model(noisy_img)
            loss = criterion(reconstructed, img)  # 与原图比较
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # ==================== 测试阶段 ====================
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for img, _ in test_loader:
                img = img.to(DEVICE)
                # 测试集已经是噪声图片，直接输入
                reconstructed = model(img)
                loss = criterion(reconstructed, img)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        history['test_loss'].append(avg_test_loss)
        
        # 更新学习率
        scheduler.step(avg_test_loss)
        
        # 打印epoch结果
        print(f"[Autoencoder] Epoch {epoch+1:02d}/{AE_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f}")
        
        # 保存最佳模型
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), MODEL_DIR / "autoencoder_best.pth")
            print(f"  💾 保存最佳模型 (loss: {best_loss:.4f})")
    
    # 保存最终模型
    torch.save(model.state_dict(), MODEL_DIR / "autoencoder_final.pth")
    print(f"\n✅ 自编码器训练完成！最佳Loss: {best_loss:.4f}")
    
    return model, history


def visualize_denoising(model, test_loader, num_samples=6):
    """可视化去噪效果"""
    print("\n🎨 生成去噪效果对比图...")
    
    model.eval()
    
    # 获取测试数据
    for data, labels in test_loader:
        data = data.to(DEVICE)
        break
    
    with torch.no_grad():
        reconstructed = model(data)
    
    # 绘图
    fig, axes = plt.subplots(3, num_samples, figsize=(2.5 * num_samples, 8))
    
    for i in range(num_samples):
        # 噪声图片（输入）
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'{CLASS_NAMES[labels[i]]}', fontsize=10)
        
        # 去噪图片（输出）
        axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        
        # 差异图
        diff = torch.abs(data[i] - reconstructed[i])
        axes[2, i].imshow(diff.cpu().squeeze(), cmap='hot')
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Noisy Input', fontsize=12)
    axes[1, 0].set_ylabel('Denoised', fontsize=12)
    axes[2, 0].set_ylabel('Difference', fontsize=12)
    
    plt.suptitle('Autoencoder Denoising Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "autoencoder_denoising.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 图片已保存: {FIGURE_DIR / 'autoencoder_denoising.png'}")


def plot_ae_loss(history):
    """绘制自编码器损失曲线"""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    plt.plot(epochs, history['test_loss'], 'r-o', label='Test Loss', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Autoencoder Training Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 标注最小loss点
    min_idx = history['test_loss'].index(min(history['test_loss']))
    plt.annotate(f'Best: {history["test_loss"][min_idx]:.4f}',
                xy=(min_idx + 1, history['test_loss'][min_idx]),
                xytext=(min_idx + 5, history['test_loss'][min_idx] + 0.002),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "autoencoder_loss.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 图片已保存: {FIGURE_DIR / 'autoencoder_loss.png'}")


if __name__ == "__main__":
    print_config()
    train_loader, test_loader, _, _ = load_data()
    model, history = train_autoencoder(train_loader, test_loader)
    plot_ae_loss(history)
    visualize_denoising(model, test_loader)