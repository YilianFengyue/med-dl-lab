"""
CNN训练模块
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import *
from dataset import load_data, add_noise
from models import Autoencoder, CNN


def train_cnn(train_loader, test_loader, autoencoder):
    """训练CNN分类器"""
    print("\n" + "=" * 60)
    print("🚀 开始训练CNN分类器")
    print("=" * 60)
    
    # 初始化模型
    model = CNN().to(DEVICE)
    autoencoder.eval()  # 自编码器设为评估模式
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # 记录训练过程
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(CNN_EPOCHS):
        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{CNN_EPOCHS}")
        for data, label in pbar:
            data, label = data.to(DEVICE), label.to(DEVICE)
            
            # 添加噪声并通过自编码器去噪
            noisy_data = add_noise(data)
            with torch.no_grad():
                denoised_data = autoencoder(noisy_data)
            
            optimizer.zero_grad()
            output = model(denoised_data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(label.view_as(pred)).sum().item()
            train_total += label.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.1f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # ==================== 测试阶段 ====================
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                
                # 测试集已有噪声，直接通过自编码器
                denoised_data = autoencoder(data)
                output = model(denoised_data)
                
                test_loss += criterion(output, label).item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(label.view_as(pred)).sum().item()
                test_total += label.size(0)
        
        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)
        
        # 更新学习率
        scheduler.step(test_acc)
        
        # 打印结果
        print(f"[CNN] Epoch {epoch+1:02d}/{CNN_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {avg_test_loss:.4f} Acc: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_DIR / "cnn_best.pth")
            print(f"  💾 保存最佳模型 (acc: {best_acc:.2f}%)")
    
    # 保存最终模型
    torch.save(model.state_dict(), MODEL_DIR / "cnn_final.pth")
    print(f"\n✅ CNN训练完成！最佳准确率: {best_acc:.2f}%")
    
    return model, history


def plot_cnn_history(history):
    """绘制CNN训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss曲线
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, history['test_loss'], 'r-o', label='Test Loss', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Cross Entropy Loss', fontsize=12)
    axes[0].set_title('CNN Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Accuracy曲线
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=4)
    axes[1].plot(epochs, history['test_acc'], 'r-o', label='Test Acc', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('CNN Training Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_ylim([0, 105])
    
    # 标注最佳点
    best_idx = history['test_acc'].index(max(history['test_acc']))
    axes[1].annotate(f'Best: {history["test_acc"][best_idx]:.1f}%',
                    xy=(best_idx + 1, history['test_acc'][best_idx]),
                    xytext=(best_idx + 3, history['test_acc'][best_idx] - 10),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cnn_training_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 图片已保存: {FIGURE_DIR / 'cnn_training_history.png'}")


if __name__ == "__main__":
    print_config()
    
    # 加载数据
    train_loader, test_loader, _, _ = load_data()
    
    # 加载预训练的自编码器
    autoencoder = Autoencoder().to(DEVICE)
    autoencoder.load_state_dict(torch.load(MODEL_DIR / "autoencoder_best.pth"))
    print("✅ 已加载预训练自编码器")
    
    # 训练CNN
    model, history = train_cnn(train_loader, test_loader, autoencoder)
    plot_cnn_history(history)