"""
端到端联合训练模块 (V3核心)
解冻Autoencoder，让分类Loss反向传播到AE，保留对分类有用的纹理特征
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import *
from dataset import add_noise
from models import Autoencoder, CNN


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA, mode='max'):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.counter, self.best_score, self.early_stop, self.best_epoch = 0, None, False, 0
    
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score, self.best_epoch = score, epoch
            return False
        improved = score > self.best_score + self.min_delta if self.mode == 'max' else score < self.best_score - self.min_delta
        if improved:
            self.best_score, self.best_epoch, self.counter = score, epoch, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⏹️  Early Stopping! Best Epoch: {self.best_epoch+1}, Acc: {self.best_score:.2f}%")
        return self.early_stop


class EndToEndModel(nn.Module):
    """端到端模型：AE + CNN"""
    def __init__(self, ae, cnn):
        super().__init__()
        self.ae, self.cnn = ae, cnn
    
    def forward(self, x, return_denoised=False):
        denoised = self.ae(x)
        logits = self.cnn(denoised)
        return (logits, denoised) if return_denoised else logits


def train_e2e(train_loader, test_loader, autoencoder, cnn, class_weights):
    """端到端联合训练"""
    print("\n" + "=" * 60)
    print("🔗 端到端联合训练 (V3)")
    print("=" * 60)
    print("💡 AE和CNN同时训练，分类loss会反向传播到AE")
    
    model = EndToEndModel(autoencoder, cnn).to(DEVICE)
    
    # 分层学习率
    optimizer = torch.optim.AdamW([
        {'params': model.ae.parameters(), 'lr': E2E_LR_AE, 'weight_decay': 1e-5},
        {'params': model.cnn.parameters(), 'lr': E2E_LR_CNN, 'weight_decay': 1e-4}
    ])
    
    class_weights = class_weights.to(DEVICE)
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_rec = nn.MSELoss()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'rec_loss': []}
    best_acc = 0.0
    
    for epoch in range(E2E_EPOCHS):
        # ========== 训练 ==========
        model.train()
        train_loss, train_correct, train_total, rec_loss_sum = 0, 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"E2E {epoch+1:02d}/{E2E_EPOCHS}")
        for data, label in pbar:
            data, label = data.to(DEVICE), label.to(DEVICE)
            clean_data = data.clone()
            noisy_data = add_noise(data)
            
            optimizer.zero_grad()
            logits, denoised = model(noisy_data, return_denoised=True)
            
            # 联合损失: 分类(主) + 重建(辅)
            loss_cls = criterion_cls(logits, label)
            loss_rec = criterion_rec(denoised, clean_data)
            loss = loss_cls + E2E_AE_WEIGHT * loss_rec
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss_cls.item()
            rec_loss_sum += loss_rec.item()
            pred = logits.argmax(dim=1)
            train_correct += pred.eq(label).sum().item()
            train_total += label.size(0)
            
            pbar.set_postfix({'cls': f'{loss_cls.item():.3f}', 'rec': f'{loss_rec.item():.4f}', 'acc': f'{100.*train_correct/train_total:.1f}%'})
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        avg_rec_loss = rec_loss_sum / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['rec_loss'].append(avg_rec_loss)
        
        # ========== 测试 ==========
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(DEVICE), label.to(DEVICE)
                logits = model(data)  # 测试集已有噪声
                test_loss += criterion_cls(logits, label).item()
                test_correct += logits.argmax(dim=1).eq(label).sum().item()
                test_total += label.size(0)
        
        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)
        
        lr_ae = optimizer.param_groups[0]['lr']
        lr_cnn = optimizer.param_groups[1]['lr']
        print(f"[E2E] Epoch {epoch+1:02d} | Train: {avg_train_loss:.3f} {train_acc:.1f}% | "
              f"Test: {avg_test_loss:.3f} {test_acc:.1f}% | Rec: {avg_rec_loss:.4f} | "
              f"LR: ae={lr_ae:.6f} cnn={lr_cnn:.5f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'ae': model.ae.state_dict(), 'cnn': model.cnn.state_dict()}, MODEL_DIR / "e2e_best.pth")
            print(f"  💾 保存最佳模型 (acc: {best_acc:.2f}%)")
        
        if early_stopping(test_acc, epoch):
            break
    
    # 加载最佳
    ckpt = torch.load(MODEL_DIR / "e2e_best.pth", weights_only=True)
    model.ae.load_state_dict(ckpt['ae'])
    model.cnn.load_state_dict(ckpt['cnn'])
    print(f"\n✅ 端到端训练完成！最佳准确率: {best_acc:.2f}% (Epoch {early_stopping.best_epoch+1})")
    
    return model.ae, model.cnn, history


def plot_e2e_history(history):
    """绘制端到端训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train', markersize=3)
    axes[0].plot(epochs, history['test_loss'], 'r-o', label='Test', markersize=3)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Classification Loss')
    axes[0].set_title('E2E Training Loss', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train', markersize=3)
    axes[1].plot(epochs, history['test_acc'], 'r-o', label='Test', markersize=3)
    best_idx = history['test_acc'].index(max(history['test_acc']))
    axes[1].axvline(x=best_idx+1, color='g', linestyle='--', alpha=0.7)
    axes[1].annotate(f'Best: {history["test_acc"][best_idx]:.1f}%', xy=(best_idx+1, history['test_acc'][best_idx]),
                    xytext=(best_idx+3, history['test_acc'][best_idx]-8), arrowprops=dict(arrowstyle='->', color='r'), color='r')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('E2E Training Accuracy', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.7); axes[1].set_ylim([30, 100])
    
    # Reconstruction Loss
    axes[2].plot(epochs, history['rec_loss'], 'g-o', markersize=3)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('MSE Loss')
    axes[2].set_title('Reconstruction Loss (AE)', fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "e2e_training_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 图片已保存: {FIGURE_DIR / 'e2e_training_history.png'}")