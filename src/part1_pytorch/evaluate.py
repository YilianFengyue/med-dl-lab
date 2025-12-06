"""
模型评估与可视化模块
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from config import *
from dataset import load_data
from models import Autoencoder, CNN


def evaluate_model(cnn, autoencoder, test_loader):
    """完整评估模型"""
    print("\n" + "=" * 60)
    print("📊 模型评估")
    print("=" * 60)
    
    cnn.eval()
    autoencoder.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            
            # 通过自编码器去噪
            denoised = autoencoder(data)
            output = cnn(denoised)
            
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算各项指标
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    # 打印分类报告
    print("\n📋 Classification Report:")
    print("-" * 60)
    report = classification_report(all_labels, all_preds, 
                                   target_names=CLASS_NAMES, digits=4)
    print(report)
    
    # 打印总体指标
    print("-" * 60)
    print(f"🎯 Overall Accuracy: {accuracy:.2f}%")
    print(f"📈 Macro F1-Score: {np.mean(f1):.4f}")
    print(f"📈 Weighted F1-Score: {precision_recall_fscore_support(all_labels, all_preds, average='weighted')[2]:.4f}")
    
    return all_preds, all_labels, all_probs


def plot_confusion_matrix(y_true, y_pred):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={'size': 16})
    
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    
    # 添加准确率信息
    accuracy = np.trace(cm) / np.sum(cm) * 100
    plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.2f}%', 
                ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 图片已保存: {FIGURE_DIR / 'confusion_matrix.png'}")


def plot_normalized_confusion_matrix(y_true, y_pred):
    """绘制归一化混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={'size': 14}, vmin=0, vmax=1)
    
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Normalized Confusion Matrix (Recall per Class)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix_normalized.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 图片已保存: {FIGURE_DIR / 'confusion_matrix_normalized.png'}")


def plot_roc_curves(y_true, y_probs):
    """绘制ROC曲线 (多分类 One-vs-Rest)"""
    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_true_bin.shape[1]
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    # 计算每个类别的ROC曲线
    for i, (color, class_name) in enumerate(zip(colors, CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # 对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 图片已保存: {FIGURE_DIR / 'roc_curves.png'}")


def plot_precision_recall_curves(y_true, y_probs):
    """绘制Precision-Recall曲线"""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    for i, (color, class_name) in enumerate(zip(colors, CLASS_NAMES)):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                label=f'{class_name} (AP = {ap:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "precision_recall_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 图片已保存: {FIGURE_DIR / 'precision_recall_curves.png'}")


def plot_per_class_metrics(y_true, y_pred):
    """绘制每类别的指标柱状图"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Per-Class Classification Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "per_class_metrics.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 图片已保存: {FIGURE_DIR / 'per_class_metrics.png'}")


def plot_sample_predictions(cnn, autoencoder, test_loader, num_samples=12):
    """可视化预测样本"""
    cnn.eval()
    autoencoder.eval()
    
    for data, labels in test_loader:
        data = data.to(DEVICE)
        labels = labels.numpy()
        break
    
    with torch.no_grad():
        denoised = autoencoder(data)
        output = cnn(denoised)
        preds = output.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(output, dim=1).cpu().numpy()
    
    # 绘图
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 4 * rows))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(data))):
        ax = axes[i]
        ax.imshow(denoised[i].cpu().squeeze(), cmap='gray')
        
        true_label = CLASS_NAMES[labels[i]]
        pred_label = CLASS_NAMES[preds[i]]
        confidence = probs[i][preds[i]] * 100
        
        color = 'green' if labels[i] == preds[i] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                    color=color, fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sample_predictions.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 图片已保存: {FIGURE_DIR / 'sample_predictions.png'}")


def generate_full_report(cnn, autoencoder, test_loader):
    """生成完整评估报告"""
    print("\n" + "=" * 60)
    print("📑 生成完整评估报告")
    print("=" * 60)
    
    # 1. 评估模型
    y_pred, y_true, y_probs = evaluate_model(cnn, autoencoder, test_loader)
    
    # 2. 混淆矩阵
    print("\n🔲 绘制混淆矩阵...")
    plot_confusion_matrix(y_true, y_pred)
    plot_normalized_confusion_matrix(y_true, y_pred)
    
    # 3. ROC曲线
    print("\n📈 绘制ROC曲线...")
    plot_roc_curves(y_true, y_probs)
    
    # 4. PR曲线
    print("\n📉 绘制Precision-Recall曲线...")
    plot_precision_recall_curves(y_true, y_probs)
    
    # 5. 每类指标
    print("\n📊 绘制每类指标...")
    plot_per_class_metrics(y_true, y_pred)
    
    # 6. 预测样本
    print("\n🖼️ 绘制预测样本...")
    plot_sample_predictions(cnn, autoencoder, test_loader)
    
    print("\n" + "=" * 60)
    print("✅ 评估报告生成完成！")
    print(f"📁 所有图片保存在: {FIGURE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
    
    # 加载数据
    _, test_loader, _, _ = load_data()
    
    # 加载模型
    autoencoder = Autoencoder().to(DEVICE)
    autoencoder.load_state_dict(torch.load(MODEL_DIR / "autoencoder_best.pth"))
    
    cnn = CNN().to(DEVICE)
    cnn.load_state_dict(torch.load(MODEL_DIR / "cnn_best.pth"))
    
    print("✅ 模型加载完成")
    
    # 生成完整报告
    generate_full_report(cnn, autoencoder, test_loader)