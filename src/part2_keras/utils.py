"""绘图与评估工具"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
import seaborn as sns
from config import OUTPUT_DIR

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CLASS_NAMES = ['Negative', 'Neutral', 'Positive']


def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # 准确率
    ax = axes[0]
    ax.plot(epochs, history.history['accuracy'], 'b-o', label='Training')
    ax.plot(epochs, history.history['val_accuracy'], 'r-o', label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 损失
    ax = axes[1]
    ax.plot(epochs, history.history['loss'], 'b-o', label='Training')
    ax.plot(epochs, history.history['val_loss'], 'r-o', label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 原始混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix')
    
    # 归一化混淆矩阵
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Normalized Confusion Matrix')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.show()


def plot_roc_curves(y_true_onehot, y_pred_proba, save_path=None):
    """绘制ROC曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.show()


def plot_precision_recall_curves(y_true_onehot, y_pred_proba, save_path=None):
    """绘制PR曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{name} (AUC = {pr_auc:.3f})')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.show()


def plot_per_class_metrics(report_dict, save_path=None):
    """绘制各类别指标对比"""
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, metric in enumerate(metrics):
        values = [report_dict[name][metric] for name in CLASS_NAMES]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize())
        ax.bar_label(bars, fmt='%.3f', fontsize=8)
    
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
    plt.show()


def evaluate_model(model, X_test, y_test, y_test_labels):
    """完整评估模型"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # 预测
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 分类报告
    print("\n[Classification Report]")
    report = classification_report(y_test_labels, y_pred, 
                                   target_names=CLASS_NAMES, digits=4)
    print(report)
    
    report_dict = classification_report(y_test_labels, y_pred,
                                        target_names=CLASS_NAMES, 
                                        output_dict=True)
    
    # 总体指标
    print("\n[Overall Metrics]")
    print(f"  Accuracy:  {report_dict['accuracy']:.4f}")
    print(f"  Macro F1:  {report_dict['macro avg']['f1-score']:.4f}")
    print(f"  Weighted F1: {report_dict['weighted avg']['f1-score']:.4f}")
    
    # 绘图
    fig_dir = os.path.join(OUTPUT_DIR, 'figures')
    
    plot_confusion_matrix(
        y_test_labels, y_pred,
        os.path.join(fig_dir, 'confusion_matrix.png')
    )
    
    plot_roc_curves(
        y_test, y_pred_proba,
        os.path.join(fig_dir, 'roc_curves.png')
    )
    
    plot_precision_recall_curves(
        y_test, y_pred_proba,
        os.path.join(fig_dir, 'pr_curves.png')
    )
    
    plot_per_class_metrics(
        report_dict,
        os.path.join(fig_dir, 'per_class_metrics.png')
    )
    
    return report_dict