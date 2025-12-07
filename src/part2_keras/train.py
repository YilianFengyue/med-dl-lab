"""LSTM情感分析训练主程序"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from config import *
from dataset import load_data, create_tokenizer, prepare_data, get_class_weights
from models import build_lstm_model, build_simple_lstm, get_callbacks
from utils import plot_training_history, evaluate_model


def main():
    print("="*60)
    print("Drug Review Sentiment Analysis with LSTM")
    print("="*60)
    
    # 检查设备
    print(f"\n[INFO] TensorFlow version: {tf.__version__}")
    print(f"[INFO] Running on: CPU")
    
    # 加载数据
    train_reviews, train_labels, test_reviews, test_labels = load_data()
    
    # 创建分词器
    tokenizer = create_tokenizer(train_reviews)
    
    # 准备数据
    X_train, y_train, X_test, y_test = prepare_data(
        train_reviews, train_labels, test_reviews, test_labels, tokenizer
    )
    
    # 计算类别权重
    class_weights = get_class_weights(train_labels)
    
    # 构建模型
    print("\n[INFO] Building model...")
    model = build_lstm_model()
    model.summary()
    
    # 训练回调
    model_path = os.path.join(OUTPUT_DIR, 'models', 'best_lstm.keras')
    callbacks = get_callbacks(model_path)
    
    # 训练
    print("\n[INFO] Training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # 绘制训练历史
    plot_training_history(
        history,
        os.path.join(OUTPUT_DIR, 'figures', 'training_history.png')
    )
    
    # 评估模型
    evaluate_model(model, X_test, y_test, test_labels)
    
    # 最终测试
    print("\n[INFO] Final test evaluation:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Model saved to: {model_path}")
    print(f"Figures saved to: {os.path.join(OUTPUT_DIR, 'figures')}")
    print("="*60)


if __name__ == '__main__':
    main()