"""数据加载与预处理"""
import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from config import *


def clean_text(text):
    """文本清洗"""
    if not isinstance(text, str):
        return ""
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除特殊字符，保留字母数字空格
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # 转小写，去除多余空格
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def rating_to_label(rating):
    """评分转标签: 1-4消极(0), 5-6中性(1), 7-10积极(2)"""
    if rating <= 4:
        return 0  # 消极
    elif rating <= 6:
        return 1  # 中性
    else:
        return 2  # 积极


def load_data():
    """加载并预处理数据"""
    print("[INFO] Loading data...")
    
    # 读取CSV
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # 提取评论和评分
    train_reviews = train_df['review'].apply(clean_text).values
    test_reviews = test_df['review'].apply(clean_text).values
    
    train_labels = train_df['rating'].apply(rating_to_label).values
    test_labels = test_df['rating'].apply(rating_to_label).values
    
    # 打印标签分布
    print("\n[INFO] Label distribution:")
    for split, labels in [("Train", train_labels), ("Test", test_labels)]:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  {split}: " + ", ".join([f"Class {u}: {c}" for u, c in zip(unique, counts)]))
    
    return train_reviews, train_labels, test_reviews, test_labels


def create_tokenizer(train_texts):
    """创建并拟合分词器"""
    print("\n[INFO] Creating tokenizer...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)
    print(f"  Vocabulary size: {min(len(tokenizer.word_index), VOCAB_SIZE)}")
    return tokenizer


def prepare_data(train_reviews, train_labels, test_reviews, test_labels, tokenizer):
    """准备训练和测试数据"""
    print("\n[INFO] Preparing sequences...")
    
    # 文本序列化
    train_seq = tokenizer.texts_to_sequences(train_reviews)
    test_seq = tokenizer.texts_to_sequences(test_reviews)
    
    # 填充序列
    X_train = pad_sequences(train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test = pad_sequences(test_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 标签独热编码
    y_train = to_categorical(train_labels, num_classes=NUM_CLASSES)
    y_test = to_categorical(test_labels, num_classes=NUM_CLASSES)
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


def get_class_weights(labels):
    """计算类别权重（处理不平衡）"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = {i: total / (len(unique) * c) for i, c in zip(unique, counts)}
    print(f"\n[INFO] Class weights: {weights}")
    return weights