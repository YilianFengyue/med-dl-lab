"""LSTM模型定义"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, 
    Bidirectional, SpatialDropout1D, GlobalMaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from config import *


def build_lstm_model(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, 
                     lstm_units=LSTM_UNITS, num_classes=NUM_CLASSES):
    """构建增强版LSTM模型"""
    model = Sequential([
        # 嵌入层
        Embedding(vocab_size, embed_dim, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        
        # 双向LSTM
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(DROPOUT_RATE),
        
        # 第二层LSTM
        Bidirectional(LSTM(lstm_units // 2)),
        Dropout(DROPOUT_RATE),
        
        # 全连接层
        Dense(64, activation='relu'),
        Dropout(DROPOUT_RATE),
        
        # 输出层
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_simple_lstm(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                      lstm_units=LSTM_UNITS, num_classes=NUM_CLASSES):
    """构建基础LSTM模型（实验指导书版本）"""
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=MAX_LEN),
        LSTM(lstm_units),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_path):
    """获取训练回调"""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks