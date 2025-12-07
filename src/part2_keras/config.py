"""配置参数"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', '数据集', 'review')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs_lstm')

# 数据配置
TRAIN_FILE = os.path.join(DATA_DIR, 'drugsComTrain_raw.csv')
TEST_FILE = os.path.join(DATA_DIR, 'drugsComTest_raw.csv')

# 模型超参数
VOCAB_SIZE = 20000      # 词汇表大小
MAX_LEN = 200           # 序列最大长度
EMBED_DIM = 128         # 嵌入维度
LSTM_UNITS = 128        # LSTM单元数
DROPOUT_RATE = 0.3      # Dropout率
NUM_CLASSES = 3         # 分类数

# 训练超参数
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# 创建输出目录
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)