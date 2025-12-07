"""
配置文件 - 超参数和路径设置
"""
import torch
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent.parent.parent  # Exp6/
DATA_DIR = BASE_DIR / "data" / "数据集" / "covid19"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "noisy_test"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

# 创建输出目录
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 模型超参数 ====================
# 图像参数
IMG_SIZE = 256
IMG_CHANNELS = 1  # 灰度图

# 自编码器参数
AE_EPOCHS = 30
AE_BATCH_SIZE = 16
AE_LR = 0.0005
NOISE_FACTOR = 0.3  # 降低噪声，保留更多纹理

# CNN参数
CNN_EPOCHS = 60
CNN_BATCH_SIZE = 16
CNN_LR = 0.0005
NUM_CLASSES = 3

# 端到端联合训练参数 (V3核心)
E2E_EPOCHS = 50
E2E_LR_AE = 0.00005   # AE用更小的学习率微调
E2E_LR_CNN = 0.0003   # CNN正常学习率
E2E_AE_WEIGHT = 0.1   # 重建loss的权重(辅助)

# Early Stopping
PATIENCE = 20  # 增加耐心
MIN_DELTA = 0.3  # 降低阈值

# 数据增强强度
AUGMENT_LEVEL = 'strong'

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别名称
CLASS_NAMES = ['Covid', 'Normal', 'Viral Pneumonia']

# ==================== 打印配置 ====================
def print_config():
    print("=" * 50)
    print("📋 实验配置")
    print("=" * 50)
    print(f"🖥️  设备: {DEVICE}")
    print(f"📁 训练数据: {TRAIN_DIR}")
    print(f"📁 测试数据: {TEST_DIR}")
    print(f"🖼️  图像尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print(f"🔢 类别数: {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")
    print("-" * 50)
    print(f"🤖 自编码器: epochs={AE_EPOCHS}, batch={AE_BATCH_SIZE}, lr={AE_LR}")
    print(f"🧠 CNN: epochs={CNN_EPOCHS}, batch={CNN_BATCH_SIZE}, lr={CNN_LR}")
    print(f"🔗 端到端: epochs={E2E_EPOCHS}, lr_ae={E2E_LR_AE}, lr_cnn={E2E_LR_CNN}")
    print(f"⏹️  Early Stopping: patience={PATIENCE}, min_delta={MIN_DELTA}%")
    print(f"🔄 数据增强: {AUGMENT_LEVEL}, 噪声因子: {NOISE_FACTOR}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()