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
AE_EPOCHS = 50
AE_BATCH_SIZE = 32
AE_LR = 0.001
NOISE_FACTOR = 0.5

# CNN参数
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 32
CNN_LR = 0.001
NUM_CLASSES = 3

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
    print("=" * 50)

if __name__ == "__main__":
    print_config()