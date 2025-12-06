"""
数据加载模块
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import *


def get_transforms(is_train=True):
    """获取数据变换 - 强化版数据增强"""
    if is_train:
        # 强数据增强 - 有效扩充小数据集
        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),  # 稍大一点用于裁剪
            transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),  # 随机裁剪
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # 医学图像可以垂直翻转
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # 平移
                scale=(0.9, 1.1),  # 缩放
                shear=5  # 剪切
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # 随机擦除
        ]
    else:
        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    
    return transforms.Compose(transform_list)


def load_data():
    """加载训练集和测试集"""
    print("\n📂 加载数据集...")
    
    # 训练集
    train_transform = get_transforms(is_train=True)
    train_dataset = datasets.ImageFolder(
        root=str(TRAIN_DIR),
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True  # 丢弃不完整的batch，稳定训练
    )
    
    # 测试集（已加噪）
    test_transform = get_transforms(is_train=False)
    test_dataset = datasets.ImageFolder(
        root=str(TEST_DIR),
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),  # 一次性加载所有测试数据
        shuffle=False,
        num_workers=0
    )
    
    # 打印数据集信息
    print(f"✅ 训练集: {len(train_dataset)} 张图片")
    print(f"✅ 测试集: {len(test_dataset)} 张图片")
    print(f"📊 类别映射: {train_dataset.class_to_idx}")
    
    # 统计每类数量并计算类别权重
    class_counts = [0] * NUM_CLASSES
    for _, label in train_dataset:
        class_counts[label] += 1
    
    print(f"📈 训练集分布: {dict(zip(CLASS_NAMES, class_counts))}")
    
    # 计算类别权重（反比于样本数）
    total = sum(class_counts)
    class_weights = [total / (NUM_CLASSES * c) for c in class_counts]
    class_weights = torch.FloatTensor(class_weights)
    print(f"⚖️  类别权重: {[f'{w:.2f}' for w in class_weights.tolist()]}")
    
    return train_loader, test_loader, train_dataset, test_dataset, class_weights


def add_noise(img, noise_factor=NOISE_FACTOR):
    """添加高斯噪声"""
    noisy_img = img + noise_factor * torch.randn_like(img)
    noisy_img = torch.clamp(noisy_img, 0., 1.)
    return noisy_img


if __name__ == "__main__":
    train_loader, test_loader, _, _, class_weights = load_data()
    print(f"\n类别权重: {class_weights}")
    
    # 测试数据加载
    for img, label in train_loader:
        print(f"Batch shape: {img.shape}, Labels: {label[:5]}")
        break