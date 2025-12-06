"""
数据加载模块
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import *


def get_transforms(is_train=True):
    """获取数据变换"""
    transform_list = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
    
    # 训练集可以添加数据增强
    if is_train:
        transform_list.insert(2, transforms.RandomHorizontalFlip(p=0.3))
        transform_list.insert(2, transforms.RandomRotation(degrees=10))
    
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
        pin_memory=True
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
    
    # 统计每类数量
    train_counts = {}
    for _, label in train_dataset:
        class_name = CLASS_NAMES[label]
        train_counts[class_name] = train_counts.get(class_name, 0) + 1
    
    test_counts = {}
    for _, label in test_dataset:
        class_name = CLASS_NAMES[label]
        test_counts[class_name] = test_counts.get(class_name, 0) + 1
    
    print(f"📈 训练集分布: {train_counts}")
    print(f"📈 测试集分布: {test_counts}")
    
    return train_loader, test_loader, train_dataset, test_dataset


def add_noise(img, noise_factor=NOISE_FACTOR):
    """添加高斯噪声"""
    noisy_img = img + noise_factor * torch.randn_like(img)
    noisy_img = torch.clamp(noisy_img, 0., 1.)
    return noisy_img


if __name__ == "__main__":
    train_loader, test_loader, _, _ = load_data()
    
    # 测试数据加载
    for img, label in train_loader:
        print(f"Batch shape: {img.shape}, Labels: {label[:5]}")
        break