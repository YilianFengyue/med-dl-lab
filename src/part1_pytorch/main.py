"""
主入口文件 - 一键运行整个实验流程
"""
import argparse
import torch
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import *
from dataset import load_data
from models import Autoencoder, CNN, count_parameters
from train_autoencoder import train_autoencoder, plot_ae_loss, visualize_denoising
from train_cnn import train_cnn, plot_cnn_history
from evaluate import generate_full_report


def print_banner():
    """打印欢迎信息"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🏥 基于自编码器和CNN的肺炎图像识别系统                      ║
    ║   Pneumonia Image Classification with Autoencoder + CNN       ║
    ║                                                               ║
    ║   📚 实验六：基于深度学习的医药诊断评估系统 (模块1)           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_full_pipeline():
    """运行完整流程"""
    print_banner()
    print_config()
    
    # ==================== 1. 加载数据 ====================
    print("\n" + "=" * 60)
    print("📦 Step 1/4: 加载数据集")
    print("=" * 60)
    train_loader, test_loader, train_dataset, test_dataset = load_data()
    
    # ==================== 2. 训练自编码器 ====================
    print("\n" + "=" * 60)
    print("🔧 Step 2/4: 训练自编码器")
    print("=" * 60)
    autoencoder, ae_history = train_autoencoder(train_loader, test_loader)
    plot_ae_loss(ae_history)
    visualize_denoising(autoencoder, test_loader)
    
    # ==================== 3. 训练CNN ====================
    print("\n" + "=" * 60)
    print("🧠 Step 3/4: 训练CNN分类器")
    print("=" * 60)
    cnn, cnn_history = train_cnn(train_loader, test_loader, autoencoder)
    plot_cnn_history(cnn_history)
    
    # ==================== 4. 评估 ====================
    print("\n" + "=" * 60)
    print("📊 Step 4/4: 模型评估与可视化")
    print("=" * 60)
    generate_full_report(cnn, autoencoder, test_loader)
    
    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("🎉 实验完成！")
    print("=" * 60)
    print(f"📁 模型保存位置: {MODEL_DIR}")
    print(f"📁 图片保存位置: {FIGURE_DIR}")
    print("\n生成的文件:")
    for f in FIGURE_DIR.glob("*.png"):
        print(f"   📊 {f.name}")
    for f in MODEL_DIR.glob("*.pth"):
        print(f"   🤖 {f.name}")


def run_train_only():
    """只训练模型"""
    print_banner()
    print_config()
    
    train_loader, test_loader, _, _ = load_data()
    
    # 训练自编码器
    autoencoder, ae_history = train_autoencoder(train_loader, test_loader)
    plot_ae_loss(ae_history)
    visualize_denoising(autoencoder, test_loader)
    
    # 训练CNN
    cnn, cnn_history = train_cnn(train_loader, test_loader, autoencoder)
    plot_cnn_history(cnn_history)
    
    print("\n✅ 训练完成！")


def run_eval_only():
    """只评估模型"""
    print_banner()
    print_config()
    
    _, test_loader, _, _ = load_data()
    
    # 加载模型
    autoencoder = Autoencoder().to(DEVICE)
    cnn = CNN().to(DEVICE)
    
    ae_path = MODEL_DIR / "autoencoder_best.pth"
    cnn_path = MODEL_DIR / "cnn_best.pth"
    
    if not ae_path.exists() or not cnn_path.exists():
        print("❌ 错误: 找不到预训练模型，请先运行训练!")
        print(f"   期望路径: {ae_path}")
        print(f"   期望路径: {cnn_path}")
        return
    
    autoencoder.load_state_dict(torch.load(ae_path))
    cnn.load_state_dict(torch.load(cnn_path))
    print("✅ 模型加载完成")
    
    generate_full_report(cnn, autoencoder, test_loader)


def show_model_info():
    """显示模型信息"""
    print_banner()
    
    ae = Autoencoder()
    cnn = CNN()
    
    print("\n" + "=" * 60)
    print("🤖 模型架构信息")
    print("=" * 60)
    
    print(ae.get_architecture())
    print(f"📊 Autoencoder 参数量: {count_parameters(ae):,}")
    
    print(cnn.get_architecture())
    print(f"📊 CNN 参数量: {count_parameters(cnn):,}")


def main():
    parser = argparse.ArgumentParser(
        description='肺炎图像识别系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    # 运行完整流程
  python main.py --train            # 只训练模型
  python main.py --eval             # 只评估模型
  python main.py --info             # 显示模型信息
  python main.py --ae-epochs 100    # 自定义epoch数
        """
    )
    
    parser.add_argument('--train', action='store_true', help='只运行训练')
    parser.add_argument('--eval', action='store_true', help='只运行评估')
    parser.add_argument('--info', action='store_true', help='显示模型信息')
    parser.add_argument('--ae-epochs', type=int, help='自编码器训练轮数')
    parser.add_argument('--cnn-epochs', type=int, help='CNN训练轮数')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    
    args = parser.parse_args()
    
    # 动态修改配置
    if args.ae_epochs:
        import config
        config.AE_EPOCHS = args.ae_epochs
    if args.cnn_epochs:
        import config
        config.CNN_EPOCHS = args.cnn_epochs
    if args.lr:
        import config
        config.AE_LR = args.lr
        config.CNN_LR = args.lr
    if args.batch_size:
        import config
        config.AE_BATCH_SIZE = args.batch_size
        config.CNN_BATCH_SIZE = args.batch_size
    
    # 执行对应操作
    if args.info:
        show_model_info()
    elif args.train:
        run_train_only()
    elif args.eval:
        run_eval_only()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()