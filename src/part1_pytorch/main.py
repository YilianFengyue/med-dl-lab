"""
主入口文件 - 支持V2(分离训练)和V3(端到端训练)
"""
import argparse
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from dataset import load_data
from models import Autoencoder, CNN, count_parameters
from train_autoencoder import train_autoencoder, plot_ae_loss, visualize_denoising
from train_cnn import train_cnn, plot_cnn_history
from train_e2e import train_e2e, plot_e2e_history
from evaluate import generate_full_report


def print_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║   🏥 肺炎图像识别系统 V3 (端到端联合训练)                     ║
    ║   Pneumonia Classification with End-to-End Training          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


def run_v2_pipeline():
    """V2流程：分离训练（对比用）"""
    print_banner()
    print("📌 运行V2流程：分离训练 (AE冻结)")
    print_config()
    
    train_loader, test_loader, _, _, class_weights = load_data()
    
    # 训练自编码器
    autoencoder, ae_history = train_autoencoder(train_loader, test_loader)
    plot_ae_loss(ae_history)
    visualize_denoising(autoencoder, test_loader)
    
    # 训练CNN（冻结AE）
    cnn, cnn_history = train_cnn(train_loader, test_loader, autoencoder, class_weights)
    plot_cnn_history(cnn_history)
    
    # 评估
    generate_full_report(cnn, autoencoder, test_loader)


def run_v3_pipeline():
    """V3流程：端到端联合训练（推荐）"""
    print_banner()
    print("📌 运行V3流程：端到端联合训练")
    print_config()
    
    train_loader, test_loader, _, _, class_weights = load_data()
    
    # Step 1: 先预训练AE和CNN
    print("\n" + "=" * 60)
    print("🔧 Phase 1: 预训练自编码器")
    print("=" * 60)
    autoencoder, ae_history = train_autoencoder(train_loader, test_loader)
    plot_ae_loss(ae_history)
    
    print("\n" + "=" * 60)
    print("🧠 Phase 2: 预训练CNN (AE冻结)")
    print("=" * 60)
    cnn, cnn_history = train_cnn(train_loader, test_loader, autoencoder, class_weights)
    plot_cnn_history(cnn_history)
    
    # Step 2: 端到端联合训练
    print("\n" + "=" * 60)
    print("🔗 Phase 3: 端到端联合微调 (核心)")
    print("=" * 60)
    autoencoder, cnn, e2e_history = train_e2e(train_loader, test_loader, autoencoder, cnn, class_weights)
    plot_e2e_history(e2e_history)
    
    # Step 3: 可视化去噪效果
    visualize_denoising(autoencoder, test_loader)
    
    # Step 4: 评估
    print("\n" + "=" * 60)
    print("📊 Phase 4: 模型评估")
    print("=" * 60)
    generate_full_report(cnn, autoencoder, test_loader)
    
    print("\n" + "=" * 60)
    print("🎉 V3训练完成！")
    print("=" * 60)


def run_e2e_only():
    """仅运行端到端训练（需要预训练模型）"""
    print_banner()
    print("📌 仅运行端到端联合训练（加载预训练权重）")
    print_config()
    
    train_loader, test_loader, _, _, class_weights = load_data()
    
    # 加载预训练模型
    autoencoder = Autoencoder().to(DEVICE)
    cnn = CNN().to(DEVICE)
    
    ae_path = MODEL_DIR / "autoencoder_best.pth"
    cnn_path = MODEL_DIR / "cnn_best.pth"
    
    if not ae_path.exists() or not cnn_path.exists():
        print("❌ 找不到预训练模型，请先运行 --v2 或完整V3流程")
        return
    
    autoencoder.load_state_dict(torch.load(ae_path, weights_only=True))
    cnn.load_state_dict(torch.load(cnn_path, weights_only=True))
    print("✅ 加载预训练模型")
    
    # 端到端训练
    autoencoder, cnn, e2e_history = train_e2e(train_loader, test_loader, autoencoder, cnn, class_weights)
    plot_e2e_history(e2e_history)
    visualize_denoising(autoencoder, test_loader)
    generate_full_report(cnn, autoencoder, test_loader)


def run_eval_only():
    """只评估模型"""
    print_banner()
    print_config()
    
    _, test_loader, _, _, _ = load_data()
    
    autoencoder = Autoencoder().to(DEVICE)
    cnn = CNN().to(DEVICE)
    
    # 优先加载E2E模型
    e2e_path = MODEL_DIR / "e2e_best.pth"
    if e2e_path.exists():
        ckpt = torch.load(e2e_path, weights_only=True)
        autoencoder.load_state_dict(ckpt['ae'])
        cnn.load_state_dict(ckpt['cnn'])
        print("✅ 加载端到端模型 (e2e_best.pth)")
    else:
        ae_path = MODEL_DIR / "autoencoder_best.pth"
        cnn_path = MODEL_DIR / "cnn_best.pth"
        if ae_path.exists() and cnn_path.exists():
            autoencoder.load_state_dict(torch.load(ae_path, weights_only=True))
            cnn.load_state_dict(torch.load(cnn_path, weights_only=True))
            print("✅ 加载分离训练模型")
        else:
            print("❌ 找不到模型文件")
            return
    
    generate_full_report(cnn, autoencoder, test_loader)


def show_model_info():
    ae, cnn = Autoencoder(), CNN()
    print(ae.get_architecture())
    print(f"📊 Autoencoder 参数量: {count_parameters(ae):,}")
    print(cnn.get_architecture())
    print(f"📊 CNN 参数量: {count_parameters(cnn):,}")


def main():
    parser = argparse.ArgumentParser(
        description='肺炎图像识别系统 V3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py              # V3完整流程（推荐）
  python main.py --v2         # V2分离训练（对比用）
  python main.py --e2e        # 仅端到端训练（需预训练模型）
  python main.py --eval       # 仅评估
  python main.py --info       # 模型信息
        """
    )
    
    parser.add_argument('--v2', action='store_true', help='V2分离训练流程')
    parser.add_argument('--e2e', action='store_true', help='仅端到端训练')
    parser.add_argument('--eval', action='store_true', help='仅评估')
    parser.add_argument('--info', action='store_true', help='模型信息')
    
    args = parser.parse_args()
    
    if args.info:
        show_model_info()
    elif args.v2:
        run_v2_pipeline()
    elif args.e2e:
        run_e2e_only()
    elif args.eval:
        run_eval_only()
    else:
        run_v3_pipeline()  # 默认V3


if __name__ == "__main__":
    main()