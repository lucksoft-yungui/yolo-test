import argparse
import platform
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from ultralytics import YOLO

# 检测 macOS M 芯片 (Apple Silicon) 支持
def check_mps_support():
    """检测并打印 macOS M 芯片 (MPS) 支持情况"""
    print("=" * 60)
    print("系统信息检测:")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"处理器架构: {platform.machine()}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 检测 MPS (Metal Performance Shaders) 支持
    if torch.backends.mps.is_available():
        print("✅ macOS M 芯片 (Apple Silicon) MPS 加速: 支持")
        print("✅ 将使用 MPS 设备进行训练加速")
        device = "mps"
    elif torch.cuda.is_available():
        print("✅ CUDA 加速: 支持")
        print(f"✅ GPU 设备: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️  GPU 加速: 不支持")
        print("⚠️  将使用 CPU 进行训练（速度较慢）")
        device = "cpu"
    
    print("=" * 60)
    return device

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLO 模型训练脚本，支持断点恢复")
    parser.add_argument("--data", default="play-phone.yaml", help="训练数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--resume", action="store_true", help="从最近一次训练断点恢复")
    parser.add_argument("--checkpoint", type=str, help="指定断点权重文件 (last.pt) 路径")
    parser.add_argument("--runs-dir", default="runs/detect", help="训练输出目录，用于自动寻找断点")
    parser.add_argument("--project", type=str, help="自定义 Ultralytics 项目目录")
    parser.add_argument("--name", type=str, help="自定义本次训练 run 名称")
    return parser.parse_args()

def find_latest_checkpoint(runs_dir: Path) -> Optional[Path]:
    """在指定 runs 目录中查找最新的 last.pt"""
    if not runs_dir.exists():
        return None
    candidates = sorted(
        (path for path in runs_dir.iterdir() if path.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        ckpt = run_dir / "weights" / "last.pt"
        if ckpt.exists():
            return ckpt
    return None

def load_model(args) -> Tuple[YOLO, bool]:
    """根据参数加载模型，并确定是否需要恢复训练"""
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ 指定的断点文件不存在: {checkpoint_path}")
            sys.exit(1)
        print(f"🔄 使用指定断点继续训练: {checkpoint_path}")
        return YOLO(str(checkpoint_path)), True

    if args.resume:
        checkpoint_path = find_latest_checkpoint(Path(args.runs_dir))
        if checkpoint_path:
            print(f"🔄 检测到最近的断点文件: {checkpoint_path}")
            return YOLO(str(checkpoint_path)), True
        print("⚠️ 未在 runs 目录中找到可用的断点，改为重新训练")

    print("🆕 未指定断点，将从预训练权重开始训练")
    return YOLO("yolo11n.yaml").load("yolo11n.pt"), False

def main():
    args = parse_args()

    # 检测设备支持
    device = check_mps_support()

    # 加载模型/断点
    model, resume_mode = load_model(args)

    # 组装训练参数
    train_kwargs = {"device": device}
    if resume_mode:
        train_kwargs["resume"] = True
    else:
        train_kwargs.update(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
        )

    if args.project:
        train_kwargs["project"] = args.project
    if args.name:
        train_kwargs["name"] = args.name

    # 开始训练
    print(f"\n开始训练，使用设备: {device}")
    if resume_mode:
        print("模式: 断点恢复\n")
    else:
        print(f"数据集: {args.data}")
        print(f"轮次: {args.epochs}")
        print(f"图像尺寸: {args.imgsz}\n")

    model.train(**train_kwargs)

if __name__ == "__main__":
    main()
