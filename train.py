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
    parser.add_argument(
        "--data",
        default="play-phone",
        help="数据集配置或名称，例如 labcoat 或 datasets/labcoat/labcoat.yaml",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--resume", action="store_true", help="从最近一次训练断点恢复")
    parser.add_argument("--checkpoint", type=str, help="指定断点权重文件 (last.pt) 路径")
    parser.add_argument("--runs-dir", type=str, help="训练输出目录（默认为 project 设置）")
    parser.add_argument("--project", type=str, help="自定义 Ultralytics 项目目录（默认 model）")
    parser.add_argument("--name", type=str, help="自定义本次训练 run 名称（默认数据集名）")
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

    # 解析数据集路径与名称（支持 data=labcoat 形式）
    data_path, dataset_name = resolve_data_path(args.data)

    # 训练输出位置：默认 project=model，name=数据集名
    project = Path(args.project) if args.project else Path("model")
    run_name = args.name if args.name else dataset_name
    runs_dir = Path(args.runs_dir) if args.runs_dir else project

    # 检测设备支持
    device = check_mps_support()

    # 加载模型/断点
    model, resume_mode = load_model(argparse.Namespace(**{**vars(args), "runs_dir": runs_dir}))

    # 组装训练参数
    train_kwargs = {"device": device}
    if resume_mode:
        train_kwargs["resume"] = True
    else:
        train_kwargs.update(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
        )

    train_kwargs["project"] = str(project)
    train_kwargs["name"] = run_name

    # 开始训练
    print(f"\n开始训练，使用设备: {device}")
    if resume_mode:
        print("模式: 断点恢复\n")
    else:
        print(f"数据集: {data_path}")
        print(f"轮次: {args.epochs}")
        print(f"图像尺寸: {args.imgsz}\n")

    model.train(**train_kwargs)


def resolve_data_path(data_arg: str) -> Tuple[Path, str]:
    """
    解析 data 参数，支持：
    1) 直接传 .yaml 路径
    2) 传数据集名称（如 labcoat），自动查找 datasets/<name>/<name>.yaml 或 datasets/<name>.yaml
    """
    path_candidate = Path(data_arg)
    if path_candidate.suffix.lower() == ".yaml":
        if path_candidate.exists():
            return path_candidate.resolve(), path_candidate.stem
        # 相对路径尝试
        rel_candidate = Path("datasets") / path_candidate
        if rel_candidate.exists():
            return rel_candidate.resolve(), path_candidate.stem
        print(f"❌ 未找到数据配置文件: {path_candidate}")
        sys.exit(1)

    dataset_name = data_arg
    candidates = [
        Path("datasets") / dataset_name / f"{dataset_name}.yaml",
        Path("datasets") / f"{dataset_name}.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve(), dataset_name

    print(f"❌ 未找到数据集 '{dataset_name}' 对应的配置，尝试路径: {', '.join(str(c) for c in candidates)}")
    sys.exit(1)


if __name__ == "__main__":
    main()
