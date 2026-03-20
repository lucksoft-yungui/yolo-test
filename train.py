import argparse
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import load_checkpoint

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

MODEL_CONFIGS = {
    "yolo11n": {"yaml": "yolo11n.yaml", "weights": "yolo11n.pt", "imgsz": 640},
    "yolo11s": {"yaml": "yolo11s.yaml", "weights": "yolo11s.pt", "imgsz": 640},
    "yolo11m": {"yaml": "yolo11m.yaml", "weights": "yolo11m.pt", "imgsz": 640},
    "yolo11l": {"yaml": "yolo11l.yaml", "weights": "yolo11l.pt", "imgsz": 640},
    "yolo11x": {"yaml": "yolo11x.yaml", "weights": "yolo11x.pt", "imgsz": 640},
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLO 模型训练脚本，支持断点恢复")
    parser.add_argument(
        "--data",
        default="play-phone",
        help="数据集配置或名称，例如 labcoat 或 datasets/labcoat/labcoat.yaml",
    )
    parser.add_argument(
        "--file",
        help="数据集配置文件路径，可替代 --data 直接传入 yaml",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="yolo11n",
        choices=sorted(MODEL_CONFIGS.keys()),
        help="选择基础模型规模，默认 yolo11n",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--imgsz", type=int, default=None, help="输入图片尺寸（不填则使用模型默认）")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (epoch 数)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从最近一次训练断点恢复，保持 Ultralytics 原生 resume 行为",
    )
    parser.add_argument(
        "--continus",
        action="store_true",
        help="从最近一次训练 checkpoint 继续追加训练；配合 --epochs 表示额外再训练多少轮",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="指定断点权重文件路径（通常为 last.pt）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定训练设备，如 cpu/mps/cuda/cuda:0/0 或 0,1,2（多卡）（不填则自动检测）",
    )
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

def load_model(args) -> Tuple[YOLO, Optional[Path]]:
    """根据参数加载模型，并返回所使用的 checkpoint 路径（若有）"""
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ 指定的断点文件不存在: {checkpoint_path}")
            sys.exit(1)
        print(f"🔄 使用指定断点: {checkpoint_path}")
        return YOLO(str(checkpoint_path)), checkpoint_path

    if args.resume or args.continus:
        checkpoint_path = find_latest_checkpoint(Path(args.runs_dir))
        if checkpoint_path:
            print(f"🔄 检测到最近的断点文件: {checkpoint_path}")
            return YOLO(str(checkpoint_path)), checkpoint_path
        print("⚠️ 未在 runs 目录中找到可用的断点，改为重新训练")

    model_cfg = MODEL_CONFIGS[args.model_size]
    print(f"🆕 未指定断点，将从预训练权重开始训练: {args.model_size}")
    return YOLO(model_cfg["yaml"]).load(model_cfg["weights"]), None


def read_checkpoint_info(checkpoint_path: Path) -> Dict[str, Any]:
    """读取 checkpoint 中的训练元信息，用于继续追加训练。"""
    try:
        _, ckpt = load_checkpoint(str(checkpoint_path), device="cpu", fuse=False)
    except Exception as exc:
        print(f"❌ 读取断点文件失败: {checkpoint_path}")
        print(f"原因: {exc}")
        sys.exit(1)

    train_args = ckpt.get("train_args", {}) or {}
    epoch = ckpt.get("epoch", -1)
    completed_epochs = epoch + 1 if isinstance(epoch, int) and epoch >= 0 else 0
    saved_total_epochs = train_args.get("epochs")
    return {
        "completed_epochs": completed_epochs,
        "saved_total_epochs": saved_total_epochs if isinstance(saved_total_epochs, int) else None,
        "train_args": train_args,
    }

def resolve_device(device_arg: Optional[str]) -> str:
    """根据参数或自动检测选择训练设备"""
    if not device_arg:
        return check_mps_support()

    device = device_arg.strip()
    if "," in device:
        indices = [part.strip() for part in device.split(",") if part.strip() != ""]
        if not indices or any(not idx.isdigit() for idx in indices):
            print(f"❌ 不支持的 device 参数: {device_arg}，多卡请使用如 0,1,2 的格式")
            sys.exit(1)
        if not torch.cuda.is_available():
            print("❌ 指定 CUDA 多卡但当前环境未检测到可用 GPU")
            sys.exit(1)
        max_count = torch.cuda.device_count()
        bad = [idx for idx in indices if int(idx) >= max_count]
        if bad:
            print(f"❌ 指定 GPU 索引超出范围: {', '.join(bad)}，可用数量: {max_count}")
            sys.exit(1)
        print(f"✅ 指定使用 CUDA 多卡: {', '.join(indices)}")
        return ",".join(indices)

    if device.isdigit():
        device = f"cuda:{device}"
    device_lower = device.lower()

    if device_lower == "cpu":
        print("✅ 指定使用 CPU 训练")
        return "cpu"

    if device_lower == "mps":
        if torch.backends.mps.is_available():
            print("✅ 指定使用 MPS 训练")
            return "mps"
        print("❌ 指定 MPS 但当前环境不支持")
        sys.exit(1)

    if device_lower == "cuda" or device_lower.startswith("cuda:"):
        if not torch.cuda.is_available():
            print("❌ 指定 CUDA 但当前环境未检测到可用 GPU")
            sys.exit(1)
        if device_lower.startswith("cuda:"):
            idx_str = device_lower.split("cuda:", 1)[1]
            if idx_str.isdigit():
                idx = int(idx_str)
                if idx >= torch.cuda.device_count():
                    print(f"❌ 指定 GPU 索引超出范围: {idx}，可用数量: {torch.cuda.device_count()}")
                    sys.exit(1)
                print(f"✅ 指定使用 CUDA 设备: {torch.cuda.get_device_name(idx)} (cuda:{idx})")
            else:
                print(f"✅ 指定使用 CUDA 设备: {device}")
        else:
            print(f"✅ 指定使用 CUDA 设备: {torch.cuda.get_device_name(0)} (cuda)")
        return device

    print(f"❌ 不支持的 device 参数: {device_arg}，可选 cpu/mps/cuda/cuda:<index>/<index>")
    sys.exit(1)

def main():
    args = parse_args()
    if args.resume and args.continus:
        print("❌ --resume 和 --continus 不能同时使用")
        sys.exit(1)

    # 优先使用 --file 指定的配置文件，否则回落到 --data
    data_arg = args.file if args.file else args.data

    # 解析数据集路径与名称（支持 data=labcoat 形式）
    data_path, dataset_name = resolve_data_path(data_arg)

    # 训练输出位置：默认 project=model，name=数据集名
    project = Path(args.project) if args.project else Path("model")
    run_name = args.name if args.name else dataset_name
    runs_dir = Path(args.runs_dir) if args.runs_dir else project

    # 选择训练设备
    device = resolve_device(args.device)

    # 加载模型/断点
    model, checkpoint_path = load_model(argparse.Namespace(**{**vars(args), "runs_dir": runs_dir}))

    # 组装训练参数
    model_cfg = MODEL_CONFIGS[args.model_size]
    checkpoint_info = read_checkpoint_info(checkpoint_path) if checkpoint_path and args.continus else None
    default_imgsz = model_cfg["imgsz"]
    if checkpoint_info and args.imgsz is None:
        saved_imgsz = checkpoint_info["train_args"].get("imgsz")
        imgsz = saved_imgsz if isinstance(saved_imgsz, int) and saved_imgsz > 0 else default_imgsz
    else:
        imgsz = args.imgsz if args.imgsz is not None else default_imgsz

    if args.resume and checkpoint_path:
        train_kwargs = {"device": device, "resume": True}
    else:
        train_kwargs = {
            "data": str(data_path),
            "epochs": args.epochs,
            "imgsz": imgsz,
            "patience": args.patience,
            "device": device,
        }

    if checkpoint_info:
        completed_epochs = checkpoint_info["completed_epochs"]
        train_kwargs["epochs"] = completed_epochs + args.epochs

    train_kwargs["project"] = str(project)
    train_kwargs["name"] = run_name

    # 开始训练
    print(f"\n开始训练，使用设备: {device}")
    if args.resume and checkpoint_path:
        print("模式: 原生断点恢复")
        print(f"断点: {checkpoint_path}\n")
    else:
        print(f"数据集: {data_path}")
        print(f"模型规模: {args.model_size}")
        print(f"图像尺寸: {imgsz}")
    if checkpoint_info:
        saved_total_epochs = checkpoint_info["saved_total_epochs"]
        completed_epochs = checkpoint_info["completed_epochs"]
        print("模式: 从 checkpoint 继续追加训练")
        print(f"断点: {checkpoint_path}")
        print(f"已完成轮次: {completed_epochs}")
        if saved_total_epochs is not None:
            print(f"原计划总轮次: {saved_total_epochs}")
        print(f"本次追加轮次: {args.epochs}")
        print(f"新的总轮次: {train_kwargs['epochs']}\n")
    elif not (args.resume and checkpoint_path):
        print(f"轮次: {args.epochs}\n")

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