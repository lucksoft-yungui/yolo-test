import torch
import platform
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

# 检测设备支持
device = check_mps_support()

# Load a model
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model with device specification
print(f"\n开始训练，使用设备: {device}\n")
results = model.train(data="play-phone.yaml", epochs=100, imgsz=640, device=device)