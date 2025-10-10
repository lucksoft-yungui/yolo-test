import cv2
import os
from pathlib import Path

# COCO 数据集的类别名称（前 10 个常见类别）
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
    'bus', 'train', 'truck', 'boat', 'traffic light'
]

def draw_yolo_boxes(image_path, label_path):
    """读取图片和YOLO标签，绘制边界框"""
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # 读取标签文件
    if not os.path.exists(label_path):
        print(f"标签文件不存在: {label_path}")
        return img
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # 绘制每个边界框
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])
        
        # 转换YOLO格式到像素坐标
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加类别标签
        label = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
        cv2.putText(img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img

# 设置数据集路径
dataset_path = Path("datasets/coco8")
images_dir = dataset_path / "images" / "train"
labels_dir = dataset_path / "labels" / "train"

print("=" * 60)
print("COCO8 数据集可视化工具")
print("=" * 60)
print("按任意键查看下一张图片，按 'q' 退出")
print("=" * 60)

# 遍历所有图片
image_files = sorted(images_dir.glob("*.jpg"))
for img_file in image_files:
    # 构建对应的标签文件路径
    label_file = labels_dir / (img_file.stem + ".txt")
    
    # 绘制边界框
    img_with_boxes = draw_yolo_boxes(img_file, label_file)
    
    if img_with_boxes is not None:
        # 显示图片
        window_name = f"COCO8 - {img_file.name}"
        cv2.imshow(window_name, img_with_boxes)
        
        # 等待按键
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 按 'q' 退出
        if key == ord('q'):
            break

print("\n查看完成！")
