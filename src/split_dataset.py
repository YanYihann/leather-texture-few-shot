import os
import cv2
import shutil
from pathlib import Path

def split_image_dataset(source_dir, train_dir, val_dir, split_ratio=0.7):
    """
    将单张图片物理切割为训练集(左侧)和验证集(右侧)
    split_ratio: 0.7 表示左边 70% 作为训练集，右边 30% 作为验证集
    """
    # 创建目标文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    classes = sorted(os.listdir(source_dir))
    print(f"开始切割数据集，共发现 {len(classes)} 个类别...")
    
    for folder_name in classes:
        source_folder = os.path.join(source_dir, folder_name)
        if not os.path.isdir(source_folder):
            continue
            
        # 为训练集和验证集创建对应的类别子文件夹
        os.makedirs(os.path.join(train_dir, folder_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, folder_name), exist_ok=True)
        
        # 获取图片
        files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            img_name = files[0]
            img_path = os.path.join(source_folder, img_name)
            
            # 使用 OpenCV 读取图片
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            height, width, _ = img.shape
            
            # 计算切割的像素分界线
            split_point = int(width * split_ratio)
            
            # 切割图片 (Numpy 数组切片)
            train_img = img[:, :split_point, :] # 取左边
            val_img = img[:, split_point:, :]   # 取右边
            
            # 保存到对应的文件夹
            train_save_path = os.path.join(train_dir, folder_name, f"train_{img_name}")
            val_save_path = os.path.join(val_dir, folder_name, f"val_{img_name}")
            
            cv2.imwrite(train_save_path, train_img)
            cv2.imwrite(val_save_path, val_img)
            
    print(f"数据集切割完成！")
    print(f"训练集已保存至: {train_dir}")
    print(f"验证集已保存至: {val_dir}")

if __name__ == "__main__":
    # 你的原始数据集路径
    SOURCE_DATASET = "my_dataset_1" 
    # 即将生成的新数据集路径
    TRAIN_DATASET = "dataset_train"
    VAL_DATASET = "dataset_val"
    
    split_image_dataset(SOURCE_DATASET, TRAIN_DATASET, VAL_DATASET, split_ratio=0.7)