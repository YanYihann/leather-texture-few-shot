import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==================== 1. 数据集与强力图像增强 ====================

class LeatherDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, samples_per_class=50):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.samples_per_class = samples_per_class # 虚拟扩充：让每张图在一个 Epoch 里被随机裁剪 50 次
        
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(dataset_dir))
        
        # 遍历读取 203 张图的路径
        for label_idx, folder_name in enumerate(self.classes):
            folder_path = os.path.join(dataset_dir, folder_name)
            if os.path.isdir(folder_path):
                files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    img_path = os.path.join(folder_path, files[0])
                    # 把同一张图存入列表 samples_per_class 次
                    for _ in range(self.samples_per_class):
                        self.image_paths.append(img_path)
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 使用 OpenCV 读取图片 (BGR) 并转为 RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用 albumentations 图像增强
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

# 定义 albumentations 增强流程
train_transform = A.Compose([
    # 注意这里改成了 size=(224, 224)
    A.RandomResizedCrop(size=(224, 224), scale=(0.4, 1.0)), 
    A.HorizontalFlip(p=0.5), 
    A.VerticalFlip(p=0.5),   
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ==================== 2. 模型设置 ====================

def get_training_model(num_classes):
    print("加载 ResNet50 并替换分类头...")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    # 获取原本全连接层的输入维度 (ResNet50 是 2048)
    num_ftrs = model.fc.in_features
    # 替换为一个新的全连接层，输出维度为你的皮革种类数 (203)
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

# ==================== 3. 训练流程 ====================

def train_model():
    dataset_dir = "my_dataset_1"
    num_classes = 203
    epochs = 10        # 训练轮数
    batch_size = 32    # 批次大小 (如果显存不够报错，可以改为 16 或 8)
    learning_rate = 0.0001
    
    # 准备数据加载器
    train_dataset = LeatherDataset(dataset_dir, transform=train_transform, samples_per_class=50)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model, device = get_training_model(num_classes)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 只微调模型，学习率设置得比较小
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\n🚀 开始训练，设备: {device}，总轮数: {epochs}")
    print(f"每轮包含 {len(train_loader)} 个 Batch...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用 tqdm 显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad() # 清空梯度
            
            outputs = model(inputs) # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            
            loss.backward() # 反向传播
            optimizer.step() # 更新权重
            
            # 计算准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"✅ Epoch [{epoch+1}/{epochs}] 完成! 平均 Loss: {epoch_loss:.4f}, 准确率 Acc: {epoch_acc:.2f}%")

    # 保存训练好的模型权重
    save_path = "best_leather_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n训练结束！模型权重已保存至: {save_path}")

if __name__ == "__main__":
    # 在 Windows 下运行多进程 DataLoader 需要保护
    train_model()