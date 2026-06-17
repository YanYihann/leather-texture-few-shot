import os
import csv
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

def load_trained_model(model_path, num_classes=203):
    print("正在加载模型...")
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def batch_predict_and_export(test_folder, model, device, output_csv_path, dataset_dir="dataset_train"):
    classes = sorted(os.listdir(dataset_dir))
    results = []
    
    if not os.path.exists(test_folder):
        print("测试文件夹不存在，请检查路径。")
        return
        
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print("测试文件夹中没有找到图片文件。")
        return
        
    print(f"共找到 {len(image_files)} 张测试图片，开始批量预测...")
    
    for img_name in image_files:
        img_path = os.path.join(test_folder, img_name)
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = test_transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top_prob, top_idx = torch.max(probabilities, 0)
                
            best_match_name = classes[top_idx.item()]
            similarity_score = round(top_prob.item() * 100, 2)
            
            results.append([img_name, best_match_name, f"{similarity_score}%"])
            print(f"已处理: {img_name} -> {best_match_name} ({similarity_score}%)")
            
        except Exception as e:
            print(f"处理图片 {img_name} 时发生错误: {e}")
            
    with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["图片名称", "预测皮革编号", "相似度"])
        writer.writerows(results)
        
    print(f"批量预测完成。结果已成功保存到 {output_csv_path}")

if __name__ == "__main__":
    model_file = "best_leather_model_val.pth"
    
    test_images_folder = "test_images"
    output_report_file = "leather_prediction_report.csv"
    
    if os.path.exists(model_file):
        model, device = load_trained_model(model_file)
        batch_predict_and_export(test_images_folder, model, device, output_report_file)
    else:
        print("找不到模型权重文件，请确保路径正确。")