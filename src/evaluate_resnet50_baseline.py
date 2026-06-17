import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class LeatherImageDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.class_names = sorted(
            item.name for item in self.dataset_dir.iterdir() if item.is_dir()
        )
        self.samples = []

        for label, class_name in enumerate(self.class_names):
            class_dir = self.dataset_dir / class_name
            files = sorted(
                file
                for file in class_dir.iterdir()
                if file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )
            for file in files:
                self.samples.append((file, label, class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, class_name, str(image_path)


def build_model(num_classes, model_path, device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    dataset = LeatherImageDataset(args.dataset_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = build_model(len(dataset.class_names), args.model_path, device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top3_correct = 0
    rows = []

    start_time = time.perf_counter()
    with torch.no_grad():
        for images, labels, class_names, image_paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            top1_correct += (top_indices[:, 0] == labels).sum().item()
            top3_correct += (
                top_indices == labels.unsqueeze(1)
            ).any(dim=1).sum().item()

            for i in range(batch_size):
                predicted = [
                    dataset.class_names[index.item()] for index in top_indices[i]
                ]
                scores = [round(prob.item() * 100, 4) for prob in top_probs[i]]
                rows.append(
                    {
                        "image_path": image_paths[i],
                        "true_class": class_names[i],
                        "top1_class": predicted[0],
                        "top1_confidence": scores[0],
                        "top2_class": predicted[1],
                        "top2_confidence": scores[1],
                        "top3_class": predicted[2],
                        "top3_confidence": scores[2],
                        "top1_correct": predicted[0] == class_names[i],
                        "top3_correct": class_names[i] in predicted,
                    }
                )

    elapsed = time.perf_counter() - start_time
    metrics = {
        "method": "ResNet50 fine-tuning baseline",
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "num_classes": len(dataset.class_names),
        "num_images": len(dataset),
        "device": str(device),
        "batch_size": args.batch_size,
        "loss": total_loss / total_samples,
        "top1_accuracy": top1_correct / total_samples,
        "top3_accuracy": top3_correct / total_samples,
        "elapsed_seconds": elapsed,
        "avg_inference_ms_per_image": elapsed * 1000 / total_samples,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "stage1_resnet50_baseline_metrics.json"
    predictions_path = output_dir / "stage1_resnet50_baseline_predictions.csv"

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with predictions_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset_val")
    parser.add_argument("--model-path", default="best_leather_model_val.pth")
    parser.add_argument("--output-dir", default="../02_实验结果与生成数据/results")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
