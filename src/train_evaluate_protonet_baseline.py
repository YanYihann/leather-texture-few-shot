import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class_images(root):
    root = Path(root)
    class_names = sorted(item.name for item in root.iterdir() if item.is_dir())
    data = {}
    for class_name in class_names:
        files = sorted(file for file in (root / class_name).iterdir() if file.suffix.lower() in IMAGE_EXTENSIONS)
        if files:
            data[class_name] = files
    return class_names, data


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ProtoNetConv4(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, hidden),
            ConvBlock(hidden, hidden),
            ConvBlock(hidden, hidden),
            ConvBlock(hidden, hidden),
        )

    def forward(self, images):
        features = self.encoder(images)
        return F.normalize(features.flatten(1), p=2, dim=1)


def train_transform(image_size):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.45, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_image(path):
    return Image.open(path).convert("RGB")


def sample_episode(class_names, data, transform, ways, queries, device):
    selected = random.sample(class_names, ways)
    support_images, query_images, query_labels = [], [], []
    for episode_label, class_name in enumerate(selected):
        path = data[class_name][0]
        image = load_image(path)
        support_images.append(transform(image))
        for _ in range(queries):
            query_images.append(transform(image))
            query_labels.append(episode_label)
    return (
        torch.stack(support_images).to(device),
        torch.stack(query_images).to(device),
        torch.tensor(query_labels, dtype=torch.long, device=device),
    )


def evaluate_split(model, reference_dir, eval_dir, class_names, image_size, batch_size, device):
    transform = eval_transform(image_size)
    ref_features = []
    for class_name in class_names:
        files = sorted(file for file in (Path(reference_dir) / class_name).iterdir() if file.suffix.lower() in IMAGE_EXTENSIONS)
        ref_features.append(model(transform(load_image(files[0])).unsqueeze(0).to(device)).cpu()[0])
    prototypes = F.normalize(torch.stack(ref_features), p=2, dim=1)

    samples = []
    for label, class_name in enumerate(class_names):
        class_dir = Path(eval_dir) / class_name
        if not class_dir.exists():
            continue
        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((file, label, class_name))

    rows, top1, top3 = [], 0, 0
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            images = torch.stack([transform(load_image(path)) for path, _, _ in batch]).to(device)
            features = model(images).cpu()
            scores = features @ prototypes.T
            top_scores, top_indices = torch.topk(scores, k=3, dim=1)
            for i, (path, label, class_name) in enumerate(batch):
                preds = [class_names[j.item()] for j in top_indices[i]]
                top1_ok = top_indices[i, 0].item() == label
                top3_ok = label in top_indices[i].tolist()
                top1 += int(top1_ok)
                top3 += int(top3_ok)
                rows.append(
                    {
                        "image_path": str(path),
                        "true_class": class_name,
                        "top1_class": preds[0],
                        "top1_score": round(top_scores[i, 0].item(), 6),
                        "top2_class": preds[1],
                        "top2_score": round(top_scores[i, 1].item(), 6),
                        "top3_class": preds[2],
                        "top3_score": round(top_scores[i, 2].item(), 6),
                        "top1_correct": bool(top1_ok),
                        "top3_correct": bool(top3_ok),
                    }
                )
    total = len(samples)
    return {
        "top1_accuracy": top1 / total,
        "top3_accuracy": top3 / total,
        "top1_correct": top1,
        "top3_correct": top3,
        "total": total,
        "rows": rows,
    }


def write_split(output_dir, split, result):
    metrics = {
        "split": split,
        "method": "Augmentation-supported Prototypical Network Conv-4",
        "eval_images": result["total"],
        "top1_accuracy": result["top1_accuracy"],
        "top3_accuracy": result["top3_accuracy"],
        "top1_correct": result["top1_correct"],
        "top3_correct": result["top3_correct"],
        "total": result["total"],
    }
    (output_dir / f"{split}_protonet_conv4_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / f"{split}_protonet_conv4_predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(result["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["rows"])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default="dataset_train")
    parser.add_argument("--spatial-dir", default="dataset_val")
    parser.add_argument("--independent-dir", default="my_dataset_test_filtered")
    parser.add_argument("--output-dir", default="results/generated/protonet_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--ways", type=int, default=20)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names, data = load_class_images(args.reference_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProtoNetConv4().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    transform = train_transform(args.image_size)
    start = time.perf_counter()
    model.train()
    log_rows = []
    for episode in range(1, args.episodes + 1):
        support, query, labels = sample_episode(class_names, data, transform, args.ways, args.queries, device)
        support_features = model(support)
        query_features = model(query)
        scores = query_features @ support_features.T
        loss = F.cross_entropy(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if episode == 1 or episode % 50 == 0:
            acc = (scores.argmax(dim=1) == labels).float().mean().item()
            log_rows.append({"episode": episode, "loss": float(loss.item()), "episode_accuracy": acc})
            print(json.dumps(log_rows[-1]))

    model.eval()
    summary = []
    with torch.no_grad():
        for split, eval_dir in [("spatial", args.spatial_dir), ("independent", args.independent_dir)]:
            result = evaluate_split(model, args.reference_dir, eval_dir, class_names, args.image_size, args.batch_size, device)
            metrics = write_split(output_dir, split, result)
            metrics["elapsed_seconds_total"] = round(time.perf_counter() - start, 3)
            summary.append(metrics)
            print(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), output_dir / "protonet_conv4_seed42.pt")
    with (output_dir / "protonet_conv4_training_log.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "loss", "episode_accuracy"])
        writer.writeheader()
        writer.writerows(log_rows)
    with (output_dir / "protonet_conv4_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = ["split", "method", "eval_images", "top1_accuracy", "top3_accuracy", "top1_correct", "top3_correct", "total", "elapsed_seconds_total"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


if __name__ == "__main__":
    main()
