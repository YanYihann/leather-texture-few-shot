import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LeatherDataset(Dataset):
    def __init__(self, dataset_dir, class_names=None, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.class_names = class_names or sorted(item.name for item in self.dataset_dir.iterdir() if item.is_dir())
        self.class_to_label = {name: i for i, name in enumerate(self.class_names)}
        self.transform = transform
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                continue
            for file in sorted(class_dir.iterdir()):
                if file.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((file, self.class_to_label[class_name], class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, class_name, str(image_path)


class TorchvisionFeatureExtractor(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT
            model = models.convnext_tiny(weights=weights)
            model.classifier[-1] = nn.Identity()
        elif name == "efficientnet_v2_s":
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
            model = models.efficientnet_v2_s(weights=weights)
            model.classifier[-1] = nn.Identity()
        elif name == "swin_t":
            weights = models.Swin_T_Weights.DEFAULT
            model = models.swin_t(weights=weights)
            model.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {name}")
        self.model = model.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        features = self.model(images).float()
        return F.normalize(features, p=2, dim=1)


def build_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def extract_features(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features, labels, classes, paths = [], [], [], []
    with torch.no_grad():
        for images, batch_labels, batch_classes, batch_paths in loader:
            features.append(model(images.to(device)).cpu())
            labels.extend(batch_labels.tolist())
            classes.extend(batch_classes)
            paths.extend(batch_paths)
    return {"features": torch.cat(features, dim=0), "labels": labels, "classes": classes, "paths": paths}


def build_prototypes(feature_data, num_classes):
    feature_dim = feature_data["features"].shape[1]
    sums = torch.zeros((num_classes, feature_dim), dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for feature, label in zip(feature_data["features"], feature_data["labels"]):
        sums[label] += feature
        counts[label] += 1
    if torch.any(counts == 0):
        raise ValueError("At least one class has no reference image.")
    return F.normalize(sums / counts.unsqueeze(1), p=2, dim=1)


def evaluate_scores(scores, labels, class_names, eval_data):
    top_scores, top_indices = torch.topk(scores, k=3, dim=1)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    top1_correct = top_indices[:, 0] == label_tensor
    top3_correct = (top_indices == label_tensor[:, None]).any(dim=1)
    rows = []
    for i in range(len(labels)):
        predicted = [class_names[j.item()] for j in top_indices[i]]
        score_values = [round(v.item(), 6) for v in top_scores[i]]
        rows.append(
            {
                "image_path": eval_data["paths"][i],
                "true_class": eval_data["classes"][i],
                "top1_class": predicted[0],
                "top1_score": score_values[0],
                "top2_class": predicted[1],
                "top2_score": score_values[1],
                "top3_class": predicted[2],
                "top3_score": score_values[2],
                "top1_correct": bool(top1_correct[i].item()),
                "top3_correct": bool(top3_correct[i].item()),
            }
        )
    total = len(labels)
    return {
        "top1_accuracy": top1_correct.sum().item() / total,
        "top3_accuracy": top3_correct.sum().item() / total,
        "top1_correct": int(top1_correct.sum().item()),
        "top3_correct": int(top3_correct.sum().item()),
        "total": total,
        "rows": rows,
    }


def evaluate_model(args, model_name, split_name, eval_dir, device):
    transform = build_transform(args.image_size)
    reference = LeatherDataset(args.reference_dir, transform=transform)
    evaluation = LeatherDataset(eval_dir, class_names=reference.class_names, transform=transform)
    model = TorchvisionFeatureExtractor(model_name).to(device)
    start = time.perf_counter()
    ref_data = extract_features(model, reference, args.batch_size, device)
    prototypes = build_prototypes(ref_data, len(reference.class_names))
    eval_data = extract_features(model, evaluation, args.batch_size, device)
    scores = eval_data["features"] @ prototypes.T
    result = evaluate_scores(scores, eval_data["labels"], reference.class_names, eval_data)
    elapsed = time.perf_counter() - start

    method = f"Frozen {model_name} + cosine similarity"
    return {
        "method": method,
        "model": model_name,
        "split": split_name,
        "reference_dir": str(Path(args.reference_dir).resolve()),
        "eval_dir": str(Path(eval_dir).resolve()),
        "num_classes": len(reference.class_names),
        "reference_images": len(reference),
        "eval_images": len(evaluation),
        "top1_accuracy": result["top1_accuracy"],
        "top3_accuracy": result["top3_accuracy"],
        "top1_correct": result["top1_correct"],
        "top3_correct": result["top3_correct"],
        "total": result["total"],
        "elapsed_seconds": elapsed,
        "rows": result["rows"],
    }


def write_result(output_dir, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result['split']}_{result['model']}"
    metrics = {k: v for k, v in result.items() if k != "rows"}
    (output_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / f"{stem}_predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(result["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["rows"])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default="dataset_train")
    parser.add_argument("--spatial-dir", default="dataset_val")
    parser.add_argument("--independent-dir", default="my_dataset_test_filtered")
    parser.add_argument("--output-dir", default="results/generated/additional_backbone_baselines")
    parser.add_argument("--models", nargs="+", default=["convnext_tiny", "efficientnet_v2_s", "swin_t"])
    parser.add_argument("--splits", nargs="+", default=["spatial", "independent"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dirs = {"spatial": args.spatial_dir, "independent": args.independent_dir}
    output_dir = Path(args.output_dir)
    summary = []
    for model_name in args.models:
        for split_name in args.splits:
            result = evaluate_model(args, model_name, split_name, split_dirs[split_name], device)
            metrics = write_result(output_dir, result)
            summary.append(metrics)
            print(json.dumps(metrics, indent=2))

    with (output_dir / "additional_backbone_baselines_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "split",
            "method",
            "eval_images",
            "top1_accuracy",
            "top3_accuracy",
            "top1_correct",
            "top3_correct",
            "total",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({name: row[name] for name in fieldnames})


if __name__ == "__main__":
    main()
