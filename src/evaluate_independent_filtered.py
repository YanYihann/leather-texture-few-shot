import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class LabelMappedDataset(Dataset):
    def __init__(self, dataset_dir, class_names, transform=None, return_pil=False):
        self.dataset_dir = Path(dataset_dir)
        self.class_names = class_names
        self.class_to_label = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        self.return_pil = return_pil
        self.samples = []

        for class_dir in sorted(item for item in self.dataset_dir.iterdir() if item.is_dir()):
            if class_dir.name not in self.class_to_label:
                raise ValueError(f"Unknown class in test set: {class_dir.name}")
            label = self.class_to_label[class_dir.name]
            files = sorted(
                file for file in class_dir.iterdir() if file.suffix.lower() in IMAGE_EXTENSIONS
            )
            for file in files:
                self.samples.append((file, label, class_dir.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.return_pil:
            transformed = self.transform(image) if self.transform else image
            return transformed, image, label, class_name, str(image_path)
        transformed = self.transform(image) if self.transform else image
        return transformed, label, class_name, str(image_path)


class AugmentedReferenceDataset(Dataset):
    def __init__(self, dataset_dir, class_names, views_per_class, transform):
        self.dataset_dir = Path(dataset_dir)
        self.class_names = class_names
        self.class_to_label = {name: i for i, name in enumerate(class_names)}
        self.views_per_class = views_per_class
        self.transform = transform
        self.samples = []

        for class_name in class_names:
            class_dir = self.dataset_dir / class_name
            files = sorted(
                file for file in class_dir.iterdir() if file.suffix.lower() in IMAGE_EXTENSIONS
            )
            for file in files:
                for view_index in range(views_per_class):
                    self.samples.append((file, self.class_to_label[class_name], class_name, view_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name, view_index = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label, class_name, str(image_path), view_index


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_names(reference_dir):
    return sorted(item.name for item in Path(reference_dir).iterdir() if item.is_dir())


def eval_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def aug_transform(image_size):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def load_resnet50_classifier(model_path, num_classes, device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        features = self.features(images)
        features = torch.flatten(features, 1)
        return F.normalize(features.float(), p=2, dim=1)


def load_dinov2(model_name, device):
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True, trust_repo=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def encode_model(model, images):
    features = model(images)
    if isinstance(features, dict):
        features = features.get("x_norm_clstoken", next(iter(features.values())))
    return F.normalize(features.float(), p=2, dim=1)


def extract_features(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []
    all_labels = []
    all_classes = []
    all_paths = []
    with torch.no_grad():
        for images, labels, class_names, image_paths in loader:
            images = images.to(device)
            all_features.append(encode_model(model, images).cpu())
            all_labels.extend(labels.tolist())
            all_classes.extend(class_names)
            all_paths.extend(image_paths)
    return {
        "features": torch.cat(all_features, dim=0),
        "labels": all_labels,
        "class_names": all_classes,
        "image_paths": all_paths,
    }


def build_prototypes(feature_data, num_classes):
    feature_dim = feature_data["features"].shape[1]
    prototypes = torch.zeros((num_classes, feature_dim), dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for feature, label in zip(feature_data["features"], feature_data["labels"]):
        prototypes[label] += feature
        counts[label] += 1
    if torch.any(counts == 0):
        raise ValueError("At least one reference class has no prototype image.")
    return F.normalize(prototypes / counts.unsqueeze(1), p=2, dim=1)


def build_augmented_prototypes(model, dataset, num_classes, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feature_sum = None
    counts = torch.zeros(num_classes, dtype=torch.float32)
    with torch.no_grad():
        for images, labels, class_names, image_paths, view_indices in loader:
            images = images.to(device)
            features = encode_model(model, images).cpu()
            if feature_sum is None:
                feature_sum = torch.zeros((num_classes, features.shape[1]), dtype=torch.float32)
            for feature, label in zip(features, labels.tolist()):
                feature_sum[label] += feature
                counts[label] += 1
    if torch.any(counts == 0):
        raise ValueError("At least one reference class has no augmented prototype image.")
    return F.normalize(feature_sum / counts.unsqueeze(1), p=2, dim=1)


def evaluate_logits(method, outputs, labels, class_names, class_labels, image_paths):
    probabilities = torch.softmax(outputs, dim=1)
    top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)
    rows = []
    top1 = 0
    top3 = 0
    for i, label in enumerate(labels.tolist()):
        predicted = [class_names[index.item()] for index in top_indices[i]]
        scores = [round(prob.item() * 100, 4) for prob in top_probs[i]]
        top1_correct = top_indices[i, 0].item() == label
        top3_correct = label in top_indices[i].tolist()
        top1 += int(top1_correct)
        top3 += int(top3_correct)
        rows.append(
            {
                "image_path": image_paths[i],
                "true_class": class_labels[i],
                "top1_class": predicted[0],
                "top1_score": scores[0],
                "top2_class": predicted[1],
                "top2_score": scores[1],
                "top3_class": predicted[2],
                "top3_score": scores[2],
                "top1_correct": top1_correct,
                "top3_correct": top3_correct,
            }
        )
    return rows, top1, top3


def evaluate_similarity(similarities, labels, class_names, class_labels, image_paths):
    top_scores, top_indices = torch.topk(similarities, k=3, dim=1)
    rows = []
    top1 = 0
    top3 = 0
    for i, label in enumerate(labels):
        predicted = [class_names[index.item()] for index in top_indices[i]]
        scores = [round(score.item(), 6) for score in top_scores[i]]
        top1_correct = top_indices[i, 0].item() == label
        top3_correct = label in top_indices[i].tolist()
        top1 += int(top1_correct)
        top3 += int(top3_correct)
        rows.append(
            {
                "image_path": image_paths[i],
                "true_class": class_labels[i],
                "top1_class": predicted[0],
                "top1_score": scores[0],
                "top2_class": predicted[1],
                "top2_score": scores[1],
                "top3_class": predicted[2],
                "top3_score": scores[2],
                "top1_correct": top1_correct,
                "top3_correct": top3_correct,
            }
        )
    return rows, top1, top3


def save_outputs(output_dir, name, metrics, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{name}_metrics.json"
    predictions_path = output_dir / f"{name}_predictions.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with predictions_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def run_resnet50_classifier(args, class_names, device):
    dataset = LabelMappedDataset(args.test_dir, class_names, transform=eval_transform(args.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = load_resnet50_classifier(args.model_path, len(class_names), device)
    criterion = nn.CrossEntropyLoss()
    rows = []
    total = 0
    top1 = 0
    top3 = 0
    loss_sum = 0.0
    start = time.perf_counter()
    with torch.no_grad():
        for images, labels, class_labels, image_paths in loader:
            images = images.to(device)
            labels_device = labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels_device).item() * labels.size(0)
            batch_rows, batch_top1, batch_top3 = evaluate_logits(
                "resnet50", outputs.cpu(), labels, class_names, class_labels, image_paths
            )
            rows.extend(batch_rows)
            total += labels.size(0)
            top1 += batch_top1
            top3 += batch_top3
    elapsed = time.perf_counter() - start
    metrics = {
        "method": "ResNet50 fine-tuning baseline",
        "reference_classes": len(class_names),
        "test_images": total,
        "top1_accuracy": top1 / total,
        "top3_accuracy": top3 / total,
        "top1_correct": top1,
        "top3_correct": top3,
        "loss": loss_sum / total,
        "device": str(device),
        "avg_eval_ms_per_image": elapsed * 1000 / total,
    }
    save_outputs(Path(args.output_dir), "filtered_resnet50_baseline", metrics, rows)


def run_feature_matching(args, class_names, device, kind):
    reference = LabelMappedDataset(args.reference_dir, class_names, transform=eval_transform(args.image_size))
    test = LabelMappedDataset(args.test_dir, class_names, transform=eval_transform(args.image_size))
    if kind == "resnet50":
        model = ResNet50FeatureExtractor().to(device)
        method = "Frozen ResNet50 + cosine similarity"
        out_name = "filtered_frozen_resnet50_similarity"
    else:
        model = load_dinov2(args.dinov2_model, device)
        method = f"Frozen {args.dinov2_model} + cosine similarity"
        out_name = f"filtered_{args.dinov2_model}_similarity"
    start = time.perf_counter()
    ref_data = extract_features(model, reference, args.batch_size, device)
    prototypes = build_prototypes(ref_data, len(class_names))
    test_data = extract_features(model, test, args.batch_size, device)
    similarities = test_data["features"] @ prototypes.T
    rows, top1, top3 = evaluate_similarity(
        similarities,
        test_data["labels"],
        class_names,
        test_data["class_names"],
        test_data["image_paths"],
    )
    elapsed = time.perf_counter() - start
    total = len(test)
    metrics = {
        "method": method,
        "reference_classes": len(class_names),
        "test_images": total,
        "top1_accuracy": top1 / total,
        "top3_accuracy": top3 / total,
        "top1_correct": top1,
        "top3_correct": top3,
        "device": str(device),
        "avg_total_ms_per_image": elapsed * 1000 / total,
    }
    save_outputs(Path(args.output_dir), out_name, metrics, rows)


def run_dinov2_augmented(args, class_names, device, views):
    seed_everything(args.seed)
    reference = AugmentedReferenceDataset(
        args.reference_dir,
        class_names,
        views_per_class=views,
        transform=aug_transform(args.image_size),
    )
    test = LabelMappedDataset(args.test_dir, class_names, transform=eval_transform(args.image_size))
    model = load_dinov2(args.dinov2_model, device)
    start = time.perf_counter()
    prototypes = build_augmented_prototypes(model, reference, len(class_names), args.batch_size, device)
    test_data = extract_features(model, test, args.batch_size, device)
    similarities = test_data["features"] @ prototypes.T
    rows, top1, top3 = evaluate_similarity(
        similarities,
        test_data["labels"],
        class_names,
        test_data["class_names"],
        test_data["image_paths"],
    )
    elapsed = time.perf_counter() - start
    total = len(test)
    metrics = {
        "method": f"{args.dinov2_model} + augmentation prototypes",
        "views_per_class": views,
        "reference_classes": len(class_names),
        "test_images": total,
        "top1_accuracy": top1 / total,
        "top3_accuracy": top3 / total,
        "top1_correct": top1,
        "top3_correct": top3,
        "device": str(device),
        "avg_total_ms_per_image": elapsed * 1000 / total,
    }
    save_outputs(Path(args.output_dir), f"filtered_{args.dinov2_model}_augmented_N{views}", metrics, rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default="dataset_train")
    parser.add_argument("--test-dir", default="my_dataset_test_filtered")
    parser.add_argument("--output-dir", default="results/generated/independent_filtered")
    parser.add_argument("--model-path", default="best_leather_model_val.pth")
    parser.add_argument("--dinov2-model", default="dinov2_vits14")
    parser.add_argument("--methods", nargs="+", default=["resnet50", "frozen_resnet50", "dinov2", "dinov2_aug10", "dinov2_aug30"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    class_names = get_class_names(args.reference_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "resnet50" in args.methods:
        run_resnet50_classifier(args, class_names, device)
    if "frozen_resnet50" in args.methods:
        run_feature_matching(args, class_names, device, "resnet50")
    if "dinov2" in args.methods:
        run_feature_matching(args, class_names, device, "dinov2")
    if "dinov2_aug10" in args.methods:
        run_dinov2_augmented(args, class_names, device, 10)
    if "dinov2_aug30" in args.methods:
        run_dinov2_augmented(args, class_names, device, 30)


if __name__ == "__main__":
    main()
