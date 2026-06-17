import argparse
import csv
import io
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
SEVERITY_LEVELS = ("mild", "moderate", "severe")
PERTURBATIONS = (
    "brightness",
    "contrast",
    "color_jitter",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg_compression",
    "rotation",
    "crop_scale",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_names(reference_dir):
    return sorted(item.name for item in Path(reference_dir).iterdir() if item.is_dir())


def image_files(class_dir):
    return sorted(file for file in class_dir.iterdir() if file.suffix.lower() in IMAGE_EXTENSIONS)


def eval_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def reference_aug_transform(image_size):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class LabelMappedDataset(Dataset):
    def __init__(self, dataset_dir, class_names, transform=None, max_samples=None):
        self.dataset_dir = Path(dataset_dir)
        self.class_names = class_names
        self.class_to_label = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        self.samples = []

        for class_dir in sorted(item for item in self.dataset_dir.iterdir() if item.is_dir()):
            if class_dir.name not in self.class_to_label:
                raise ValueError(f"Unknown class in dataset: {class_dir.name}")
            files = image_files(class_dir)
            if max_samples is not None:
                files = files[:max_samples]
            for file in files:
                self.samples.append((file, self.class_to_label[class_dir.name], class_dir.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, class_name, str(image_path)


class AugmentedReferenceDataset(Dataset):
    def __init__(self, dataset_dir, class_names, views_per_class, transform):
        self.dataset_dir = Path(dataset_dir)
        self.class_names = class_names
        self.class_to_label = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        self.samples = []

        for class_name in class_names:
            files = image_files(self.dataset_dir / class_name)
            for file in files:
                for view_index in range(views_per_class):
                    self.samples.append((file, self.class_to_label[class_name], class_name, view_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name, view_index = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label, class_name, str(image_path), view_index


def severity_value(severity, mild, moderate, severe):
    return {"mild": mild, "moderate": moderate, "severe": severe}[severity]


def apply_perturbation(image, perturbation, severity, rng):
    if perturbation == "brightness":
        factor = severity_value(severity, 0.85, 0.70, 0.55)
        return ImageEnhance.Brightness(image).enhance(factor)
    if perturbation == "contrast":
        factor = severity_value(severity, 0.80, 0.60, 0.45)
        return ImageEnhance.Contrast(image).enhance(factor)
    if perturbation == "color_jitter":
        color = severity_value(severity, 0.85, 0.65, 0.50)
        bright = severity_value(severity, 1.10, 1.20, 1.35)
        return ImageEnhance.Brightness(ImageEnhance.Color(image).enhance(color)).enhance(bright)
    if perturbation == "gaussian_blur":
        radius = severity_value(severity, 0.8, 1.6, 2.4)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    if perturbation == "gaussian_noise":
        sigma = severity_value(severity, 8.0, 16.0, 28.0)
        arr = np.asarray(image).astype(np.float32)
        arr = np.clip(arr + rng.normal(0.0, sigma, size=arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    if perturbation == "jpeg_compression":
        quality = severity_value(severity, 70, 45, 25)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if perturbation == "rotation":
        degrees = severity_value(severity, 5, 10, 15)
        direction = -1 if rng.random() < 0.5 else 1
        return image.rotate(direction * degrees, resample=Image.Resampling.BILINEAR, expand=False)
    if perturbation == "crop_scale":
        scale = severity_value(severity, 0.90, 0.80, 0.70)
        width, height = image.size
        crop_w = max(1, int(width * scale))
        crop_h = max(1, int(height * scale))
        left = int(rng.integers(0, max(1, width - crop_w + 1)))
        top = int(rng.integers(0, max(1, height - crop_h + 1)))
        return image.crop((left, top, left + crop_w, top + crop_h)).resize(
            (width, height), Image.Resampling.BICUBIC
        )
    raise ValueError(f"Unknown perturbation: {perturbation}")


class PerturbedTestDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        class_names,
        perturbations,
        severities,
        transform,
        seed,
        variants_per_image=1,
        max_samples=None,
    ):
        self.base = LabelMappedDataset(dataset_dir, class_names, transform=None, max_samples=max_samples)
        self.transform = transform
        self.seed = seed
        self.samples = []
        for image_path, label, class_name in self.base.samples:
            for perturbation in perturbations:
                for severity in severities:
                    for variant_index in range(variants_per_image):
                        self.samples.append(
                            (image_path, label, class_name, perturbation, severity, variant_index)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name, perturbation, severity, variant_index = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        rng = np.random.default_rng(abs(hash((str(image_path), perturbation, severity, variant_index, self.seed))) % (2**32))
        image = apply_perturbation(image, perturbation, severity, rng)
        return (
            self.transform(image),
            label,
            class_name,
            str(image_path),
            perturbation,
            severity,
            variant_index,
        )


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
            features = encode_model(model, images.to(device)).cpu()
            all_features.append(features)
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
    feature_sum = torch.zeros((num_classes, feature_data["features"].shape[1]), dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for feature, label in zip(feature_data["features"], feature_data["labels"]):
        feature_sum[label] += feature
        counts[label] += 1
    if torch.any(counts == 0):
        raise ValueError("At least one reference class has no prototype image.")
    return F.normalize(feature_sum / counts.unsqueeze(1), p=2, dim=1)


def build_augmented_prototypes(model, dataset, num_classes, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feature_sum = None
    counts = torch.zeros(num_classes, dtype=torch.float32)
    with torch.no_grad():
        for images, labels, class_names, image_paths, view_indices in loader:
            features = encode_model(model, images.to(device)).cpu()
            if feature_sum is None:
                feature_sum = torch.zeros((num_classes, features.shape[1]), dtype=torch.float32)
            for feature, label in zip(features, labels.tolist()):
                feature_sum[label] += feature
                counts[label] += 1
    if torch.any(counts == 0):
        raise ValueError("At least one reference class has no augmented prototype image.")
    return F.normalize(feature_sum / counts.unsqueeze(1), p=2, dim=1)


def bootstrap_ci(values, num_bootstrap=2000, seed=123):
    if not values:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float32)
    means = []
    for _ in range(num_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["method"], row["perturbation"], row["severity"])
        groups[key].append(row)

    summary = []
    for (method, perturbation, severity), group_rows in sorted(groups.items()):
        source_top1 = defaultdict(list)
        source_top3 = defaultdict(list)
        for row in group_rows:
            source_top1[row["source_image_path"]].append(1.0 if row["top1_correct"] else 0.0)
            source_top3[row["source_image_path"]].append(1.0 if row["top3_correct"] else 0.0)
        top1_grouped = [float(np.mean(values)) for values in source_top1.values()]
        top3_grouped = [float(np.mean(values)) for values in source_top3.values()]
        summary.append(
            {
                "method": method,
                "perturbation": perturbation,
                "severity": severity,
                "source_images": len(source_top1),
                "augmented_variants": len(group_rows),
                "top1_accuracy": float(np.mean(top1_grouped)),
                "top3_accuracy": float(np.mean(top3_grouped)),
                "top1_ci95": bootstrap_ci(top1_grouped),
                "top3_ci95": bootstrap_ci(top3_grouped),
            }
        )
    return summary


def evaluate_method(args, method, seed, class_names, device):
    seed_everything(seed)
    if method == "frozen_resnet50":
        model = ResNet50FeatureExtractor().to(device)
        method_label = "Frozen ResNet50 + cosine similarity"
        reference = LabelMappedDataset(args.reference_dir, class_names, transform=eval_transform(args.image_size))
        prototypes = build_prototypes(extract_features(model, reference, args.batch_size, device), len(class_names))
    elif method == "dinov2":
        model = load_dinov2(args.dinov2_model, device)
        method_label = f"Frozen {args.dinov2_model} + cosine similarity"
        reference = LabelMappedDataset(args.reference_dir, class_names, transform=eval_transform(args.image_size))
        prototypes = build_prototypes(extract_features(model, reference, args.batch_size, device), len(class_names))
    elif method.startswith("dinov2_aug"):
        views = int(method.replace("dinov2_aug", ""))
        model = load_dinov2(args.dinov2_model, device)
        method_label = f"{args.dinov2_model} + augmentation prototypes, N={views}"
        reference = AugmentedReferenceDataset(
            args.reference_dir,
            class_names,
            views_per_class=views,
            transform=reference_aug_transform(args.image_size),
        )
        prototypes = build_augmented_prototypes(model, reference, len(class_names), args.batch_size, device)
    else:
        raise ValueError(f"Unsupported method: {method}")

    test = PerturbedTestDataset(
        args.test_dir,
        class_names,
        args.perturbations,
        args.severities,
        eval_transform(args.image_size),
        seed=seed,
        variants_per_image=args.variants_per_image,
        max_samples=args.max_samples,
    )
    loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    rows = []
    with torch.no_grad():
        for images, labels, class_labels, image_paths, perturbations, severities, variant_indices in loader:
            features = encode_model(model, images.to(device)).cpu()
            similarities = features @ prototypes.T
            top_scores, top_indices = torch.topk(similarities, k=3, dim=1)
            for i, label in enumerate(labels.tolist()):
                predicted = [class_names[index.item()] for index in top_indices[i]]
                score_values = [float(score.item()) for score in top_scores[i]]
                rows.append(
                    {
                        "method": method_label,
                        "seed": seed,
                        "source_image_path": image_paths[i],
                        "true_class": class_labels[i],
                        "perturbation": perturbations[i],
                        "severity": severities[i],
                        "variant_index": int(variant_indices[i]),
                        "top1_class": predicted[0],
                        "top3_classes": "|".join(predicted),
                        "top1_correct": top_indices[i, 0].item() == label,
                        "top3_correct": label in top_indices[i].tolist(),
                        "top1_score": round(score_values[0], 6),
                        "margin": round(score_values[0] - score_values[1], 6),
                    }
                )
    return rows


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default="dataset_train")
    parser.add_argument("--test-dir", default="my_dataset_test_filtered")
    parser.add_argument("--output-dir", default="results/generated/robustness")
    parser.add_argument("--methods", nargs="+", default=["frozen_resnet50", "dinov2", "dinov2_aug30"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS), choices=list(PERTURBATIONS))
    parser.add_argument("--severities", nargs="+", default=list(SEVERITY_LEVELS), choices=list(SEVERITY_LEVELS))
    parser.add_argument("--dinov2-model", default="dinov2_vits14")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--variants-per-image", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = get_class_names(args.reference_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows = []
    incremental_path = output_dir / "augmentation_robustness_predictions_partial.csv"
    for seed in args.seeds:
        for method in args.methods:
            method_rows = evaluate_method(args, method, seed, class_names, device)
            append_csv(incremental_path, method_rows)
            all_rows.extend(method_rows)

    summary = {
        "note": "Synthetic perturbations are augmentation-based robustness tests, not additional independent test images.",
        "reference_dir": str(Path(args.reference_dir).resolve()),
        "test_dir": str(Path(args.test_dir).resolve()),
        "methods": args.methods,
        "seeds": args.seeds,
        "perturbations": args.perturbations,
        "severities": args.severities,
        "variants_per_image": args.variants_per_image,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - start,
        "results": summarize_rows(all_rows),
    }
    write_csv(output_dir / "augmentation_robustness_predictions.csv", all_rows)
    (output_dir / "augmentation_robustness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
