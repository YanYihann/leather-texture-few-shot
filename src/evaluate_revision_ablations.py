import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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


def get_class_names(reference_dir):
    return sorted(item.name for item in Path(reference_dir).iterdir() if item.is_dir())


def image_files(path):
    return sorted(file for file in path.iterdir() if file.suffix.lower() in IMAGE_EXTENSIONS)


class LeatherDataset(Dataset):
    def __init__(self, dataset_dir, class_names, transform=None, max_samples=None, return_pil=False):
        self.dataset_dir = Path(dataset_dir)
        self.class_names = class_names
        self.class_to_label = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        self.return_pil = return_pil
        self.samples = []
        for class_name in class_names:
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                continue
            files = image_files(class_dir)
            if max_samples is not None:
                files = files[:max_samples]
            for file in files:
                self.samples.append((file, self.class_to_label[class_name], class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        transformed = self.transform(image) if self.transform else image
        if self.return_pil:
            return transformed, image, label, class_name, str(image_path)
        return transformed, label, class_name, str(image_path)


class AugmentedReferenceDataset(Dataset):
    def __init__(self, dataset_dir, class_names, views_per_class, transform, max_samples=None):
        self.samples = []
        self.transform = transform
        class_to_label = {name: i for i, name in enumerate(class_names)}
        for class_name in class_names:
            files = image_files(Path(dataset_dir) / class_name)
            if max_samples is not None:
                files = files[:max_samples]
            for file in files:
                for view_index in range(views_per_class):
                    self.samples.append((file, class_to_label[class_name], class_name, view_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name, view_index = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label, class_name, str(image_path), view_index


def base_eval_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def reference_transform(name, image_size):
    tensor_norm = [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if name == "center":
        return base_eval_transform(image_size)
    if name == "crop":
        return transforms.Compose([transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)), *tensor_norm])
    if name == "flip":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                *tensor_norm,
            ]
        )
    if name == "color_jitter":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
                *tensor_norm,
            ]
        )
    if name == "crop_flip":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                *tensor_norm,
            ]
        )
    if name == "crop_flip_color":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
                *tensor_norm,
            ]
        )
    raise ValueError(f"Unknown augmentation type: {name}")


def load_dinov2(model_name, device):
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True, trust_repo=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def encode_model(model, images, normalize=True):
    features = model(images)
    if isinstance(features, dict):
        features = features.get("x_norm_clstoken", next(iter(features.values())))
    features = features.float()
    if normalize:
        features = F.normalize(features, p=2, dim=1)
    return features


def extract_features(model, dataset, batch_size, device, normalize=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    labels = []
    classes = []
    paths = []
    with torch.no_grad():
        for images, batch_labels, batch_classes, batch_paths in loader:
            features.append(encode_model(model, images.to(device), normalize=normalize).cpu())
            labels.extend(batch_labels.tolist())
            classes.extend(batch_classes)
            paths.extend(batch_paths)
    return {"features": torch.cat(features, dim=0), "labels": labels, "class_names": classes, "image_paths": paths}


def extract_aug_features(model, dataset, batch_size, device, normalize=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    labels = []
    with torch.no_grad():
        for images, batch_labels, batch_classes, batch_paths, view_indices in loader:
            features.append(encode_model(model, images.to(device), normalize=normalize).cpu())
            labels.extend(batch_labels.tolist())
    return {"features": torch.cat(features, dim=0), "labels": labels}


def build_prototypes(feature_data, num_classes, normalize=True):
    feature_sum = torch.zeros((num_classes, feature_data["features"].shape[1]), dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for feature, label in zip(feature_data["features"], feature_data["labels"]):
        feature_sum[label] += feature
        counts[label] += 1
    if torch.any(counts == 0):
        raise ValueError("At least one class has no prototype image.")
    prototypes = feature_sum / counts.unsqueeze(1)
    if normalize:
        prototypes = F.normalize(prototypes, p=2, dim=1)
    return prototypes


def estimate_inverse_covariance(features, shrinkage=0.1, ridge=1e-4):
    centered = features - features.mean(dim=0, keepdim=True)
    denom = max(centered.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    diag_cov = torch.diag(torch.diag(cov))
    shrunk = (1.0 - shrinkage) * cov + shrinkage * diag_cov
    shrunk = shrunk + ridge * torch.eye(shrunk.shape[0], dtype=shrunk.dtype)
    return torch.linalg.pinv(shrunk)


def compute_scores(query_features, prototypes, metric, inverse_covariance=None):
    if metric == "cosine":
        return query_features @ prototypes.T
    if metric == "euclidean":
        return -torch.cdist(query_features, prototypes, p=2)
    if metric == "mahalanobis":
        if inverse_covariance is None:
            raise ValueError("Mahalanobis metric requires an inverse covariance matrix.")
        diff = query_features[:, None, :] - prototypes[None, :, :]
        distances = torch.einsum("ncd,df,ncf->nc", diff, inverse_covariance, diff)
        return -distances
    raise ValueError(f"Unknown metric: {metric}")


def evaluate_scores(scores, labels):
    top_scores, top_indices = torch.topk(scores, k=3, dim=1)
    top1 = 0
    top3 = 0
    margins = []
    for i, label in enumerate(labels):
        top1 += int(top_indices[i, 0].item() == label)
        top3 += int(label in top_indices[i].tolist())
        margins.append(float(top_scores[i, 0].item() - top_scores[i, 1].item()))
    total = len(labels)
    return top1 / total, top3 / total, float(np.mean(margins)), top1, top3, total


def lbp_histogram(image, image_size):
    gray = image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.uint8)
    center = arr[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        arr[:-2, :-2],
        arr[:-2, 1:-1],
        arr[:-2, 2:],
        arr[1:-1, 2:],
        arr[2:, 2:],
        arr[2:, 1:-1],
        arr[2:, :-2],
        arr[1:-1, :-2],
    ]
    for bit, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << bit)
    hist = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    hist /= max(hist.sum(), 1.0)
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


def extract_lbp(dataset, image_size, num_classes=None):
    if num_classes is None:
        features = []
        labels = []
        for _, image, label, _, _ in DataLoader(dataset, batch_size=None, shuffle=False):
            features.append(lbp_histogram(image, image_size))
            labels.append(label)
        return torch.tensor(np.stack(features), dtype=torch.float32), labels

    sums = np.zeros((num_classes, 256), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, image, label, _, _ in DataLoader(dataset, batch_size=None, shuffle=False):
        sums[label] += lbp_histogram(image, image_size)
        counts[label] += 1
    if np.any(counts == 0):
        raise ValueError("At least one class has no LBP prototype image.")
    prototypes = sums / counts[:, None]
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes = prototypes / np.maximum(norms, 1e-12)
    return torch.tensor(prototypes, dtype=torch.float32)


def run_visual_ablation(args, model, class_names, device):
    rows = []
    eval_data = LeatherDataset(args.eval_dir, class_names, transform=base_eval_transform(args.image_size), max_samples=args.max_samples)
    eval_cache = {}
    for normalize in args.l2_normalization:
        normalize_bool = normalize == "on"
        eval_cache[normalize] = extract_features(model, eval_data, args.batch_size, device, normalize=normalize_bool)

    for seed in args.seeds:
        seed_everything(seed)
        for aug_name in args.augmentation_types:
            for views in args.views:
                ref_dataset = AugmentedReferenceDataset(
                    args.reference_dir,
                    class_names,
                    views,
                    reference_transform(aug_name, args.image_size),
                    max_samples=args.max_samples,
                )
                for normalize in args.l2_normalization:
                    normalize_bool = normalize == "on"
                    ref_features = extract_aug_features(model, ref_dataset, args.batch_size, device, normalize=normalize_bool)
                    prototypes = build_prototypes(ref_features, len(class_names), normalize=normalize_bool)
                    eval_features = eval_cache[normalize]
                    inverse_covariance = None
                    if "mahalanobis" in args.similarity_metrics:
                        inverse_covariance = estimate_inverse_covariance(ref_features["features"])
                    for metric in args.similarity_metrics:
                        scores = compute_scores(eval_features["features"], prototypes, metric, inverse_covariance)
                        top1, top3, margin, top1_count, top3_count, total = evaluate_scores(scores, eval_features["labels"])
                        rows.append(
                            {
                                "experiment": "visual_ablation",
                                "model": args.dinov2_model,
                                "eval_dir": args.eval_dir,
                                "seed": seed,
                                "augmentation_type": aug_name,
                                "views_per_class": views,
                                "similarity_metric": metric,
                                "l2_normalization": normalize,
                                "lbp_weight": "",
                                "top1_accuracy": top1,
                                "top3_accuracy": top3,
                                "mean_margin": margin,
                                "top1_correct": top1_count,
                                "top3_correct": top3_count,
                                "total": total,
                            }
                        )
    return rows


def run_lbp_ablation(args, model, class_names, device):
    rows = []
    ref_visual = LeatherDataset(args.reference_dir, class_names, transform=base_eval_transform(args.image_size), max_samples=args.max_samples)
    eval_visual = LeatherDataset(args.eval_dir, class_names, transform=base_eval_transform(args.image_size), max_samples=args.max_samples)
    ref_pil = LeatherDataset(args.reference_dir, class_names, transform=None, max_samples=args.max_samples, return_pil=True)
    eval_pil = LeatherDataset(args.eval_dir, class_names, transform=None, max_samples=args.max_samples, return_pil=True)

    ref_features = extract_features(model, ref_visual, args.batch_size, device, normalize=True)
    eval_features = extract_features(model, eval_visual, args.batch_size, device, normalize=True)
    visual_prototypes = build_prototypes(ref_features, len(class_names), normalize=True)
    visual_scores = compute_scores(eval_features["features"], visual_prototypes, "cosine")

    lbp_prototypes = extract_lbp(ref_pil, args.image_size, num_classes=len(class_names))
    lbp_features, labels = extract_lbp(eval_pil, args.image_size)
    lbp_scores = lbp_features @ lbp_prototypes.T

    for weight in args.lbp_weights:
        scores = (1.0 - weight) * visual_scores + weight * lbp_scores
        top1, top3, margin, top1_count, top3_count, total = evaluate_scores(scores, labels)
        rows.append(
            {
                "experiment": "lbp_weight_ablation",
                "model": args.dinov2_model,
                "eval_dir": args.eval_dir,
                "seed": "",
                "augmentation_type": "center",
                "views_per_class": 1,
                "similarity_metric": "cosine",
                "l2_normalization": "on",
                "lbp_weight": weight,
                "top1_accuracy": top1,
                "top3_accuracy": top3,
                "mean_margin": margin,
                "top1_correct": top1_count,
                "top3_correct": top3_count,
                "total": total,
            }
        )
    return rows


def write_outputs(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "revision_ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "note": "Ablations use the available validation or filtered test split; they do not create additional real independent images.",
        "rows": rows,
    }
    (output_dir / "revision_ablation_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default="dataset_train")
    parser.add_argument("--eval-dir", default="dataset_val")
    parser.add_argument("--output-dir", default="results/generated/ablation")
    parser.add_argument("--dinov2-model", default="dinov2_vits14")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--views", nargs="+", type=int, default=[1, 5, 10, 20, 30, 50])
    parser.add_argument(
        "--augmentation-types",
        nargs="+",
        default=["center", "crop", "flip", "color_jitter", "crop_flip", "crop_flip_color"],
    )
    parser.add_argument("--similarity-metrics", nargs="+", default=["cosine", "euclidean"])
    parser.add_argument("--l2-normalization", nargs="+", default=["on", "off"], choices=["on", "off"])
    parser.add_argument("--lbp-weights", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--skip-lbp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.perf_counter()
    class_names = get_class_names(args.reference_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(args.dinov2_model, device)
    rows = []
    if not args.skip_visual:
        rows.extend(run_visual_ablation(args, model, class_names, device))
    if not args.skip_lbp:
        rows.extend(run_lbp_ablation(args, model, class_names, device))
    for row in rows:
        row["elapsed_seconds_total"] = round(time.perf_counter() - start, 3)
    write_outputs(Path(args.output_dir), rows)


if __name__ == "__main__":
    main()
