import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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


class AugmentedLeatherViews(Dataset):
    def __init__(self, dataset_dir, views_per_class, transform=None):
        self.base_dataset = LeatherImageDataset(dataset_dir, transform=None)
        self.views_per_class = views_per_class
        self.transform = transform
        self.class_names = self.base_dataset.class_names
        self.samples = []

        for image_path, label, class_name in self.base_dataset.samples:
            for view_index in range(views_per_class):
                self.samples.append((image_path, label, class_name, view_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, class_name, view_index = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, class_name, str(image_path), view_index


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_transform(image_size):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_eval_transform(image_size):
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


def load_dinov2_model(model_name, device):
    model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name,
        pretrained=True,
        trust_repo=True,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def encode_images(model, images):
    features = model(images)
    if isinstance(features, dict):
        features = features.get("x_norm_clstoken", next(iter(features.values())))
    return F.normalize(features.float(), p=2, dim=1)


def build_augmented_prototypes(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feature_sum = None
    counts = torch.zeros(len(dataset.class_names), dtype=torch.float32)
    source_paths = {}

    with torch.no_grad():
        for images, labels, class_names, image_paths, view_indices in loader:
            images = images.to(device)
            features = encode_images(model, images).cpu()

            if feature_sum is None:
                feature_sum = torch.zeros(
                    (len(dataset.class_names), features.shape[1]),
                    dtype=torch.float32,
                )

            for feature, label, image_path in zip(features, labels.tolist(), image_paths):
                feature_sum[label] += feature
                counts[label] += 1
                source_paths[label] = image_path

    if torch.any(counts == 0):
        missing = [
            dataset.class_names[i] for i, count in enumerate(counts) if count == 0
        ]
        raise ValueError(f"Missing prototype samples for classes: {missing}")

    prototypes = feature_sum / counts.unsqueeze(1)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    ordered_source_paths = [source_paths[i] for i in range(len(dataset.class_names))]
    return prototypes, ordered_source_paths


def extract_eval_features(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []
    all_labels = []
    all_class_names = []
    all_paths = []

    with torch.no_grad():
        for images, labels, class_names, image_paths in loader:
            images = images.to(device)
            features = encode_images(model, images).cpu()
            all_features.append(features)
            all_labels.extend(labels.tolist())
            all_class_names.extend(class_names)
            all_paths.extend(image_paths)

    return {
        "features": torch.cat(all_features, dim=0),
        "labels": all_labels,
        "class_names": all_class_names,
        "image_paths": all_paths,
    }


def evaluate(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = AugmentedLeatherViews(
        args.train_dir,
        views_per_class=args.views_per_class,
        transform=build_train_transform(args.image_size),
    )
    val_dataset = LeatherImageDataset(
        args.val_dir,
        transform=build_eval_transform(args.image_size),
    )

    if train_dataset.class_names != val_dataset.class_names:
        raise ValueError("Train and validation class names do not match.")

    model = load_dinov2_model(args.model_name, device)

    start_build = time.perf_counter()
    prototypes, source_paths = build_augmented_prototypes(
        model,
        train_dataset,
        args.batch_size,
        device,
    )
    build_seconds = time.perf_counter() - start_build

    feature_path = Path(
        args.feature_path
        or f"features/{args.model_name}_augmented_prototypes_N{args.views_per_class}.pt"
    )
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": f"{args.model_name} + augmentation prototypes",
            "model_name": args.model_name,
            "views_per_class": args.views_per_class,
            "class_names": train_dataset.class_names,
            "prototypes": prototypes,
            "source_image_paths": source_paths,
        },
        feature_path,
    )

    start_eval = time.perf_counter()
    val_data = extract_eval_features(model, val_dataset, args.batch_size, device)
    similarities = val_data["features"] @ prototypes.T
    top_scores, top_indices = torch.topk(similarities, k=3, dim=1)
    eval_seconds = time.perf_counter() - start_eval

    total = len(val_dataset)
    labels = torch.tensor(val_data["labels"], dtype=torch.long)
    top1_correct_tensor = top_indices[:, 0] == labels
    top3_correct_tensor = (top_indices == labels.unsqueeze(1)).any(dim=1)
    top1_correct = top1_correct_tensor.sum().item()
    top3_correct = top3_correct_tensor.sum().item()

    rows = []
    for i in range(total):
        predicted = [train_dataset.class_names[index.item()] for index in top_indices[i]]
        scores = [round(score.item(), 6) for score in top_scores[i]]
        rows.append(
            {
                "image_path": val_data["image_paths"][i],
                "true_class": val_data["class_names"][i],
                "top1_class": predicted[0],
                "top1_similarity": scores[0],
                "top2_class": predicted[1],
                "top2_similarity": scores[1],
                "top3_class": predicted[2],
                "top3_similarity": scores[2],
                "top1_correct": bool(top1_correct_tensor[i].item()),
                "top3_correct": bool(top3_correct_tensor[i].item()),
            }
        )

    metrics = {
        "method": f"{args.model_name} + augmentation prototypes",
        "model_name": args.model_name,
        "views_per_class": args.views_per_class,
        "train_dir": str(Path(args.train_dir).resolve()),
        "val_dir": str(Path(args.val_dir).resolve()),
        "feature_path": str(feature_path.resolve()),
        "num_classes": len(train_dataset.class_names),
        "train_source_images": len(train_dataset.base_dataset),
        "prototype_views": len(train_dataset),
        "validation_images": len(val_dataset),
        "device": str(device),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seed": args.seed,
        "top1_accuracy": top1_correct / total,
        "top3_accuracy": top3_correct / total,
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "prototype_build_seconds": build_seconds,
        "evaluation_seconds": eval_seconds,
        "avg_eval_ms_per_image": eval_seconds * 1000 / total,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        output_dir
        / f"stage4_{args.model_name}_augmented_N{args.views_per_class}_metrics.json"
    )
    predictions_path = (
        output_dir
        / f"stage4_{args.model_name}_augmented_N{args.views_per_class}_predictions.csv"
    )

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with predictions_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved prototypes to {feature_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset_train")
    parser.add_argument("--val-dir", default="dataset_val")
    parser.add_argument("--model-name", default="dinov2_vits14")
    parser.add_argument("--views-per-class", type=int, default=10)
    parser.add_argument("--feature-path", default=None)
    parser.add_argument("--output-dir", default="../02_实验结果与生成数据/results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
