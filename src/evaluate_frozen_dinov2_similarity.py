import argparse
import csv
import json
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


def build_transform(image_size):
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


def extract_features(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_features = []
    all_labels = []
    all_class_names = []
    all_paths = []

    with torch.no_grad():
        for images, labels, class_names, image_paths in loader:
            images = images.to(device)
            features = model(images)
            if isinstance(features, dict):
                features = features.get("x_norm_clstoken", next(iter(features.values())))
            features = F.normalize(features.float(), p=2, dim=1).cpu()
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


def build_prototypes(train_data, class_names):
    feature_dim = train_data["features"].shape[1]
    prototypes = torch.zeros((len(class_names), feature_dim), dtype=torch.float32)
    counts = torch.zeros(len(class_names), dtype=torch.float32)

    for feature, label in zip(train_data["features"], train_data["labels"]):
        prototypes[label] += feature
        counts[label] += 1

    if torch.any(counts == 0):
        missing = [class_names[i] for i, count in enumerate(counts) if count == 0]
        raise ValueError(f"Missing prototype samples for classes: {missing}")

    prototypes = prototypes / counts.unsqueeze(1)
    return F.normalize(prototypes, p=2, dim=1)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transform(args.image_size)
    train_dataset = LeatherImageDataset(args.train_dir, transform=transform)
    val_dataset = LeatherImageDataset(args.val_dir, transform=transform)

    if train_dataset.class_names != val_dataset.class_names:
        raise ValueError("Train and validation class names do not match.")

    model = load_dinov2_model(args.model_name, device)

    start_build = time.perf_counter()
    train_data = extract_features(model, train_dataset, args.batch_size, device)
    prototypes = build_prototypes(train_data, train_dataset.class_names)
    build_seconds = time.perf_counter() - start_build

    prototype_path = Path(args.feature_path)
    prototype_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": f"Frozen {args.model_name} + cosine similarity",
            "model_name": args.model_name,
            "class_names": train_dataset.class_names,
            "prototypes": prototypes,
            "source_image_paths": train_data["image_paths"],
        },
        prototype_path,
    )

    start_eval = time.perf_counter()
    val_data = extract_features(model, val_dataset, args.batch_size, device)
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
        "method": f"Frozen {args.model_name} + cosine similarity",
        "model_name": args.model_name,
        "train_dir": str(Path(args.train_dir).resolve()),
        "val_dir": str(Path(args.val_dir).resolve()),
        "feature_path": str(prototype_path.resolve()),
        "num_classes": len(train_dataset.class_names),
        "train_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "device": str(device),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
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
    metrics_path = output_dir / f"stage3_{args.model_name}_similarity_metrics.json"
    predictions_path = output_dir / f"stage3_{args.model_name}_similarity_predictions.csv"

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with predictions_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved prototypes to {prototype_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset_train")
    parser.add_argument("--val-dir", default="dataset_val")
    parser.add_argument("--model-name", default="dinov2_vits14")
    parser.add_argument("--feature-path", default="features/dinov2_vits14_prototypes.pt")
    parser.add_argument("--output-dir", default="../02_实验结果与生成数据/results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
