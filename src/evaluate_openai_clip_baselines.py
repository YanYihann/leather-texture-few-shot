import argparse
import csv
import json
import time
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class LeatherDataset(Dataset):
    def __init__(self, dataset_dir, class_names=None, preprocess=None):
        self.dataset_dir = Path(dataset_dir)
        self.class_names = class_names or sorted(item.name for item in self.dataset_dir.iterdir() if item.is_dir())
        self.class_to_label = {name: i for i, name in enumerate(self.class_names)}
        self.preprocess = preprocess
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
        return self.preprocess(image), label, class_name, str(image_path)


def encode_images(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features, labels, classes, paths = [], [], [], []
    with torch.no_grad():
        for images, batch_labels, batch_classes, batch_paths in loader:
            feature = model.encode_image(images.to(device)).float()
            features.append(F.normalize(feature, p=2, dim=1).cpu())
            labels.extend(batch_labels.tolist())
            classes.extend(batch_classes)
            paths.extend(batch_paths)
    return {"features": torch.cat(features, dim=0), "labels": labels, "classes": classes, "paths": paths}


def build_image_prototypes(data, num_classes):
    dim = data["features"].shape[1]
    sums = torch.zeros((num_classes, dim), dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for feature, label in zip(data["features"], data["labels"]):
        sums[label] += feature
        counts[label] += 1
    return F.normalize(sums / counts.unsqueeze(1), p=2, dim=1)


def encode_text_prototypes(model, class_names, device):
    prompts = [f"a close-up leather texture sample named {name.replace('_', ' ')}" for name in class_names]
    tokens = clip.tokenize(prompts, truncate=True).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens).float()
    return F.normalize(features, p=2, dim=1).cpu(), prompts


def evaluate(scores, labels, class_names, data):
    top_scores, top_indices = torch.topk(scores, k=3, dim=1)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    top1_correct = top_indices[:, 0] == label_tensor
    top3_correct = (top_indices == label_tensor[:, None]).any(dim=1)
    rows = []
    for i in range(len(labels)):
        predicted = [class_names[j.item()] for j in top_indices[i]]
        rows.append(
            {
                "image_path": data["paths"][i],
                "true_class": data["classes"][i],
                "top1_class": predicted[0],
                "top1_score": round(top_scores[i, 0].item(), 6),
                "top2_class": predicted[1],
                "top2_score": round(top_scores[i, 1].item(), 6),
                "top3_class": predicted[2],
                "top3_score": round(top_scores[i, 2].item(), 6),
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


def write_result(output_dir, split, method, result):
    safe = method.lower().replace(" ", "_").replace("/", "_").replace("+", "plus")
    metrics = {
        "split": split,
        "method": method,
        "eval_images": result["total"],
        "top1_accuracy": result["top1_accuracy"],
        "top3_accuracy": result["top3_accuracy"],
        "top1_correct": result["top1_correct"],
        "top3_correct": result["top3_correct"],
        "total": result["total"],
    }
    (output_dir / f"{split}_{safe}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / f"{split}_{safe}_predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(result["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["rows"])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default="dataset_train")
    parser.add_argument("--spatial-dir", default="dataset_val")
    parser.add_argument("--independent-dir", default="my_dataset_test_filtered")
    parser.add_argument("--output-dir", default="results/generated/openai_clip_baselines")
    parser.add_argument("--model", default="ViT-B/32")
    parser.add_argument("--splits", nargs="+", default=["spatial", "independent"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(args.model, device=device)
    model.eval()
    reference = LeatherDataset(args.reference_dir, preprocess=preprocess)
    ref_data = encode_images(model, reference, args.batch_size, device)
    image_prototypes = build_image_prototypes(ref_data, len(reference.class_names))
    text_prototypes, prompts = encode_text_prototypes(model, reference.class_names, device)
    split_dirs = {"spatial": args.spatial_dir, "independent": args.independent_dir}
    summary, start = [], time.perf_counter()
    for split in args.splits:
        eval_data = encode_images(model, LeatherDataset(split_dirs[split], class_names=reference.class_names, preprocess=preprocess), args.batch_size, device)
        for method, prototypes in [
            ("CLIP ViT-B/32 zero-shot text prompts", text_prototypes),
            ("CLIP ViT-B/32 one-image prototype", image_prototypes),
        ]:
            result = evaluate(eval_data["features"] @ prototypes.T, eval_data["labels"], reference.class_names, eval_data)
            metrics = write_result(output_dir, split, method, result)
            metrics["elapsed_seconds_total"] = round(time.perf_counter() - start, 3)
            summary.append(metrics)
            print(json.dumps(metrics, indent=2))
    with (output_dir / "openai_clip_baselines_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = ["split", "method", "eval_images", "top1_accuracy", "top3_accuracy", "top1_correct", "top3_correct", "total", "elapsed_seconds_total"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    (output_dir / "clip_zero_shot_prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
