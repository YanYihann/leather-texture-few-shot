import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MODEL_DIMS = {
    "Frozen ResNet50 + cosine similarity": 2048,
    "Frozen dinov2_vits14 + cosine similarity": 384,
    "Frozen dinov2_vitb14 + cosine similarity": 768,
    "Frozen dinov2_vitl14 + cosine similarity": 1024,
}


class LabelMappedDataset(Dataset):
    def __init__(self, dataset_dir, class_names, transform):
        self.dataset_dir = Path(dataset_dir)
        self.class_names = class_names
        self.class_to_label = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        self.samples = []
        for class_dir in sorted(item for item in self.dataset_dir.iterdir() if item.is_dir()):
            if class_dir.name not in self.class_to_label:
                raise ValueError(f"Unknown class in dataset: {class_dir.name}")
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
        return self.transform(image), label, class_name, str(image_path)


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


def get_class_names(reference_dir):
    return sorted(item.name for item in Path(reference_dir).iterdir() if item.is_dir())


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(value):
    return f"{100 * value:.2f}%"


def truth(value):
    return str(value).strip().lower() == "true"


def margin(row):
    first = float(row.get("top1_score", row.get("top1_similarity", 0.0)))
    second = float(row.get("top2_score", row.get("top2_similarity", 0.0)))
    return first - second


def summarize_prediction_rows(method, rows):
    top1 = sum(truth(row["top1_correct"]) for row in rows)
    top3 = sum(truth(row["top3_correct"]) for row in rows)
    total = len(rows)
    correct_margins = [margin(row) for row in rows if truth(row["top1_correct"])]
    wrong_margins = [margin(row) for row in rows if not truth(row["top1_correct"])]
    return {
        "Method": method,
        "Feature dim": MODEL_DIMS.get(method, ""),
        "Images": total,
        "Top-1 Acc": pct(top1 / total),
        "Top-3 Acc": pct(top3 / total),
        "Top-1 correct": f"{top1} / {total}",
        "Top-3 correct": f"{top3} / {total}",
        "Mean margin, correct": f"{sum(correct_margins) / len(correct_margins):.4f}" if correct_margins else "",
        "Mean margin, wrong": f"{sum(wrong_margins) / len(wrong_margins):.4f}" if wrong_margins else "",
    }


def make_discrepancy_tables(args):
    independent_dir = Path(args.independent_dir)
    scale_dir = Path(args.scale_dir)
    spatial_dir = Path(args.spatial_dir)
    output_dir = Path(args.output_dir)

    independent_sources = [
        (
            "Frozen ResNet50 + cosine similarity",
            independent_dir / "filtered_frozen_resnet50_similarity_predictions.csv",
        ),
        (
            "Frozen dinov2_vits14 + cosine similarity",
            scale_dir / "filtered_dinov2_vits14_similarity_predictions.csv",
        ),
        (
            "Frozen dinov2_vitb14 + cosine similarity",
            scale_dir / "filtered_dinov2_vitb14_similarity_predictions.csv",
        ),
        (
            "Frozen dinov2_vitl14 + cosine similarity",
            scale_dir / "filtered_dinov2_vitl14_similarity_predictions.csv",
        ),
    ]
    rows_by_method = {method: read_csv(path) for method, path in independent_sources}
    independent_summary = []
    for method, rows in rows_by_method.items():
        item = summarize_prediction_rows(method, rows)
        item["Split"] = "Filtered independent test"
        independent_summary.append(item)

    spatial_sources = [
        (
            "Frozen dinov2_vits14 + cosine similarity",
            spatial_dir / "stage3_dinov2_vits14_similarity_predictions.csv",
        ),
        (
            "Frozen dinov2_vitb14 + cosine similarity",
            spatial_dir / "stage3_dinov2_vitb14_similarity_predictions.csv",
        ),
        (
            "Frozen dinov2_vitl14 + cosine similarity",
            spatial_dir / "stage3_dinov2_vitl14_similarity_predictions.csv",
        ),
    ]
    spatial_summary = []
    for method, path in spatial_sources:
        item = summarize_prediction_rows(method, read_csv(path))
        item["Split"] = "Controlled spatial validation"
        spatial_summary.append(item)

    summary_rows = independent_summary + spatial_summary
    summary_fields = [
        "Split",
        "Method",
        "Feature dim",
        "Images",
        "Top-1 Acc",
        "Top-3 Acc",
        "Top-1 correct",
        "Top-3 correct",
        "Mean margin, correct",
        "Mean margin, wrong",
    ]
    write_csv(output_dir / "resnet_dinov2_discrepancy_summary.csv", summary_rows, summary_fields)

    resnet = {row["true_class"]: row for row in rows_by_method["Frozen ResNet50 + cosine similarity"]}
    dino_s = {row["true_class"]: row for row in rows_by_method["Frozen dinov2_vits14 + cosine similarity"]}
    per_class = []
    for class_name in sorted(resnet):
        r = resnet[class_name]
        d = dino_s[class_name]
        r_ok = truth(r["top1_correct"])
        d_ok = truth(d["top1_correct"])
        if r_ok and d_ok:
            group = "both_correct"
        elif r_ok and not d_ok:
            group = "resnet_only_correct"
        elif (not r_ok) and d_ok:
            group = "dinov2_only_correct"
        else:
            group = "both_wrong"
        per_class.append(
            {
                "Class": class_name,
                "Outcome group": group,
                "ResNet50 top1": r["top1_class"],
                "DINOv2-S top1": d["top1_class"],
                "ResNet50 top1 correct": r["top1_correct"],
                "DINOv2-S top1 correct": d["top1_correct"],
                "ResNet50 margin": f"{margin(r):.6f}",
                "DINOv2-S margin": f"{margin(d):.6f}",
                "ResNet50 top3": "|".join([r["top1_class"], r["top2_class"], r["top3_class"]]),
                "DINOv2-S top3": "|".join([d["top1_class"], d["top2_class"], d["top3_class"]]),
                "Image path": r["image_path"],
            }
        )
    write_csv(
        output_dir / "resnet_dinov2_per_class_outcomes.csv",
        per_class,
        list(per_class[0].keys()),
    )

    counts = {}
    for row in per_class:
        counts[row["Outcome group"]] = counts.get(row["Outcome group"], 0) + 1
    counts_rows = [{"Outcome group": key, "Classes": value} for key, value in sorted(counts.items())]
    write_csv(output_dir / "resnet_dinov2_outcome_counts.csv", counts_rows, ["Outcome group", "Classes"])

    return summary_rows, per_class


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
    return torch.cat(all_features, dim=0), all_labels, all_classes, all_paths


def pca_2d(features):
    centered = features - features.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def scale_points(points, x0, y0, width, height):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    out = []
    for x, y in points:
        sx = x0 + (x - xmin) / (xmax - xmin + 1e-12) * width
        sy = y0 + height - (y - ymin) / (ymax - ymin + 1e-12) * height
        out.append((sx, sy))
    return out


def class_short(name):
    return name.replace("Leather_", "L")


def svg_escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_pca_svg(path, panels):
    width = 1080
    height = 420
    panel_w = 460
    panel_h = 300
    margin_left = 70
    top = 58
    gap = 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,DejaVu Sans,sans-serif;} .axis{stroke:#333;stroke-width:1;} .ref{fill:#b9c0c8;opacity:.42;} .ok{fill:#16875d;opacity:.78;} .bad{fill:#c93f3f;opacity:.9;} .link{stroke:#d0d4d9;stroke-width:.6;opacity:.32;}</style>',
    ]
    for pi, panel in enumerate(panels):
        x0 = margin_left + pi * (panel_w + gap)
        y0 = top
        coords = scale_points(panel["coords"], x0, y0, panel_w, panel_h)
        n_ref = panel["n_ref"]
        parts.append(f'<text x="{x0}" y="28" font-size="18" font-weight="700">{svg_escape(panel["title"])}</text>')
        parts.append(f'<text x="{x0}" y="48" font-size="12" fill="#555">{svg_escape(panel["subtitle"])}</text>')
        parts.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#d8d8d8"/>')
        for i, (sx, sy) in enumerate(coords[:n_ref]):
            parts.append(f'<circle class="ref" cx="{sx:.2f}" cy="{sy:.2f}" r="2.2"/>')
        labels_to_draw = []
        for i, (sx, sy) in enumerate(coords[n_ref:], start=0):
            ref_sx, ref_sy = coords[panel["true_ref_indices"][i]]
            ok = panel["correct"][i]
            if not ok:
                parts.append(f'<line class="link" x1="{ref_sx:.2f}" y1="{ref_sy:.2f}" x2="{sx:.2f}" y2="{sy:.2f}"/>')
            parts.append(f'<circle class="{"ok" if ok else "bad"}" cx="{sx:.2f}" cy="{sy:.2f}" r="{3.2 if ok else 4.2}"/>')
            if not ok:
                labels_to_draw.append((sx + 5, sy - 5, class_short(panel["test_classes"][i])))
        for sx, sy, label in labels_to_draw[:18]:
            parts.append(f'<text x="{sx:.2f}" y="{sy:.2f}" font-size="9" fill="#333">{svg_escape(label)}</text>')
        parts.append(f'<line class="axis" x1="{x0}" y1="{y0 + panel_h + 20}" x2="{x0 + 90}" y2="{y0 + panel_h + 20}"/>')
        parts.append(f'<text x="{x0 + 98}" y="{y0 + panel_h + 24}" font-size="11" fill="#444">PC1</text>')
        parts.append(f'<line class="axis" x1="{x0 - 20}" y1="{y0 + panel_h}" x2="{x0 - 20}" y2="{y0 + panel_h - 90}"/>')
        parts.append(f'<text x="{x0 - 38}" y="{y0 + panel_h - 98}" font-size="11" fill="#444">PC2</text>')
    parts.append('<circle class="ref" cx="70" cy="392" r="4"/><text x="82" y="396" font-size="12">reference prototype</text>')
    parts.append('<circle class="ok" cx="240" cy="392" r="4"/><text x="252" y="396" font-size="12">correct test</text>')
    parts.append('<circle class="bad" cx="360" cy="392" r="4"/><text x="372" y="396" font-size="12">Top-1 error</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_pca_png(path, panels, scale=2):
    width = 1080
    height = 420
    panel_w = 460
    panel_h = 300
    margin_left = 70
    top = 58
    gap = 70
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 12 * scale)
        small = ImageFont.truetype("arial.ttf", 9 * scale)
        title_font = ImageFont.truetype("arialbd.ttf", 18 * scale)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
        title_font = ImageFont.load_default()

    def xy(value):
        return tuple(int(round(v * scale)) for v in value)

    for pi, panel in enumerate(panels):
        x0 = margin_left + pi * (panel_w + gap)
        y0 = top
        coords = scale_points(panel["coords"], x0, y0, panel_w, panel_h)
        n_ref = panel["n_ref"]
        draw.text(xy((x0, 24)), panel["title"], fill=(0, 0, 0), font=title_font)
        draw.text(xy((x0, 45)), panel["subtitle"], fill=(80, 80, 80), font=font)
        draw.rectangle(xy((x0, y0, x0 + panel_w, y0 + panel_h)), outline=(216, 216, 216), width=scale)
        for sx, sy in coords[:n_ref]:
            r = 2.1 * scale
            draw.ellipse((sx * scale - r, sy * scale - r, sx * scale + r, sy * scale + r), fill=(185, 192, 200))
        labels_to_draw = []
        for i, (sx, sy) in enumerate(coords[n_ref:], start=0):
            ref_sx, ref_sy = coords[panel["true_ref_indices"][i]]
            ok = panel["correct"][i]
            if not ok:
                draw.line(xy((ref_sx, ref_sy, sx, sy)), fill=(210, 214, 219), width=scale)
            color = (22, 135, 93) if ok else (201, 63, 63)
            r = (3.2 if ok else 4.2) * scale
            draw.ellipse((sx * scale - r, sy * scale - r, sx * scale + r, sy * scale + r), fill=color)
            if not ok:
                labels_to_draw.append((sx + 5, sy - 5, class_short(panel["test_classes"][i])))
        for sx, sy, label in labels_to_draw[:18]:
            draw.text(xy((sx, sy)), label, fill=(50, 50, 50), font=small)
        draw.line(xy((x0, y0 + panel_h + 20, x0 + 90, y0 + panel_h + 20)), fill=(51, 51, 51), width=scale)
        draw.text(xy((x0 + 98, y0 + panel_h + 12)), "PC1", fill=(68, 68, 68), font=font)
        draw.line(xy((x0 - 20, y0 + panel_h, x0 - 20, y0 + panel_h - 90)), fill=(51, 51, 51), width=scale)
        draw.text(xy((x0 - 38, y0 + panel_h - 110)), "PC2", fill=(68, 68, 68), font=font)

    legend = [(70, "reference prototype", (185, 192, 200)), (240, "correct test", (22, 135, 93)), (360, "Top-1 error", (201, 63, 63))]
    for x, label, color in legend:
        r = 4 * scale
        draw.ellipse((x * scale - r, 392 * scale - r, x * scale + r, 392 * scale + r), fill=color)
        draw.text(xy((x + 12, 384)), label, fill=(0, 0, 0), font=font)
    img.save(path, dpi=(600, 600))


def make_feature_pca(args):
    output_dir = Path(args.output_dir)
    class_names = get_class_names(args.reference_dir)
    class_to_index = {name: i for i, name in enumerate(class_names)}
    transform = eval_transform(args.image_size)
    reference = LabelMappedDataset(args.reference_dir, class_names, transform)
    test = LabelMappedDataset(args.test_dir, class_names, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_to_run = [
        ("Frozen ResNet50", ResNet50FeatureExtractor().to(device), args.resnet_batch_size),
        ("Frozen DINOv2 ViT-S/14", load_dinov2("dinov2_vits14", device), args.dino_batch_size),
    ]
    panels = []
    coordinate_rows = []
    for title, model, batch_size in models_to_run:
        ref_features, _, ref_classes, _ = extract_features(model, reference, batch_size, device)
        test_features, _, test_classes, test_paths = extract_features(model, test, batch_size, device)
        prototypes = F.normalize(ref_features, p=2, dim=1)
        similarities = test_features @ prototypes.T
        top_indices = torch.argmax(similarities, dim=1).tolist()
        correct = [class_names[idx] == cls for idx, cls in zip(top_indices, test_classes)]
        combined = torch.cat([ref_features, test_features], dim=0)
        coords = pca_2d(combined).tolist()
        true_ref_indices = [class_to_index[cls] for cls in test_classes]
        top1 = sum(correct)
        panels.append(
            {
                "title": title,
                "subtitle": f"{top1}/{len(test_classes)} Top-1 correct on filtered independent test",
                "coords": coords,
                "n_ref": len(ref_classes),
                "true_ref_indices": true_ref_indices,
                "correct": correct,
                "test_classes": test_classes,
            }
        )
        for i, cls in enumerate(ref_classes):
            coordinate_rows.append(
                {
                    "Model": title,
                    "Point type": "reference",
                    "Class": cls,
                    "X": f"{coords[i][0]:.8f}",
                    "Y": f"{coords[i][1]:.8f}",
                    "Top-1 correct": "",
                    "Top-1 prediction": "",
                    "Image path": "",
                }
            )
        offset = len(ref_classes)
        for i, cls in enumerate(test_classes):
            coordinate_rows.append(
                {
                    "Model": title,
                    "Point type": "test",
                    "Class": cls,
                    "X": f"{coords[offset + i][0]:.8f}",
                    "Y": f"{coords[offset + i][1]:.8f}",
                    "Top-1 correct": str(correct[i]),
                    "Top-1 prediction": class_names[top_indices[i]],
                    "Image path": test_paths[i],
                }
            )
    write_csv(output_dir / "resnet_dinov2_feature_pca_coordinates.csv", coordinate_rows, list(coordinate_rows[0].keys()))
    write_pca_svg(output_dir / "resnet_dinov2_feature_pca.svg", panels)
    write_pca_png(output_dir / "resnet_dinov2_feature_pca.png", panels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default="dataset_train")
    parser.add_argument("--test-dir", default="my_dataset_test_filtered")
    parser.add_argument("--independent-dir", default="results/results_independent_filtered")
    parser.add_argument("--scale-dir", default="results/results_revision_dinov2_scale")
    parser.add_argument("--spatial-dir", default="results/results_revision_dinov2_scale_spatial")
    parser.add_argument("--output-dir", default="results/generated/revision_tables")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--make-pca", action="store_true")
    parser.add_argument("--resnet-batch-size", type=int, default=32)
    parser.add_argument("--dino-batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    summary_rows, per_class = make_discrepancy_tables(args)
    if args.make_pca:
        make_feature_pca(args)
    counts = {}
    for row in per_class:
        counts[row["Outcome group"]] = counts.get(row["Outcome group"], 0) + 1
    print(json.dumps({"summary_rows": len(summary_rows), "outcome_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
