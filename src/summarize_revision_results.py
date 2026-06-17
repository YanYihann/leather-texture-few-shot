import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METHOD_LABELS = {
    "filtered_resnet50_baseline": "ResNet50 fine-tuning baseline",
    "filtered_frozen_resnet50_similarity": "Frozen ResNet50 + cosine similarity",
    "filtered_dinov2_vits14_similarity": "Frozen DINOv2 ViT-S/14 + cosine similarity",
    "filtered_dinov2_vits14_augmented_N10": "DINOv2 augmented prototypes, N=10",
    "filtered_dinov2_vits14_augmented_N30": "DINOv2 augmented prototypes, N=30",
}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def percent(value):
    return f"{100 * float(value):.2f}%"


def bootstrap_ci(values, num_bootstrap=5000, seed=123):
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float32)
    means = np.empty(num_bootstrap, dtype=np.float32)
    for index in range(num_bootstrap):
        means[index] = rng.choice(arr, size=len(arr), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def grouped_accuracy_ci(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        source = row.get("image_path") or row.get("source_image_path")
        grouped[source].append(1.0 if str(row[key]).lower() == "true" else 0.0)
    values = [float(np.mean(items)) for items in grouped.values()]
    mean = float(np.mean(values)) if values else 0.0
    low, high = bootstrap_ci(values)
    return mean, low, high, len(values)


def independent_table(independent_dir):
    rows = []
    for stem, label in METHOD_LABELS.items():
        prediction_path = independent_dir / f"{stem}_predictions.csv"
        metrics_path = independent_dir / f"{stem}_metrics.json"
        if not prediction_path.exists():
            continue
        predictions = read_csv(prediction_path)
        top1, top1_low, top1_high, source_count = grouped_accuracy_ci(predictions, "top1_correct")
        top3, top3_low, top3_high, _ = grouped_accuracy_ci(predictions, "top3_correct")
        correct_top1 = sum(1 for row in predictions if str(row["top1_correct"]).lower() == "true")
        correct_top3 = sum(1 for row in predictions if str(row["top3_correct"]).lower() == "true")
        metric_method = label
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metric_method = metrics.get("method", label)
        rows.append(
            {
                "Method": metric_method,
                "Real source images": source_count,
                "Top-1 Acc": percent(top1),
                "Top-1 95% CI": f"[{percent(top1_low)}, {percent(top1_high)}]",
                "Top-3 Acc": percent(top3),
                "Top-3 95% CI": f"[{percent(top3_low)}, {percent(top3_high)}]",
                "Correct Top-1": f"{correct_top1} / {len(predictions)}",
                "Correct Top-3": f"{correct_top3} / {len(predictions)}",
            }
        )
    return rows


def robustness_table(revision_dir):
    summary_path = revision_dir / "augmentation_robustness_summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for item in summary.get("results", []):
        rows.append(
            {
                "Method": item["method"],
                "Perturbation": item["perturbation"],
                "Severity": item["severity"],
                "Real source images": item["source_images"],
                "Synthetic variants": item["augmented_variants"],
                "Top-1 Acc": percent(item["top1_accuracy"]),
                "Top-1 95% CI": f"[{percent(item['top1_ci95'][0])}, {percent(item['top1_ci95'][1])}]",
                "Top-3 Acc": percent(item["top3_accuracy"]),
                "Top-3 95% CI": f"[{percent(item['top3_ci95'][0])}, {percent(item['top3_ci95'][1])}]",
            }
        )
    return rows


def metric_rows_from_patterns(results_dir, patterns, fields):
    rows = []
    for path in sorted(results_dir.glob("*_metrics.json")):
        if not any(pattern in path.name for pattern in patterns):
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = {}
        for output_name, metric_key in fields.items():
            value = metrics.get(metric_key, "")
            if isinstance(value, float) and "accuracy" in metric_key:
                value = percent(value)
            row[output_name] = value
        rows.append(row)
    return rows


def ablation_tables(results_dir):
    prototype_rows = metric_rows_from_patterns(
        results_dir,
        ["augmented_N"],
        {
            "Method": "method",
            "Views": "views_per_class",
            "Top-1 Acc": "top1_accuracy",
            "Top-3 Acc": "top3_accuracy",
            "Seed": "seed",
        },
    )
    lbp_rows = metric_rows_from_patterns(
        results_dir,
        ["lbp"],
        {
            "Method": "method",
            "Views": "views_per_class",
            "LBP weight": "lbp_weight",
            "Top-1 Acc": "top1_accuracy",
            "Top-3 Acc": "top3_accuracy",
            "Seed": "seed",
        },
    )
    return prototype_rows, lbp_rows


def revision_ablation_tables(revision_dir):
    path = revision_dir / "revision_ablation_results.csv"
    if not path.exists():
        return [], [], []
    rows = read_csv(path)
    prototype_rows = []
    similarity_rows = []
    lbp_rows = []
    for row in rows:
        experiment = row.get("experiment", "")
        if experiment == "visual_ablation":
            clean = {
                "Model": row["model"],
                "Eval split": row["eval_dir"],
                "Seed": row["seed"],
                "Augmentation": row["augmentation_type"],
                "Views": row["views_per_class"],
                "Similarity": row["similarity_metric"],
                "L2 norm": row["l2_normalization"],
                "Top-1 Acc": percent(row["top1_accuracy"]),
                "Top-3 Acc": percent(row["top3_accuracy"]),
                "Mean margin": f"{float(row['mean_margin']):.4f}",
            }
            prototype_rows.append(clean)
            if row["views_per_class"] == "1" and row["augmentation_type"] == "center":
                similarity_rows.append(clean)
        elif experiment == "lbp_weight_ablation":
            lbp_rows.append(
                {
                    "Model": row["model"],
                    "Eval split": row["eval_dir"],
                    "LBP weight": row["lbp_weight"],
                    "Top-1 Acc": percent(row["top1_accuracy"]),
                    "Top-3 Acc": percent(row["top3_accuracy"]),
                    "Mean margin": f"{float(row['mean_margin']):.4f}",
                }
            )
    return prototype_rows, similarity_rows, lbp_rows


def write_revision_note(path):
    path.write_text(
        "\n".join(
            [
                "# Revision Result Summary",
                "",
                "The filtered independent test remains a 201-real-image evaluation. "
                "Any augmentation-based variants are summarized only as synthetic perturbation "
                "robustness tests and must not be described as additional independent samples.",
                "",
                "Recommended manuscript wording: the results provide preliminary evidence on the "
                "current filtered independent test set and require larger real multi-capture validation.",
                "",
                "Reviewer response sentence:",
                "",
                "We agree that the 201-image real independent test set is insufficient for broad "
                "generalization claims. Since collecting a 2000-image real independent test set is "
                "not feasible at this stage, we substantially temper the claims and reposition the "
                "study as a proof-of-concept. We add augmentation-based robustness evaluation, grouped "
                "statistical reporting, ablation studies, and expanded limitations to clarify what "
                "the current evidence can and cannot support.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/results")
    parser.add_argument("--independent-dir", default="results/results_independent_filtered")
    parser.add_argument("--revision-dir", default="results/results_revision")
    parser.add_argument("--output-dir", default="results/generated/revision_tables")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    independent_dir = Path(args.independent_dir)
    revision_dir = Path(args.revision_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    independent_rows = independent_table(independent_dir)
    if independent_rows:
        write_csv(output_dir / "real_independent_test_with_ci.csv", independent_rows)
        write_markdown_table(
            output_dir / "real_independent_test_with_ci.md",
            independent_rows,
            list(independent_rows[0].keys()),
        )

    robust_rows = robustness_table(revision_dir)
    if robust_rows:
        write_csv(output_dir / "augmentation_robustness_by_condition.csv", robust_rows)
        write_markdown_table(
            output_dir / "augmentation_robustness_by_condition.md",
            robust_rows,
            list(robust_rows[0].keys()),
        )

    prototype_rows, lbp_rows = ablation_tables(results_dir)
    if prototype_rows:
        write_csv(output_dir / "prototype_views_ablation_available.csv", prototype_rows)
        write_markdown_table(
            output_dir / "prototype_views_ablation_available.md",
            prototype_rows,
            list(prototype_rows[0].keys()),
        )
    if lbp_rows:
        write_csv(output_dir / "lbp_ablation_available.csv", lbp_rows)
        write_markdown_table(output_dir / "lbp_ablation_available.md", lbp_rows, list(lbp_rows[0].keys()))

    revision_proto_rows, revision_similarity_rows, revision_lbp_rows = revision_ablation_tables(revision_dir)
    if revision_proto_rows:
        write_csv(output_dir / "revision_visual_ablation.csv", revision_proto_rows)
        write_markdown_table(
            output_dir / "revision_visual_ablation.md",
            revision_proto_rows,
            list(revision_proto_rows[0].keys()),
        )
    if revision_similarity_rows:
        write_csv(output_dir / "revision_similarity_ablation.csv", revision_similarity_rows)
        write_markdown_table(
            output_dir / "revision_similarity_ablation.md",
            revision_similarity_rows,
            list(revision_similarity_rows[0].keys()),
        )
    if revision_lbp_rows:
        write_csv(output_dir / "revision_lbp_weight_ablation.csv", revision_lbp_rows)
        write_markdown_table(
            output_dir / "revision_lbp_weight_ablation.md",
            revision_lbp_rows,
            list(revision_lbp_rows[0].keys()),
        )

    write_revision_note(output_dir / "revision_response_note.md")
    print(f"Wrote revision tables to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
