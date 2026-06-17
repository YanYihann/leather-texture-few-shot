import csv
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results"
REVISION_TABLES = RESULT_ROOT / "revision_tables"
ROBUSTNESS_CSV = REVISION_TABLES / "augmentation_robustness_by_condition.csv"
PREDICTION_ROOT = RESULT_ROOT / "results_independent_filtered_multiseed" / "seed_42"


PREDICTION_FILES = {
    "Frozen ResNet50": PREDICTION_ROOT / "filtered_frozen_resnet50_similarity_predictions.csv",
    "Frozen DINOv2-S": PREDICTION_ROOT / "filtered_dinov2_vits14_similarity_predictions.csv",
    "DINOv2-Aug30": PREDICTION_ROOT / "filtered_dinov2_vits14_augmented_N30_predictions.csv",
}


ROBUSTNESS_METHOD_MAP = {
    "Frozen ResNet50 + cosine similarity": "Frozen ResNet50",
    "Frozen dinov2_vits14 + cosine similarity": "Frozen DINOv2-S",
    "dinov2_vits14 + augmentation prototypes, N=30": "DINOv2-Aug30",
}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct_to_float(value):
    return float(value.strip().replace("%", "")) / 100.0


def fmt_pct(value):
    return f"{100.0 * value:.2f}%"


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize_robustness():
    rows = read_csv(ROBUSTNESS_CSV)
    grouped = {}
    for row in rows:
        method = ROBUSTNESS_METHOD_MAP.get(row["Method"], row["Method"])
        severity = row["Severity"]
        grouped.setdefault((method, severity), []).append(row)

    summary = []
    for (method, severity), group in sorted(grouped.items()):
        top1 = [pct_to_float(row["Top-1 Acc"]) for row in group]
        top3 = [pct_to_float(row["Top-3 Acc"]) for row in group]
        worst_row = min(group, key=lambda item: pct_to_float(item["Top-1 Acc"]))
        summary.append(
            {
                "Method": method,
                "Severity": severity,
                "Completed perturbations": len(group),
                "Mean Top-1": fmt_pct(mean(top1)),
                "Worst Top-1": fmt_pct(pct_to_float(worst_row["Top-1 Acc"])),
                "Worst perturbation": worst_row["Perturbation"],
                "Mean Top-3": fmt_pct(mean(top3)),
            }
        )
    write_csv(REVISION_TABLES / "robustness_compact_summary.csv", summary)
    return summary


def load_prediction_rows(path):
    rows = read_csv(path)
    for row in rows:
        row["top1_score_float"] = float(row["top1_score"])
        row["top2_score_float"] = float(row["top2_score"])
        row["margin_float"] = row["top1_score_float"] - row["top2_score_float"]
        row["top1_correct_bool"] = str(row["top1_correct"]).lower() == "true"
    return rows


def score_if_class_removed(row):
    if row["top1_class"] == row["true_class"]:
        return row["top2_score_float"]
    return row["top1_score_float"]


def margin_if_class_removed(row):
    if row["top1_class"] == row["true_class"]:
        return max(0.0, row["top2_score_float"] - float(row["top3_score"]))
    return row["margin_float"]


def summarize_open_set(num_splits=10, unknown_classes_per_split=20, seed=20260618):
    random.seed(seed)
    rows_out = []
    detail_rows = []
    for method, path in PREDICTION_FILES.items():
        rows = load_prediction_rows(path)
        classes = sorted(row["true_class"] for row in rows)
        if unknown_classes_per_split >= len(classes):
            raise ValueError("unknown_classes_per_split must be smaller than the number of classes")

        split_metrics = []
        for split_id in range(num_splits):
            unknown_classes = set(random.sample(classes, unknown_classes_per_split))
            known_rows = [row for row in rows if row["true_class"] not in unknown_classes]
            unknown_rows = [row for row in rows if row["true_class"] in unknown_classes]

            # Calibrate on known queries only: retain roughly 95% of known samples.
            score_threshold = percentile([row["top1_score_float"] for row in known_rows], 0.05)
            margin_threshold = percentile([row["margin_float"] for row in known_rows], 0.05)

            known_accept_score = [row["top1_score_float"] >= score_threshold for row in known_rows]
            known_accept_margin = [row["margin_float"] >= margin_threshold for row in known_rows]
            unknown_reject_score = [
                score_if_class_removed(row) < score_threshold for row in unknown_rows
            ]
            unknown_reject_margin = [
                margin_if_class_removed(row) < margin_threshold for row in unknown_rows
            ]
            known_correct_and_accepted = [
                row["top1_correct_bool"] and (row["top1_score_float"] >= score_threshold)
                for row in known_rows
            ]

            metrics = {
                "method": method,
                "split": split_id,
                "unknown_classes": "|".join(sorted(unknown_classes)),
                "known_classes": len(known_rows),
                "unknown_queries": len(unknown_rows),
                "score_threshold": score_threshold,
                "margin_threshold": margin_threshold,
                "known_acceptance_score": mean(known_accept_score),
                "known_acceptance_margin": mean(known_accept_margin),
                "unknown_rejection_score": mean(unknown_reject_score),
                "unknown_rejection_margin": mean(unknown_reject_margin),
                "known_top1_and_accepted_score": mean(known_correct_and_accepted),
            }
            split_metrics.append(metrics)
            detail_rows.append(
                {
                    "Method": method,
                    "Split": split_id,
                    "Unknown classes": metrics["unknown_classes"],
                    "Score threshold": f"{metrics['score_threshold']:.6f}",
                    "Margin threshold": f"{metrics['margin_threshold']:.6f}",
                    "Known acceptance by score": fmt_pct(metrics["known_acceptance_score"]),
                    "Unknown rejection by score": fmt_pct(metrics["unknown_rejection_score"]),
                    "Known acceptance by margin": fmt_pct(metrics["known_acceptance_margin"]),
                    "Unknown rejection by margin": fmt_pct(metrics["unknown_rejection_margin"]),
                }
            )

        rows_out.append(
            {
                "Method": method,
                "Splits": num_splits,
                "Pseudo-unknown classes per split": unknown_classes_per_split,
                "Known queries per split": len(rows) - unknown_classes_per_split,
                "Pseudo-unknown queries per split": unknown_classes_per_split,
                "Score-threshold known acceptance": fmt_pct(mean([m["known_acceptance_score"] for m in split_metrics])),
                "Score-threshold unknown rejection": fmt_pct(mean([m["unknown_rejection_score"] for m in split_metrics])),
                "Margin-threshold known acceptance": fmt_pct(mean([m["known_acceptance_margin"] for m in split_metrics])),
                "Margin-threshold unknown rejection": fmt_pct(mean([m["unknown_rejection_margin"] for m in split_metrics])),
                "Known Top-1 and accepted by score": fmt_pct(mean([m["known_top1_and_accepted_score"] for m in split_metrics])),
            }
        )

    write_csv(REVISION_TABLES / "open_set_holdout_summary.csv", rows_out)
    write_csv(REVISION_TABLES / "open_set_holdout_details.csv", detail_rows)
    metadata = {
        "note": (
            "Pseudo-open-set diagnostic: classes with available independent-test images are held out "
            "from the reference library and treated as unknown queries. This is not a substitute for "
            "a real external unknown-material dataset."
        ),
        "num_splits": num_splits,
        "unknown_classes_per_split": unknown_classes_per_split,
        "seed": seed,
        "prediction_files": {method: str(path) for method, path in PREDICTION_FILES.items()},
    }
    (REVISION_TABLES / "open_set_holdout_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return rows_out


def main():
    robustness = summarize_robustness()
    open_set = summarize_open_set()
    print("Robustness compact summary")
    for row in robustness:
        print(row)
    print("\nPseudo-open-set holdout summary")
    for row in open_set:
        print(row)


if __name__ == "__main__":
    main()
