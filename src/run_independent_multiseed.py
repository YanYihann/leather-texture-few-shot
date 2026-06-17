import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def percent(value):
    return f"{float(value) * 100:.2f}%"


def mean(values):
    return sum(values) / len(values) if values else math.nan


def sample_std(values):
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary_rows, per_seed_rows, evaluation_name, evaluation_note):
    lines = [
        f"# Multi-Seed {evaluation_name} Summary",
        "",
        evaluation_note,
        "",
        "## Mean and Standard Deviation",
        "",
    ]
    headers = [
        "Method",
        "Runs",
        "Top-1 mean",
        "Top-1 std",
        "Top-3 mean",
        "Top-3 std",
        "Top-1 range",
        "Top-3 range",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary_rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")

    lines.extend(["", "## Per-Seed Metrics", ""])
    headers = [
        "Seed",
        "Method",
        "Views",
        "Top-1 Acc",
        "Top-3 Acc",
        "Top-1 Correct",
        "Top-3 Correct",
        "Test Images",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in per_seed_rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_seed(args, seed):
    seed_dir = Path(args.output_dir) / f"seed_{seed}"
    script_dir = Path(__file__).resolve().parent
    command = [
        sys.executable,
        str(script_dir / "evaluate_independent_filtered.py"),
        "--reference-dir",
        args.reference_dir,
        "--test-dir",
        args.test_dir,
        "--output-dir",
        str(seed_dir),
        "--model-path",
        args.model_path,
        "--dinov2-model",
        args.dinov2_model,
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--seed",
        str(seed),
        "--methods",
        *args.methods,
    ]
    print("Running:", " ".join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, check=True)


def collect_metrics(output_dir):
    rows = []
    for seed_dir in sorted(Path(output_dir).glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed = seed_dir.name.replace("seed_", "")
        for metrics_path in sorted(seed_dir.glob("*_metrics.json")):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "Seed": seed,
                    "Method": metrics.get("method", ""),
                    "Views": metrics.get("views_per_class", ""),
                    "Top-1 Acc": percent(metrics.get("top1_accuracy", 0.0)),
                    "Top-3 Acc": percent(metrics.get("top3_accuracy", 0.0)),
                    "Top-1 Correct": metrics.get("top1_correct", ""),
                    "Top-3 Correct": metrics.get("top3_correct", ""),
                    "Test Images": metrics.get("test_images", ""),
                    "_top1": float(metrics.get("top1_accuracy", 0.0)),
                    "_top3": float(metrics.get("top3_accuracy", 0.0)),
                }
            )
    return rows


def summarize(per_seed_rows):
    grouped = defaultdict(list)
    for row in per_seed_rows:
        grouped[(row["Method"], row["Views"])].append(row)

    summary_rows = []
    for (method, views), rows in sorted(grouped.items()):
        top1_values = [row["_top1"] for row in rows]
        top3_values = [row["_top3"] for row in rows]
        method_label = method if views == "" else f"{method}, N={views}"
        summary_rows.append(
            {
                "Method": method_label,
                "Runs": len(rows),
                "Top-1 mean": percent(mean(top1_values)),
                "Top-1 std": percent(sample_std(top1_values)),
                "Top-3 mean": percent(mean(top3_values)),
                "Top-3 std": percent(sample_std(top3_values)),
                "Top-1 range": f"{percent(min(top1_values))}-{percent(max(top1_values))}",
                "Top-3 range": f"{percent(min(top3_values))}-{percent(max(top3_values))}",
            }
        )
    return summary_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default="dataset_train")
    parser.add_argument("--test-dir", default="my_dataset_test_filtered")
    parser.add_argument("--output-dir", default="results/generated/independent_filtered_multiseed")
    parser.add_argument("--model-path", default="best_leather_model_val.pth")
    parser.add_argument("--dinov2-model", default="dinov2_vits14")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["resnet50", "frozen_resnet50", "dinov2", "dinov2_aug10", "dinov2_aug30"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--summary-prefix", default="independent")
    parser.add_argument("--evaluation-name", default="Independent Test")
    parser.add_argument(
        "--evaluation-note",
        default=(
            "This table summarizes repeated runs on the same 201-real-image filtered independent test set. "
            "Repeated seeds measure stochastic variation in methods that use random augmented prototypes. "
            "They do not replace a larger real multi-capture independent test set."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.summarize_only:
        for seed in args.seeds:
            run_seed(args, seed)

    per_seed_rows = collect_metrics(args.output_dir)
    if not per_seed_rows:
        print("No metrics found yet.")
        return

    summary_rows = summarize(per_seed_rows)
    output_dir = Path(args.output_dir)
    prefix = args.summary_prefix
    write_csv(
        output_dir / f"{prefix}_multiseed_per_seed_metrics.csv",
        [{key: value for key, value in row.items() if not key.startswith("_")} for row in per_seed_rows],
    )
    write_csv(output_dir / f"{prefix}_multiseed_summary.csv", summary_rows)
    write_markdown(
        output_dir / f"{prefix}_multiseed_summary.md",
        summary_rows,
        per_seed_rows,
        args.evaluation_name,
        args.evaluation_note,
    )
    (output_dir / f"{prefix}_multiseed_summary.json").write_text(
        json.dumps(
            {
                "evaluation_name": args.evaluation_name,
                "note": args.evaluation_note,
                "summary": summary_rows,
                "per_seed": [{key: value for key, value in row.items() if not key.startswith("_")} for row in per_seed_rows],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote multi-seed summary to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
