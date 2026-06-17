# Leather Texture Few-Shot Classification

This repository contains the code and compact result files for a manuscript on few-shot leather texture recognition for intelligent manufacturing. The dataset itself is archived separately on Zenodo:

Dataset DOI: https://doi.org/10.5281/zenodo.20736326

The repository is intended to support reproducibility of the reported evaluation protocol, including multi-seed independent testing, controlled spatial validation, DINOv2/ResNet50 discrepancy analysis, ablations, additional baselines, robustness tests, and pseudo-open-set analysis.

## Repository Structure

```text
src/       Python scripts for training, evaluation, ablation, robustness, and summaries
data/      Instructions for downloading and placing the Zenodo dataset
results/   Compact CSV/JSON/Markdown result files used during manuscript revision
scripts/   PowerShell helper commands for common reproduction runs
```

Large image datasets, model weights, and feature caches are not stored in this Git repository. Download the dataset from Zenodo and place the extracted folders as described in `data/README.md`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For OpenAI CLIP baselines, install the original CLIP package as well:

```powershell
python -m pip install git+https://github.com/openai/CLIP.git
```

## Data Layout

After downloading the Zenodo archive, the expected layout is:

```text
data/
  my_dataset_1/
  dataset_train/
  dataset_val/
  my_dataset_test_filtered/
```

The paper evaluates the filtered independent test set (`my_dataset_test_filtered`), not an augmented test set.

## Example Runs

Run the five-seed independent evaluation:

```powershell
.\scripts\run_independent_multiseed.ps1
```

Run the five-seed controlled spatial validation:

```powershell
.\scripts\run_spatial_validation_multiseed.ps1
```

Run ablations:

```powershell
.\scripts\run_ablations.ps1
```

These scripts assume that the dataset folders are placed under `data/`. Some ResNet50 evaluations require a trained `best_leather_model_val.pth`; if the model file is not available, regenerate it with `src/train_leather_val.py` or run only frozen-feature methods.

## Included Results

The `results/` folder contains compact outputs generated during manuscript revision, including:

- independent and spatial multi-seed summaries;
- prototype-view, augmentation, similarity-metric, and LBP-fusion ablations;
- CLIP, ProtoNet, ConvNeXt, EfficientNetV2, Swin Transformer, ResNet50, and DINOv2 baselines;
- robustness and pseudo-open-set summaries;
- ResNet50 versus DINOv2 discrepancy tables.

## Citation

Please cite the manuscript and the Zenodo dataset DOI when using this code or dataset:

https://doi.org/10.5281/zenodo.20736326

