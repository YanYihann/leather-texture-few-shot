<div align="center">

# Leather Texture Few-Shot Classification

Reproducibility code and compact results for few-shot leather-texture recognition in intelligent manufacturing.

[![Python](https://img.shields.io/badge/Python-ML%20research-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Dataset DOI](https://img.shields.io/badge/dataset-10.5281%2Fzenodo.20736326-1682D4)](https://doi.org/10.5281/zenodo.20736326)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## Overview

This repository contains training, evaluation, ablation, robustness, and analysis code used for a manuscript on few-shot leather-texture recognition. Large datasets, model weights, and feature caches are intentionally excluded; the dataset is archived separately on Zenodo.

## Reproduced analyses

- Multi-seed independent-test evaluation
- Controlled spatial validation
- DINOv2 / ResNet50 discrepancy analysis
- Prototype-view, augmentation, similarity-metric, and LBP-fusion ablations
- CLIP, ProtoNet, ConvNeXt, EfficientNetV2, Swin Transformer, ResNet50, and DINOv2 baselines
- Robustness and pseudo-open-set summaries

## Setup

```powershell
git clone https://github.com/YanYihann/leather-texture-few-shot.git
cd leather-texture-few-shot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the original OpenAI CLIP baseline:

```powershell
python -m pip install git+https://github.com/openai/CLIP.git
```

## Data

Download the archive from [Zenodo](https://doi.org/10.5281/zenodo.20736326), then follow `data/README.md`. The expected local layout is:

```text
data/
├── my_dataset_1/
├── dataset_train/
├── dataset_val/
└── my_dataset_test_filtered/
```

The reported independent evaluation uses `my_dataset_test_filtered`, not an augmented test set.

## Common runs

```powershell
.\scripts\run_independent_multiseed.ps1
.\scripts\run_spatial_validation_multiseed.ps1
.\scripts\run_ablations.ps1
```

Some ResNet50 evaluations require `best_leather_model_val.pth`. Regenerate it with `src/train_leather_val.py` when weights are unavailable, or run only frozen-feature methods.

## Repository map

```text
src/       Training, evaluation, robustness, ablation, and summary code
scripts/   PowerShell reproduction entry points
data/      Dataset download and layout instructions
results/   Compact CSV, JSON, and Markdown outputs
```

## Reproducibility notes

For comparable results, record the dataset archive version, Python and CUDA versions, dependency versions, device, random seeds, checkpoint identity, and the exact command used. Hardware and library differences can affect deterministic behavior.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff) and cite the dataset DOI: <https://doi.org/10.5281/zenodo.20736326>.

## License

Code is released under the [MIT License](LICENSE). Dataset usage is governed by the terms published with the Zenodo record.


