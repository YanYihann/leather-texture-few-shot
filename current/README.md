# Current code and numerical results, compact version 2.0.0

This compact GitHub release supports the 30 October 2026 working version of “Evaluating One-Reference Visual Retrieval in Material Collections: Leather and Public Fabric Evidence”. It contains the evaluated protocols and saved outputs, including the fixed-727 reanalysis and English reports. The full archive DOI is https://doi.org/10.5281/zenodo.23831908. Historical 2.0.0 remains at https://doi.org/10.5281/zenodo.23794034. It is not an accepted article or an all-author approval declaration.

## Check the release without models or original images

From the repository root, run `python current/verify_release.py`. This standard-library check verifies the release manifest, recomputes leather clean and Fabric primary Recall@1/3/5 and MRR from saved per-query ranks, and checks the Fabric role counts and absence of cross-role file/pixel hashes. It neither runs training/inference nor regenerates confidence intervals. The companion full numerical archive includes the complete stored leather rankings and Fabric score arrays, allowing `python current/verify_scores.py` to independently recompute both 355-label and 353-label Fabric summaries. Those binary arrays are omitted from this compact GitHub tree. The complete version 2.0.0 archive is published at https://doi.org/10.5281/zenodo.23794034. Neither command runs confidence-interval resampling.

`source_data/` maps reviewed numerical files to the manuscript; `manuscript_tables/` contains the current TeX table bodies. `workspace/` preserves the relative directory layout expected by the archived experiment scripts. `FILE_MANIFEST.csv` binds every current-release file to SHA-256 and bytes; `SOURCE_PROVENANCE.csv` records original relative locations and byte-identical scientific sources. A file list establishes integrity, not independent acquisition or a new experimental replication.

## Reconstructing experiments

Use Python 3.10+ with the original experiment environment (PyTorch 2.3.1, TorchVision 0.18.1; NumPy, SciPy, Pillow, scikit-learn and threadpoolctl). The historical root `requirements.txt` is not a complete lock for this current release. Install `current/requirements-current.txt` in a dedicated environment, adjusting the PyTorch wheel to the target system.

Acquire the leather dataset from https://doi.org/10.5281/zenodo.20736326 and place its split directories under `current/workspace/IC_project/01_核心代码与数据集/`. Obtain T1K+ Fabric, MatSim and DTD from their original sources as described in `DATA_SOURCES.md`; third-party photos and pretrained weights are not included or relicensed. Put Fabric originals in `current/workspace/T1k_dataset/fabric_images/`. The scripts retain their original cache/checkpoint contracts; creating those artifacts is required before the dependent stages can run.

For Fabric, the original sequence is `prepare_dataset.py`, `extract_features.py`, `train_adaptation.py`, `evaluate_results.py` and `verify_results.py`, under `workspace/output/T1K_fabric_experiments_20261029/scripts/`. Before extracting features, acquire ResNet50 and DINOv2 weights and the local DINOv2 source expected by `common.py`; acquire official CLIP source/weights at the documented workspace path. Do not reuse saved caches after changing inputs, preprocessing, split manifests or weights. The preparation script must reproduce the deposited photo split hash before any comparison to this release. Archived scripts can overwrite results; execute experiments in a copy.

For leather, scripts from `Sensors_revision_20261010`, `Sensors_revision_20261011`, `Sensors_revision_20261027` and `Sensors_restructured_20261027` implement the successive reference, adaptation, synthetic-query and reconciliation stages. `Sensors_extended_20261028/scripts/extended_adaptation.py` and `clip_matching.py` implement the 50-epoch and clean CLIP extensions. These scripts require original photographs and their documented intermediate artifacts. This release has not been validated as a fresh end-to-end training environment; the verified command above is saved-output arithmetic and integrity checking. Complete historical caches, exact adapted checkpoints and pixel caches are absent, so bitwise model reconstruction is not promised.

## Evidence limits and versions

Leather has 203 registered entries and 201 Q images. The photographs were taken by the research team, as confirmed by the depositing author. Capture-level records are unavailable, so acquisition independence, cross-device/session generalization and physical reflectance measurement are not established. Spatial development images and synthetic descendants are dependent computational measurements.

Fabric retains 355 official labels and 1,792 records, with five exact duplicate pairs under a452/a456 (1,787 unique files). Cross-role duplicates are prevented. The separate 353-label sensitivity removes both ambiguous labels at scoring; it is not a retraining experiment. Frozen retrieval, validation-selected score fusion and ResNet50 adaptation are distinct evaluated configurations.

The fixed historical software DOI https://doi.org/10.5281/zenodo.20736698 identifies v1.0.1 only. Its concept DOI https://doi.org/10.5281/zenodo.20736697 resolves across software versions. A new version receives a distinct DOI; do not cite the old fixed DOI as this code. Dataset DOI https://doi.org/10.5281/zenodo.20736326 continues to identify the unchanged original data files.

A current supporting-materials draft has been saved in the ScienceDB JDIS community. Its DOI remains unregistered; the authors administer private reviewer access through the submission package. Zenodo/GitHub do not replace that journal requirement. SI and original photos are not included in this code archive.

## Licensing

Source code is MIT, retaining the repository LICENSE and any third-party notices. Derived CSV/JSON numerical results and split/rank/score data are CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Third-party source photographs, pretrained models and upstream software remain under their own terms. No license here extends to excluded materials. Permission for publication, public sharing and CC BY 4.0 of the team's leather photographs was confirmed by the user on 30 October 2026; no photographer identity or written agreement was supplied in this update. See the historical dataset record for its published CC BY 4.0 terms.

## Added in 2.1.0

`reanalyses/fixed727/` preserves the complete original companion. `reports/` provides English revision explanations and the concise portrait report. Run `python current/verify_fixed727.py` to verify alignment and point-estimate arithmetic without resampling or models. Original code-creator attribution is retained; it does not certify paper authorship roles or approval.
