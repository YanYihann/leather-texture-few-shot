# Fixed-query Fabric reference-investment reanalysis

This companion belongs to the manuscript revision dated 31 October 2026.
It is additional to the GitHub/Zenodo 2.0.0 snapshots and the earlier ScienceDB SI.

## Replay

Requires Python and NumPy. The recorded run used Python 3.12.14 and NumPy 2.3.5.

    python fixed727_analysis.py --input-dir inputs --output-dir replay

Compare `replay/fixed727_summary.csv` and `replay/fixed727_paired_contrasts.csv`
with the same filenames in `results`. No images, weights, features or inference
are required. All four specified contrasts are reported. Intervals use 2,000
source-instance-cluster bootstrap draws (seed 20261029), with a452/a456 together.
They are exploratory, conditional and unadjusted for multiplicity.

`inputs` contains only the required prediction subsets from the original release.
`environment.json` records hashes of the original full prediction files. A replay
on subsets therefore has different input hashes but must reproduce the results.
`analysis_plan.json` was written before this new computation, after earlier study
results had already been inspected. It is not a preregistration.

The center row preserves the saved R1 assignments. Two label records of the known
exact-copy pair exchange Top-1 assignments between saved primary and R1 scoring
paths. Their shared-cluster correctness sum is identical. The point estimates
and paired cluster-bootstrap calculations are invariant to that exchange.

The primary test has 1,082 records; every new comparison here instead uses the
same 727 records, 725 distinct source photographs and 354 clusters. Frozen
center, A5, A30 and fusion retain the original gallery and settings. R2 adds the
already allocated second original reference photograph. Photographic effort and
Fabric runtime were not measured; view counts are not economic costs.

Script: MIT, following the code release. Derived prediction and numerical
outputs: CC BY 4.0, following the numerical release. Third-party photographs and
pretrained weights are not included or relicensed.
