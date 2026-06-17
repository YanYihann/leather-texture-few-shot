$ErrorActionPreference = "Stop"

python src/evaluate_revision_ablations.py `
  --reference-dir data/dataset_train `
  --eval-dir data/dataset_val `
  --output-dir results/generated/ablation `
  --views 1 5 10 20 30 50 `
  --similarity-metrics cosine euclidean mahalanobis `
  --lbp-weights 0.0 0.1 0.2 0.3 0.5
