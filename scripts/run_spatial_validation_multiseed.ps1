$ErrorActionPreference = "Stop"

python src/run_independent_multiseed.py `
  --reference-dir data/dataset_train `
  --test-dir data/dataset_val `
  --model-path models/best_leather_model_val.pth `
  --output-dir results/generated/spatial_validation_multiseed `
  --summary-prefix spatial_validation `
  --evaluation-name "Controlled Spatial Validation" `
  --evaluation-note "This table summarizes repeated runs on the controlled spatial validation split."

