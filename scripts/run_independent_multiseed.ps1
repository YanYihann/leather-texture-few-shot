$ErrorActionPreference = "Stop"

python src/run_independent_multiseed.py `
  --reference-dir data/dataset_train `
  --test-dir data/my_dataset_test_filtered `
  --model-path models/best_leather_model_val.pth `
  --output-dir results/generated/independent_filtered_multiseed `
  --summary-prefix independent `
  --evaluation-name "Independent Test"

