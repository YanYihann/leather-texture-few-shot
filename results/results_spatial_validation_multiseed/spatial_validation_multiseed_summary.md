# Multi-Seed Controlled Spatial Validation Summary

This table summarizes repeated runs on the same 203-image controlled spatial validation split. Repeated seeds measure stochastic variation in methods that use random augmented prototypes. This split is controlled validation only and is not an independent test set.

## Mean and Standard Deviation

| Method | Runs | Top-1 mean | Top-1 std | Top-3 mean | Top-3 std | Top-1 range | Top-3 range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen ResNet50 + cosine similarity | 5 | 54.19% | 0.00% | 73.89% | 0.00% | 54.19%-54.19% | 73.89%-73.89% |
| Frozen dinov2_vits14 + cosine similarity | 5 | 58.13% | 0.00% | 74.38% | 0.00% | 58.13%-58.13% | 74.38%-74.38% |
| ResNet50 fine-tuning baseline | 5 | 80.30% | 0.00% | 94.09% | 0.00% | 80.30%-80.30% | 94.09%-94.09% |
| dinov2_vits14 + augmentation prototypes, N=10 | 5 | 66.90% | 0.81% | 83.84% | 1.49% | 65.52%-67.49% | 81.77%-85.71% |
| dinov2_vits14 + augmentation prototypes, N=30 | 5 | 68.57% | 0.88% | 84.53% | 0.56% | 67.49%-69.46% | 83.74%-85.22% |

## Per-Seed Metrics

| Seed | Method | Views | Top-1 Acc | Top-3 Acc | Top-1 Correct | Top-3 Correct | Test Images |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | dinov2_vits14 + augmentation prototypes | 10 | 67.00% | 84.73% | 136 | 172 | 203 |
| 42 | dinov2_vits14 + augmentation prototypes | 30 | 69.46% | 83.74% | 141 | 170 | 203 |
| 42 | Frozen dinov2_vits14 + cosine similarity |  | 58.13% | 74.38% | 118 | 151 | 203 |
| 42 | Frozen ResNet50 + cosine similarity |  | 54.19% | 73.89% | 110 | 150 | 203 |
| 42 | ResNet50 fine-tuning baseline |  | 80.30% | 94.09% | 163 | 191 | 203 |
| 43 | dinov2_vits14 + augmentation prototypes | 10 | 67.49% | 81.77% | 137 | 166 | 203 |
| 43 | dinov2_vits14 + augmentation prototypes | 30 | 67.98% | 84.73% | 138 | 172 | 203 |
| 43 | Frozen dinov2_vits14 + cosine similarity |  | 58.13% | 74.38% | 118 | 151 | 203 |
| 43 | Frozen ResNet50 + cosine similarity |  | 54.19% | 73.89% | 110 | 150 | 203 |
| 43 | ResNet50 fine-tuning baseline |  | 80.30% | 94.09% | 163 | 191 | 203 |
| 44 | dinov2_vits14 + augmentation prototypes | 10 | 67.49% | 85.71% | 137 | 174 | 203 |
| 44 | dinov2_vits14 + augmentation prototypes | 30 | 69.46% | 85.22% | 141 | 173 | 203 |
| 44 | Frozen dinov2_vits14 + cosine similarity |  | 58.13% | 74.38% | 118 | 151 | 203 |
| 44 | Frozen ResNet50 + cosine similarity |  | 54.19% | 73.89% | 110 | 150 | 203 |
| 44 | ResNet50 fine-tuning baseline |  | 80.30% | 94.09% | 163 | 191 | 203 |
| 45 | dinov2_vits14 + augmentation prototypes | 10 | 65.52% | 83.74% | 133 | 170 | 203 |
| 45 | dinov2_vits14 + augmentation prototypes | 30 | 68.47% | 84.73% | 139 | 172 | 203 |
| 45 | Frozen dinov2_vits14 + cosine similarity |  | 58.13% | 74.38% | 118 | 151 | 203 |
| 45 | Frozen ResNet50 + cosine similarity |  | 54.19% | 73.89% | 110 | 150 | 203 |
| 45 | ResNet50 fine-tuning baseline |  | 80.30% | 94.09% | 163 | 191 | 203 |
| 46 | dinov2_vits14 + augmentation prototypes | 10 | 67.00% | 83.25% | 136 | 169 | 203 |
| 46 | dinov2_vits14 + augmentation prototypes | 30 | 67.49% | 84.24% | 137 | 171 | 203 |
| 46 | Frozen dinov2_vits14 + cosine similarity |  | 58.13% | 74.38% | 118 | 151 | 203 |
| 46 | Frozen ResNet50 + cosine similarity |  | 54.19% | 73.89% | 110 | 150 | 203 |
| 46 | ResNet50 fine-tuning baseline |  | 80.30% | 94.09% | 163 | 191 | 203 |
