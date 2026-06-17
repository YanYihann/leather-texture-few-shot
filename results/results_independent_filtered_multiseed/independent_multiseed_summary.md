# Multi-Seed Independent Test Summary

This table summarizes repeated runs on the same 201-real-image filtered independent test set. Repeated seeds measure stochastic variation in methods that use random augmented prototypes. They do not replace a larger real multi-capture independent test set.

## Mean and Standard Deviation

| Method | Runs | Top-1 mean | Top-1 std | Top-3 mean | Top-3 std | Top-1 range | Top-3 range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen ResNet50 + cosine similarity | 5 | 98.01% | 0.00% | 99.50% | 0.00% | 98.01%-98.01% | 99.50%-99.50% |
| Frozen dinov2_vits14 + cosine similarity | 5 | 94.53% | 0.00% | 98.51% | 0.00% | 94.53%-94.53% | 98.51%-98.51% |
| ResNet50 fine-tuning baseline | 5 | 87.56% | 0.00% | 97.01% | 0.00% | 87.56%-87.56% | 97.01%-97.01% |
| dinov2_vits14 + augmentation prototypes, N=10 | 5 | 94.23% | 1.30% | 99.20% | 0.57% | 92.04%-95.02% | 98.51%-100.00% |
| dinov2_vits14 + augmentation prototypes, N=30 | 5 | 96.22% | 0.44% | 99.70% | 0.27% | 95.52%-96.52% | 99.50%-100.00% |

## Per-Seed Metrics

| Seed | Method | Views | Top-1 Acc | Top-3 Acc | Top-1 Correct | Top-3 Correct | Test Images |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | dinov2_vits14 + augmentation prototypes | 10 | 95.02% | 99.50% | 191 | 200 | 201 |
| 42 | dinov2_vits14 + augmentation prototypes | 30 | 96.52% | 100.00% | 194 | 201 | 201 |
| 42 | Frozen dinov2_vits14 + cosine similarity |  | 94.53% | 98.51% | 190 | 198 | 201 |
| 42 | Frozen ResNet50 + cosine similarity |  | 98.01% | 99.50% | 197 | 200 | 201 |
| 42 | ResNet50 fine-tuning baseline |  | 87.56% | 97.01% | 176 | 195 | 201 |
| 43 | dinov2_vits14 + augmentation prototypes | 10 | 95.02% | 99.00% | 191 | 199 | 201 |
| 43 | dinov2_vits14 + augmentation prototypes | 30 | 96.52% | 99.50% | 194 | 200 | 201 |
| 43 | Frozen dinov2_vits14 + cosine similarity |  | 94.53% | 98.51% | 190 | 198 | 201 |
| 43 | Frozen ResNet50 + cosine similarity |  | 98.01% | 99.50% | 197 | 200 | 201 |
| 43 | ResNet50 fine-tuning baseline |  | 87.56% | 97.01% | 176 | 195 | 201 |
| 44 | dinov2_vits14 + augmentation prototypes | 10 | 92.04% | 98.51% | 185 | 198 | 201 |
| 44 | dinov2_vits14 + augmentation prototypes | 30 | 96.52% | 99.50% | 194 | 200 | 201 |
| 44 | Frozen dinov2_vits14 + cosine similarity |  | 94.53% | 98.51% | 190 | 198 | 201 |
| 44 | Frozen ResNet50 + cosine similarity |  | 98.01% | 99.50% | 197 | 200 | 201 |
| 44 | ResNet50 fine-tuning baseline |  | 87.56% | 97.01% | 176 | 195 | 201 |
| 45 | dinov2_vits14 + augmentation prototypes | 10 | 94.03% | 99.00% | 189 | 199 | 201 |
| 45 | dinov2_vits14 + augmentation prototypes | 30 | 95.52% | 99.50% | 192 | 200 | 201 |
| 45 | Frozen dinov2_vits14 + cosine similarity |  | 94.53% | 98.51% | 190 | 198 | 201 |
| 45 | Frozen ResNet50 + cosine similarity |  | 98.01% | 99.50% | 197 | 200 | 201 |
| 45 | ResNet50 fine-tuning baseline |  | 87.56% | 97.01% | 176 | 195 | 201 |
| 46 | dinov2_vits14 + augmentation prototypes | 10 | 95.02% | 100.00% | 191 | 201 | 201 |
| 46 | dinov2_vits14 + augmentation prototypes | 30 | 96.02% | 100.00% | 193 | 201 | 201 |
| 46 | Frozen dinov2_vits14 + cosine similarity |  | 94.53% | 98.51% | 190 | 198 | 201 |
| 46 | Frozen ResNet50 + cosine similarity |  | 98.01% | 99.50% | 197 | 200 | 201 |
| 46 | ResNet50 fine-tuning baseline |  | 87.56% | 97.01% | 176 | 195 | 201 |
