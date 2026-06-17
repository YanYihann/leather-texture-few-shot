| Method | Real source images | Top-1 Acc | Top-1 95% CI | Top-3 Acc | Top-3 95% CI | Correct Top-1 | Correct Top-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet50 fine-tuning baseline | 201 | 87.56% | [83.08%, 92.04%] | 97.01% | [94.53%, 99.00%] | 176 / 201 | 195 / 201 |
| Frozen ResNet50 + cosine similarity | 201 | 98.01% | [96.02%, 99.50%] | 99.50% | [98.51%, 100.00%] | 197 / 201 | 200 / 201 |
| Frozen dinov2_vits14 + cosine similarity | 201 | 94.53% | [91.04%, 97.51%] | 98.51% | [96.52%, 100.00%] | 190 / 201 | 198 / 201 |
| dinov2_vits14 + augmentation prototypes | 201 | 95.02% | [92.04%, 98.01%] | 99.50% | [98.51%, 100.00%] | 191 / 201 | 200 / 201 |
| dinov2_vits14 + augmentation prototypes | 201 | 96.52% | [93.53%, 99.00%] | 100.00% | [100.00%, 100.00%] | 194 / 201 | 201 / 201 |
