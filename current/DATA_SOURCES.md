# Original data and model sources

- Leather original archive, 203 registered entries and 201 Q images: https://doi.org/10.5281/zenodo.20736326 (CC BY 4.0 on that record). Team capture is author-confirmed; the acquisition/session logs remain unavailable.
- T1K+ Fabric: Cusano, Napoletano and Schettini (2021), https://doi.org/10.3390/s21031010. Obtain images through the original paper's data source and observe its noncommercial research terms. This archive redistributes splits, hashes, predictions and numeric results, not Fabric images.
- MatSim: Drehwald et al. (2023), official source https://github.com/ZuseZ4/MatSim-Dataset-Generator-Scripts-And-Neural-net. The evaluated subset is Set2 with the supplied masks, 79 valid-mask photos and 16 material classes. The file inventories and acquisition route are in the archived evaluation scripts and metadata. Do not substitute a different sample silently.
- DTD: Cimpoi et al. (2014), https://www.robots.ox.ac.uk/~vgg/data/dtd/. The current evaluation uses all ten official train/test split definitions. Images are excluded.
- ResNet50: official TorchVision weights `resnet50-11ad3fa6.pth`.
- DINOv2: official https://github.com/facebookresearch/dinov2 source and ViT-S/14 pretrained weights; follow upstream licenses and obtain separately.
- CLIP: official https://github.com/openai/CLIP, MIT source and separate model weights. `clip_matching.py` documents the model hash used.

The manuscript's third-party data citations remain authoritative. Numerical retrieval evidence does not certify acquisition independence or leather material equivalence.
