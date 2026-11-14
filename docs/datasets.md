# Dataset Research

## Prey-specific datasets (the hard-to-find part)

All the usable "cat with prey in mouth" data lives on **Roboflow Universe**
(CC BY 4.0, downloadable as YOLO/COCO/classification formats):

| Dataset | Images | Classes | Notes |
|---|---|---|---|
| [cat with prey detection (Pino Carenza)](https://universe.roboflow.com/pino-carenza-mgn6v/cat-with-prey-detection) | ~1.8k (2.5k augmented) | cat, prey | Collected over ~2 years at a real cat flap for exactly our use case. 20 dataset versions, their YOLOv5 hits mAP@50 ~94.8% |
| [safsd (uzd)](https://universe.roboflow.com/uzd/safsd) | 517 | cat, prey | Same label scheme, extra variety |
| [Prey-Detection-V2 (Moritz K)](https://universe.roboflow.com/moritz-k/prey-detection-v2) | 302 | cat, prey | CC BY 4.0 |
| [Prey-Detection-V3 (Moritz K)](https://universe.roboflow.com/moritz-k/prey-detection-v3) | 106 | cat, prey | Smaller refresh of V2 |
| [Cat_Prey_Analyzer](https://github.com/niciBume/Cat_Prey_Analyzer) | ~150 prey imgs | prey/no-prey | ETH-style RPi project, IR snout crops, used photoshopped prey for augmentation |

Combined: roughly 2.5-3k labeled images, enough for fine-tuning, nowhere near
enough for training from scratch.

Kaggle and Hugging Face have **no** prey-in-mouth datasets (only generic
cat/dog/animal classification sets). Roboflow is the niche source.

## Supporting datasets

- [CAT dataset (Kaggle, crawford)](https://www.kaggle.com/datasets/crawford/cat-dataset):
  9k+ cat images with 9 head landmarks (eyes, **mouth**, ears). Useful for
  training/validating a head-crop stage and for compositing prey onto mouths.
- Oxford-IIIT Pet: cat breeds, bounding boxes. Generic no-prey negatives.
- iNaturalist / ImageNet: mouse, bird, lizard classes if we do prey-type
  classification.
- Own footage later: by far the most valuable no-prey negatives (matching
  camera angle + IR).

## Reference numbers from prior projects

- Prey base rate at the flap: ~3% of entries (heavy class imbalance)
- TDS cat flap project: 1532 night-vision images over 4 months, 415 with prey
  (they oversampled prey events by hand-labeling via an app)
- Cat_Prey_Analyzer: 93% prey recall at 28% false-positive rate using a
  cumulative "trust points" scheme over multiple frames
