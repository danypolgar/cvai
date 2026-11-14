# cvai - Katzentür Prey Detection

CVAI homework project (HSLU HS26): a camera is mounted at the cat flap (Katzentür).
When the cat comes home, an image model classifies whether it is carrying prey in
its mouth (and ideally what kind: mouse, bird, ...), so the flap can be locked or
an alert sent before the prey ends up in the living room.

## Planned pipeline

1. Camera at the cat door (IR-capable for night shots) triggers on motion
2. Detect the cat and crop the head/snout region
3. Classify the crop: no prey vs prey (optionally prey type)
4. Action: lock flap / send notification

## Key challenges

- Very little prey data: cats bring prey home only a few % of the time
  (class imbalance), so transfer learning + heavy augmentation / synthetic
  compositing are needed
- Night / IR images: most prey arrives at night, model must handle grayscale IR
- Decision must happen in a few seconds at the flap

## Prior art

- [Cat_Prey_Analyzer](https://github.com/niciBume/Cat_Prey_Analyzer) - CNN
  cascade on RPi, snout crops, photoshopped-prey augmentation
- [BalrogCatPreyAnalyzer](https://github.com/dieriver/BalrogCatPreyAnalyzer) -
  maintained fork of the above
- [catflap-prey-detector](https://github.com/fl2o/catflap-prey-detector) -
  YOLO11n + prey classifier on RPi 5 with IR camera

## Quickstart

```bash
uv sync                                     # install deps
cp .env.example .env                        # add your free Roboflow API key
uv run python src/cvai/download_data.py     # fetch datasets -> data/roboflow/
uv run python src/cvai/train.py             # fine-tune YOLO11n (COCO-pretrained)
```

We fine-tune pretrained models (transfer learning), we do not train from
scratch: ~2-3k prey images is far too little for that. See
`docs/datasets.md` for the dataset research.

### GPU training on Modal

Local MPS training works but is slow (~3.5 min/epoch). The same run on a
Modal A10G takes ~10s/epoch (30 epochs in ~6 min, ~$0.30):

```bash
modal volume put cvai-data data/roboflow /roboflow   # once
modal run src/cvai/modal_train.py                    # needs .env with W&B key
```

### Experiment tracking

All runs log to W&B: <https://forge.coreweave.com/wandb/cvai-cat-prey/cvai-katzentuer>

Baseline result (`baseline-yolo11n-a10g`, YOLO11n, 30 epochs, 640px):
**mAP50 0.954, mAP50-95 0.765, precision 0.937, recall 0.919** on the
cat-with-prey-detection val split.

## Structure

- `src/cvai/` - download & training code
- `docs/` - research notes
- `data/`, `models/` - datasets & weights (git-ignored)
- `temp/` - course material & scratch files (git-ignored)
