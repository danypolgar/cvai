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

## Structure

- `temp/` - course material & scratch files (git-ignored)
