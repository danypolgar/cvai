"""Fine-tune a pretrained YOLO model on the cat-prey dataset.

Usage:
    uv run python src/cvai/train.py                     # defaults
    uv run python src/cvai/train.py --epochs 50 --model yolo11s.pt

Starts from COCO-pretrained weights (transfer learning); we never train
from scratch - see README "Key challenges".
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "roboflow" / "cat-with-prey-detection-11" / "data.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="yolo11n.pt", help="pretrained checkpoint")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="dataset data.yaml")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(
            f"{data} not found - run `uv run python src/cvai/download_data.py` first"
        )

    model = YOLO(args.model)  # downloads COCO weights on first run
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / "models"),
        name="prey-yolo",
        exist_ok=True,
    )
    metrics = model.val()
    print(f"\nmAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
