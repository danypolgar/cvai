"""Run the trained prey detector on images.

Usage:
    uv run python src/cvai/predict.py path/to/image.jpg [more images/dirs ...]
    uv run python src/cvai/predict.py --weights models/prey-yolo/weights/best.pt img.jpg

Annotated images are written to models/predict/. Any 'prey' detection is
flagged loudly: that is the signal the cat flap would act on.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "prey-yolo" / "weights" / "best.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources", nargs="+", help="image files or directories")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    args = p.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"{weights} not found - train first (src/cvai/train.py)")

    model = YOLO(weights)
    results = model.predict(
        source=[str(s) for s in args.sources],
        conf=args.conf,
        save=True,
        project=str(ROOT / "models"),
        name="predict",
        exist_ok=True,
    )

    print()
    for r in results:
        names = [model.names[int(c)] for c in r.boxes.cls]
        confs = [f"{float(c):.2f}" for c in r.boxes.conf]
        found = ", ".join(f"{n} ({c})" for n, c in zip(names, confs)) or "nothing"
        prey = any(n.lower() == "prey" for n in names)
        flag = "  🚨 PREY -> lock the flap!" if prey else ""
        print(f"{Path(r.path).name}: {found}{flag}")

    print(f"\nAnnotated images: {ROOT / 'models' / 'predict'}")


if __name__ == "__main__":
    main()
