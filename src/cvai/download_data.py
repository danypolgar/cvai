"""Download prey-detection datasets from Roboflow Universe.

Needs a free Roboflow account: put ROBOFLOW_API_KEY in .env or the environment.
Datasets land in data/roboflow/<name>/ in YOLO format (images + labels + data.yaml).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

# (workspace, project, version) - see docs/datasets.md
DATASETS = [
    ("pino-carenza-mgn6v", "cat-with-prey-detection", 11),  # ~1.8k imgs, cat+prey
    ("uzd", "safsd", 1),                                    # 517 imgs, cat+prey
    ("moritz-k", "prey-detection-v2", 1),                   # 302 imgs, cat+prey
]

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "roboflow"


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("Set ROBOFLOW_API_KEY (free account: https://app.roboflow.com)")

    rf = Roboflow(api_key=api_key)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for workspace, project, version in DATASETS:
        target = OUT_DIR / f"{project}-{version}"
        if target.exists():
            print(f"skip {project} (already at {target})")
            continue
        print(f"downloading {workspace}/{project} v{version} ...")
        try:
            proj = rf.workspace(workspace).project(project)
            proj.version(version).download("yolov8", location=str(target))
        except Exception as e:  # keep going if one dataset moved/changed version
            print(f"  FAILED {project}: {e}", file=sys.stderr)

    print(f"\nDone. Datasets in {OUT_DIR}")


if __name__ == "__main__":
    main()
