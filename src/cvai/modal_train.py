"""Train the prey detector on a Modal GPU instead of the local Mac.

Usage:
    modal volume create cvai-data                         # once
    modal volume put cvai-data data/roboflow /roboflow    # once (uploads dataset)
    modal run src/cvai/modal_train.py                     # 30-epoch A10G run
    modal run src/cvai/modal_train.py --epochs 50 --gpu T4

Reads WANDB_API_KEY / WANDB_ENTITY / run config from the local .env via
modal.Secret.from_dotenv(). Weights persist on the volume under /data/runs/
and the best.pt is downloaded back to models/<name>/ afterwards.
"""

import modal

app = modal.App("cvai-katzentuer")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics", "wandb")
)

vol = modal.Volume.from_name("cvai-data", create_if_missing=True)

WANDB_PROJECT = "cvai-katzentuer"
DATA_YAML = "/data/roboflow/cat-with-prey-detection-11/data.yaml"


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=2 * 60 * 60,
)
def train(
    epochs: int = 30,
    model: str = "yolo11n.pt",
    name: str = "baseline-yolo11n-a10g",
    imgsz: int = 640,
    batch: int = 32,
) -> float:
    import os

    import wandb
    from ultralytics import YOLO, settings

    settings.update({"wandb": True})
    # Explicit init: the ultralytics auto-callback is unreliable with
    # self-hosted W&B endpoints. The callback reuses an existing run.
    wandb.init(
        project=WANDB_PROJECT,
        entity=os.environ["WANDB_ENTITY"],
        name=name,
        config={"model": model, "epochs": epochs, "imgsz": imgsz, "batch": batch},
    )

    m = YOLO(model)
    m.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="/data/runs",
        name=name,
        exist_ok=True,
    )
    metrics = m.val()
    map50 = float(metrics.box.map50)
    wandb.finish()
    vol.commit()
    print(f"done: mAP50={map50:.3f}, weights in volume at /data/runs/{name}/weights/")
    return map50


@app.local_entrypoint()
def main(
    epochs: int = 30,
    model: str = "yolo11n.pt",
    name: str = "baseline-yolo11n-a10g",
    imgsz: int = 640,
    batch: int = 32,
):
    map50 = train.remote(epochs=epochs, model=model, name=name, imgsz=imgsz, batch=batch)
    print(f"\nRemote training finished. mAP50 = {map50:.3f}")
    print(f"Fetch weights: modal volume get cvai-data /runs/{name}/weights/best.pt models/{name}/best.pt")
