"""Fine-tune YOLOv8s (COCO-pretrained) on the two-class hard-hat dataset (default) or, with
--data/--out, on another YOLO dataset such as the site-mode warehouse set
(scripts/prepare_site_dataset.py).

YOLOv8s is chosen because the same architecture is in the Hailo Model Zoo for
Hailo-10H (166 FPS @ batch 1, 252 FPS @ batch 8), so the Stage II port is a
re-compile with the Hailo Dataflow Compiler rather than a new network.

Dataloader workers default to 2: on Windows each spawned worker commits ~1.2 GB and
ultralytics starts `workers` train plus `2 * workers` val workers that all stay alive,
so the old default of 8 meant 24 processes (~29 GB) on a box near its commit limit.

    python scripts/train_detector.py --epochs 40
    python scripts/train_detector.py --resume      # continue an interrupted run
    python scripts/train_detector.py --data @work/datasets/site/data.yaml --name site_v8s \
        --out site_yolov8s.pt --imgsz 960 --epochs 30
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import resolve, work_dir  # noqa: E402


def resumable(ckpt: dict | None) -> bool:
    """Same test ultralytics Model.train applies before honouring resume=True."""
    return bool(ckpt) and ckpt.get("epoch", -1) >= 0 and ckpt.get("optimizer") is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=2, help="train workers; val uses twice as many")
    ap.add_argument("--name", default="helmet_v8s")
    ap.add_argument("--data", default="@work/datasets/helmet/data.yaml")
    ap.add_argument("--out", default=None,
                    help="file name under @work/models (default: helmet_yolov8s.pt for the default --name, "
                         "else <name>.pt), so another run never overwrites the hard-hat weights")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/weights/last.pt")
    a = ap.parse_args()
    if a.out is None:
        a.out = "helmet_yolov8s.pt" if a.name == "helmet_v8s" else f"{a.name}.pt"

    from ultralytics import YOLO

    wd = work_dir()
    if a.resume:
        last = wd / "runs" / a.name / "weights" / "last.pt"
        if not last.is_file():
            sys.exit(f"nothing to resume: no checkpoint at {last}")
        m = YOLO(str(last))
        a.imgsz = int((m.ckpt or {}).get("train_args", {}).get("imgsz", a.imgsz))   # export at the trained size
        if resumable(m.ckpt):
            m.train(resume=True, workers=a.workers)
        else:
            # A finished or early-stopped run has its optimizer stripped (epoch -1).  Calling
            # train(resume=True) on it does not fail: ultralytics silently starts a NEW
            # 100-epoch run on coco8 and the copy below would then ship the old weights.
            print(f"{last} is from a finished run (no optimizer state); nothing to resume, "
                  "skipping to copy/export")
    else:
        model = YOLO(str(wd / "models" / a.model) if (wd / "models" / a.model).exists() else a.model)
        model.train(
            data=str(resolve(a.data)),
            epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers,
            device=0, amp=True, cos_lr=True, close_mosaic=5, patience=12,
            project=str(wd / "runs"), name=a.name, exist_ok=True, plots=True, seed=0,
        )
    best = wd / "runs" / a.name / "weights" / "best.pt"
    dst = wd / "models" / a.out
    shutil.copy2(best, dst)
    print("best weights ->", dst)

    # ONNX for the Hailo Dataflow Compiler (opset 11 is what the Hailo Model Zoo uses for YOLOv8).
    onnx = YOLO(str(dst)).export(format="onnx", opset=11, imgsz=a.imgsz, simplify=True)
    print("onnx ->", onnx)


if __name__ == "__main__":
    main()
