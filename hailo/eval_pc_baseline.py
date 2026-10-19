"""PC reference for hailo/eval_int8.py: the original PyTorch weights, scored by the same COCO
evaluator on the same 2,001 test images, at two settings:

  deploy   the NMS the HEF carries (score >= 0.2, IoU 0.7, <= 100 boxes per class), so
           "PC float vs Hailo float vs Hailo INT8" differ only by the compiler
  full     score >= 0.001, the setting Ultralytics uses for mAP (docs/results: 0.939)

Run in the SayGuard venv (needs ultralytics):
    python hailo/eval_pc_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_int8 import NAMES, OUT, WORK, coco_eval, load_split  # noqa: E402

SETTINGS = {"deploy": {"conf": 0.2, "iou": 0.7, "max_det": 100 * len(NAMES)},
            "full": {"conf": 0.001, "iou": 0.7, "max_det": 300}}


def main() -> None:
    from ultralytics import YOLO

    model = YOLO(str(WORK / "models" / "helmet_yolov8s.pt"))
    imgs, gt = load_split(None)
    res = {"weights": "helmet_yolov8s.pt (PyTorch, fp32)", "images": len(imgs), "instances": len(gt["annotations"])}
    for name, kw in SETTINGS.items():
        preds, t0 = [], time.time()
        for s in range(0, len(imgs), 32):
            for k, r in enumerate(model.predict([str(p) for p in imgs[s:s + 32]], imgsz=640, verbose=False, **kw)):
                for (x1, y1, x2, y2), c, sc in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    preds.append({"image_id": s + k, "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1],
                                  "score": float(sc)})
        m = coco_eval(gt, preds)
        m.update(kw)
        m["seconds"] = round(time.time() - t0, 1)
        res[name] = m
        print(name, m, flush=True)
    for d in OUT:
        d.mkdir(parents=True, exist_ok=True)
        (d / "pc_baseline.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
