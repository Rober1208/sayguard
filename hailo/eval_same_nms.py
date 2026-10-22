"""Float vs Hailo INT8 with ONE evaluator (hailo/yolo_eval.py): the same held-out images, letterbox,
NMS thresholds (Hailo's: score >= 0.2, IoU 0.7, 100 boxes per class) and pycocotools.

  onnx          the float ONNX on the CPU: what the PC runs, measured the way the chip's NMS reports
  fp_optimized  the Hailo graph after optimization, float math
  quantized     the INT8 model in the compiler's quantized emulator, which models the chip's INT8
                arithmetic (SDK_QUANTIZED; not the DFC's SDK_BIT_EXACT mode)

    # warehouse detector, validation split of site_v3 (W016 cameras 9-10, never trained on)
    python hailo/eval_same_nms.py site --har ~/hailo/build/site_optimized.har [--tag _r544]
    # hard-hat detector: only the float ONNX, to compare with the INT8 0.889 in hailo_int8.json
    python hailo/eval_same_nms.py helmet --onnx-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo_eval import coco_eval, hailo_preds, letterbox, load_yolo_split, onnx_preds  # noqa: E402

WORK = Path(os.environ.get("SAYGUARD_WORK", "D:/ugen-work" if os.name == "nt" else "/mnt/d/ugen-work"))
BUILD = Path(os.environ.get("HAILO_BUILD", Path.home() / "hailo" / "build"))
OUT = [WORK / "results", Path(__file__).resolve().parents[1] / "docs" / "results"]
SCORE_TH, IOU_TH = 0.2, 0.7          # Model Zoo yolov8s_nms_config.json, as compiled into the HEFs

MODELS = {
    "site": {"names": ["person", "forklift", "pallet_truck", "transporter", "mobile_robot", "humanoid_robot"],
             "images": WORK / "datasets" / "site_v3" / "images" / "val",
             "labels": WORK / "datasets" / "site_v3" / "labels" / "val",
             "onnx": WORK / "models" / "site_yolov8s.onnx", "hw": (960, 960)},
    "helmet": {"names": ["helmet", "head"],
               "images": WORK / "datasets" / "helmet" / "images" / "test",
               "labels": WORK / "datasets" / "helmet" / "labels" / "test",
               "onnx": WORK / "models" / "helmet_yolov8s.onnx", "hw": (640, 640)},
}


def image_dataset(paths: list[Path], h: int, w: int):
    import tensorflow as tf

    def gen():
        for p in paths:
            yield letterbox(Image.open(p), h, w)[0], {"id": np.int32(0)}
    return tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec((h, w, 3), tf.float32), {"id": tf.TensorSpec((), tf.int32)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("--har", help="optimized HAR (omit with --onnx-only)")
    ap.add_argument("--onnx", help="float ONNX (default: the model's file in @work/models)")
    ap.add_argument("--tag", default="", help="variant suffix used by build_site.py (reads its input size)")
    ap.add_argument("--onnx-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    cfg = MODELS[a.model]
    h, w = cfg["hw"]
    onnx_path = a.onnx or str(cfg["onnx"])
    meta_file = BUILD / f"site{a.tag}.json"
    if a.model == "site" and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        h, w, onnx_path = meta["h"], meta["w"], a.onnx or meta["onnx"]
    names = cfg["names"]
    paths, gt = load_yolo_split(cfg["images"], cfg["labels"], names, a.limit)
    metas = [letterbox(Image.open(p), h, w)[1] for p in paths]
    res = {"model": a.model, "variant": a.tag or "default", "input_hw": [h, w], "images": len(paths),
           "instances": len(gt["annotations"]), "split": str(cfg["images"]),
           "nms": {"score_th": SCORE_TH, "iou_th": IOU_TH, "max_per_class": 100}, "onnx": onnx_path}

    t0 = time.time()
    res["onnx_float"] = coco_eval(gt, onnx_preds(onnx_path, paths, gt, h, w, SCORE_TH, IOU_TH), names)
    res["onnx_float"]["seconds"] = round(time.time() - t0, 1)
    print("onnx_float", res["onnx_float"], flush=True)

    if not a.onnx_only:
        from hailo_sdk_client import ClientRunner, InferenceContext
        runner = ClientRunner(har=a.har)
        res["har"] = a.har
        for name, ctx_type in (("fp_optimized", InferenceContext.SDK_FP_OPTIMIZED),
                               ("quantized", InferenceContext.SDK_QUANTIZED)):
            t0 = time.time()
            with runner.infer_context(ctx_type) as ctx:
                out = runner.infer(ctx, image_dataset(paths, h, w), batch_size=a.batch)
            m = coco_eval(gt, hailo_preds(out, metas, gt, len(names), h, w), names)
            m["seconds"] = round(time.time() - t0, 1)
            res[name] = m
            print(name, m, flush=True)
        res["mAP50_drop_vs_onnx"] = round(res["onnx_float"]["mAP50"] - res["quantized"]["mAP50"], 4)
        res["mAP50_drop_vs_fp"] = round(res["fp_optimized"]["mAP50"] - res["quantized"]["mAP50"], 4)

    out_name = f"{a.model}{a.tag}_same_nms.json" if not a.onnx_only else f"{a.model}_onnx_same_nms.json"
    for d in OUT:
        d.mkdir(parents=True, exist_ok=True)
        (d / out_name).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
