"""Emulate the compiled helmet detector with the Hailo Dataflow Compiler and measure what
INT8 quantization costs: the same held-out test split, the same COCO evaluator, two
emulator contexts:

  fp_optimized - the Hailo graph after optimization, float math (no quantization)
  quantized    - bit-exact emulation of the INT8 model the Hailo-10H runs

Run inside the DFC venv (WSL):
    python hailo/eval_int8.py --har ~/hailo/build/helmet_optimized.har [--probe] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

WORK = Path(os.environ.get("SAYGUARD_WORK", "D:/ugen-work" if os.name == "nt" else "/mnt/d/ugen-work"))
DATA = WORK / "datasets" / "helmet"
OUT = [WORK / "results", Path(__file__).resolve().parents[1] / "docs" / "results"]
NAMES = ["helmet", "head"]


def load_split(limit: int | None) -> tuple[list[Path], dict]:
    imgs = sorted((DATA / "images" / "test").glob("*.jpg"))[:limit]
    gt = {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n in enumerate(NAMES)]}
    aid = 1
    for i, p in enumerate(imgs):
        W, H = Image.open(p).size
        gt["images"].append({"id": i, "file_name": p.name, "width": W, "height": H})
        lbl = DATA / "labels" / "test" / (p.stem + ".txt")
        for line in (lbl.read_text().splitlines() if lbl.exists() else []):
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            w, h = float(w) * W, float(h) * H
            x, y = float(cx) * W - w / 2, float(cy) * H - h / 2
            gt["annotations"].append({"id": aid, "image_id": i, "category_id": int(c), "bbox": [x, y, w, h],
                                      "area": w * h, "iscrowd": 0})
            aid += 1
    return imgs, gt


def image_dataset(paths: list[Path]):
    """Streamed (image, info) pairs: one runner.infer call per context.  Calling infer once per
    batch leaked ~0.6 GB a minute in the emulator (the 2,001-image run reached 10 GB and was
    stopped); a dataset also keeps all 2,001 images (9.8 GB as float32) out of memory."""
    import tensorflow as tf

    def gen():
        for i, p in enumerate(paths):
            yield np.asarray(Image.open(p).convert("RGB").resize((640, 640)), np.float32), {"id": np.int32(i)}
    return tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec((640, 640, 3), tf.float32), {"id": tf.TensorSpec((), tf.int32)}))


def describe(x, depth=0) -> str:
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]" + (" of " + describe(x[0], depth + 1) if x and depth < 4 else "")
    if isinstance(x, dict):
        return "dict{" + ", ".join(f"{k}: {describe(v, depth + 1)}" for k, v in list(x.items())[:6]) + "}"
    if hasattr(x, "shape"):
        return f"array{tuple(x.shape)} {x.dtype}"
    return type(x).__name__


def to_dets(out, n: int) -> list[list[tuple[int, float, list[float]]]]:
    """HAILO NMS by class -> per image [(class, score, [x, y, w, h] in 640 px)].
    Handles the layouts the emulator returns: per image a list per class of [K,5]
    (y1, x1, y2, x2, score), or arrays (N, C, K, 5) / (N, C, 5, K)."""
    if isinstance(out, dict):
        out = next(iter(out.values()))
    res = []
    for i in range(n):
        per = out[i]
        dets = []
        for c in range(len(NAMES)):
            arr = np.asarray(per[c], dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.ndim == 2 and arr.shape[0] == 5 and arr.shape[1] != 5:
                arr = arr.T
            for y1, x1, y2, x2, s in arr.reshape(-1, 5):
                if s <= 0:
                    continue
                dets.append((c, float(s), [x1 * 640, y1 * 640, (x2 - x1) * 640, (y2 - y1) * 640]))
        res.append(dets)
    return res


def coco_eval(gt: dict, preds: list[dict]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    G = COCO()
    G.dataset = gt
    G.createIndex()
    if not preds:
        return {"mAP50-95": 0.0, "mAP50": 0.0}
    D = G.loadRes(preds)
    out = {}
    E = COCOeval(G, D, "bbox")
    E.evaluate(); E.accumulate(); E.summarize()
    out["mAP50-95"], out["mAP50"] = round(float(E.stats[0]), 4), round(float(E.stats[1]), 4)
    for cid, name in enumerate(NAMES):
        e = COCOeval(G, D, "bbox")
        e.params.catIds = [cid]
        e.evaluate(); e.accumulate(); e.summarize()
        out[f"{name}_mAP50"] = round(float(e.stats[1]), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", required=True)
    ap.add_argument("--probe", action="store_true", help="print the output structure for 2 images and stop")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=None,
                    help="results file name in results/ and docs/results/ (default: hailo_int8.json for a full "
                         "run of both contexts, else <har stem>_int8.json so partial runs never replace it)")
    ap.add_argument("--quantized-only", action="store_true",
                    help="skip the float context (identical for every variant of the same parsed HAR)")
    a = ap.parse_args()
    out_name = a.out or ("hailo_int8.json" if not a.quantized_only and a.limit is None
                         else f"{Path(a.har).stem}_int8.json")

    from hailo_sdk_client import ClientRunner, InferenceContext
    runner = ClientRunner(har=a.har)
    imgs, gt = load_split(2 if a.probe else a.limit)
    results = {"har": a.har, "images": len(imgs), "instances": len(gt["annotations"])}
    contexts = [("fp_optimized", InferenceContext.SDK_FP_OPTIMIZED), ("quantized", InferenceContext.SDK_QUANTIZED)]
    for name, ctx_type in contexts[1:] if a.quantized_only else contexts:
        preds, t0 = [], time.time()
        with runner.infer_context(ctx_type) as ctx:
            out = runner.infer(ctx, image_dataset(imgs), batch_size=a.batch)
        if a.probe:
            print(name, "->", describe(out))
            return
        for k, dets in enumerate(to_dets(out, len(imgs))):
            preds += [{"image_id": k, "category_id": c, "bbox": b, "score": sc} for c, sc, b in dets]
        m = coco_eval(gt, preds)
        m["seconds"] = round(time.time() - t0, 1)
        results[name] = m
        print(name, m, flush=True)
    if "fp_optimized" in results:
        results["mAP50_drop"] = round(results["fp_optimized"]["mAP50"] - results["quantized"]["mAP50"], 4)
        results["mAP50-95_drop"] = round(results["fp_optimized"]["mAP50-95"] - results["quantized"]["mAP50-95"], 4)
    for d in OUT:
        d.mkdir(parents=True, exist_ok=True)
        (d / out_name).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
