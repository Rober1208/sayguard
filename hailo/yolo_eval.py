"""Shared helpers for comparing a YOLOv8 detector before and after Hailo compilation with ONE
evaluator: the same images, the same letterbox, the same NMS thresholds and pycocotools.

  letterbox()       image -> HxW network input (raw 0-255 RGB, grey 114 padding, as Ultralytics)
  load_yolo_split() YOLO-format images + labels -> COCO ground truth in original pixels
  onnx_preds()      the float ONNX on CPU with Hailo's NMS settings (the reference "PC float")
  hailo_preds()     HAILO NMS output (per class [y1, x1, y2, x2, score], normalised) -> COCO boxes
  coco_eval()       mAP50-95 / mAP50 overall and per class

The Hailo NMS keeps boxes with score >= nms_scores_th per class (Model Zoo default 0.2), so the
float reference here uses that threshold too; Ultralytics val (score >= 0.001) is a different
measurement and gives a higher mAP.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PAD = 114


def letterbox(img: Image.Image, h: int, w: int) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Resize keeping the aspect ratio, centre on grey; returns (HxWx3 float32 0-255, (scale, dx, dy))."""
    iw, ih = img.size
    s = min(w / iw, h / ih)
    nw, nh = round(iw * s), round(ih * s)
    dx, dy = (w - nw) // 2, (h - nh) // 2
    canvas = np.full((h, w, 3), PAD, np.float32)
    canvas[dy:dy + nh, dx:dx + nw] = np.asarray(img.convert("RGB").resize((nw, nh), Image.BILINEAR), np.float32)
    return canvas, (s, dx, dy)


def load_yolo_split(img_dir: Path, lbl_dir: Path, names: list[str], limit: int | None = None) -> tuple[list[Path], dict]:
    imgs = sorted(Path(img_dir).glob("*.jpg"))[:limit]
    gt = {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n in enumerate(names)]}
    aid = 1
    for i, p in enumerate(imgs):
        W, H = Image.open(p).size
        gt["images"].append({"id": i, "file_name": p.name, "width": W, "height": H})
        lbl = Path(lbl_dir) / (p.stem + ".txt")
        for line in (lbl.read_text().splitlines() if lbl.exists() else []):
            if not line.strip():
                continue
            c, cx, cy, bw, bh = line.split()[:5]
            bw, bh = float(bw) * W, float(bh) * H
            x, y = float(cx) * W - bw / 2, float(cy) * H - bh / 2
            gt["annotations"].append({"id": aid, "image_id": i, "category_id": int(c), "bbox": [x, y, bw, bh],
                                      "area": bw * bh, "iscrowd": 0})
            aid += 1
    return imgs, gt


def _unletterbox(x1, y1, x2, y2, meta, W, H) -> list[float]:
    s, dx, dy = meta
    x1, x2 = np.clip((np.array([x1, x2]) - dx) / s, 0, W)
    y1, y2 = np.clip((np.array([y1, y2]) - dy) / s, 0, H)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_th: float, keep_max: int) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size and len(keep) < keep_max:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_th]
    return keep


def onnx_preds(onnx_path: str, paths: list[Path], gt: dict, h: int, w: int, score_th: float, iou_th: float,
               max_per_class: int = 100) -> list[dict]:
    """Ultralytics ONNX export: output (1, 4 + classes, anchors), boxes cx, cy, w, h in input pixels."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    preds = []
    for k, p in enumerate(paths):
        img = Image.open(p)
        x, meta = letterbox(img, h, w)
        out = sess.run(None, {inp: (x / 255.0).transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        cxcywh, cls = out[:4].T, out[4:].T
        xyxy = np.stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                         cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2], 1)
        W, H = gt["images"][k]["width"], gt["images"][k]["height"]
        for c in range(cls.shape[1]):
            m = cls[:, c] >= score_th
            if not m.any():
                continue
            b, s = xyxy[m], cls[m, c]
            for i in _nms(b, s, iou_th, max_per_class):
                preds.append({"image_id": k, "category_id": c, "score": float(s[i]),
                              "bbox": _unletterbox(*b[i], meta, W, H)})
    return preds


def hailo_preds(out, metas: list, gt: dict, n_classes: int, h: int, w: int) -> list[dict]:
    """HAILO NMS by class: per image a list per class of [K, 5] (y1, x1, y2, x2, score) normalised to
    the network input, or arrays (N, C, K, 5) / (N, C, 5, K)."""
    if isinstance(out, dict):
        out = next(iter(out.values()))
    preds = []
    for k, meta in enumerate(metas):
        per = out[k]
        W, H = gt["images"][k]["width"], gt["images"][k]["height"]
        for c in range(n_classes):
            arr = np.asarray(per[c], dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.ndim == 2 and arr.shape[0] == 5 and arr.shape[1] != 5:
                arr = arr.T
            for y1, x1, y2, x2, s in arr.reshape(-1, 5):
                if s > 0:
                    preds.append({"image_id": k, "category_id": c, "score": float(s),
                                  "bbox": _unletterbox(x1 * w, y1 * h, x2 * w, y2 * h, meta, W, H)})
    return preds


def coco_eval(gt: dict, preds: list[dict], names: list[str]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    G = COCO()
    G.dataset = gt
    G.createIndex()
    if not preds:
        return {"mAP50-95": 0.0, "mAP50": 0.0}
    D = G.loadRes(preds)
    E = COCOeval(G, D, "bbox")
    E.evaluate(); E.accumulate(); E.summarize()
    out = {"mAP50-95": round(float(E.stats[0]), 4), "mAP50": round(float(E.stats[1]), 4)}
    present = {a["category_id"] for a in gt["annotations"]}
    for cid, name in enumerate(names):
        if cid not in present:
            continue
        e = COCOeval(G, D, "bbox")
        e.params.catIds = [cid]
        e.evaluate(); e.accumulate(); e.summarize()
        out[f"{name}_mAP50"] = round(float(e.stats[1]), 4)
    return out
