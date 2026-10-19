"""UGen300 backend (Stage II): the same YOLOv8s compiled to a HEF for Hailo-10H,
run through HailoRT's InferModel API.

The HEF comes from hailo/build_helmet.py (Hailo Dataflow Compiler 5.4, the Hailo Model Zoo
yolov8s recipe with 2 classes): the network runs on the chip in INT8; the box decoding and
per-class NMS (score >= 0.2, IoU 0.7, <= 100 boxes per class) are embedded in the HEF and run
by HailoRT on the host CPU (`nms_postprocess(..., engine=cpu)`), as in the Model Zoo build.

STATUS: the HEF is built and its INT8 accuracy checked in the DFC emulator, but this
module is written against the HailoRT 5.x Python API and NOT yet run on hardware (the
accelerator is shipped to Stage II teams).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..types import Detection

LABELS = ["helmet", "head"]
CROSS_CLASS_IOU = 0.6     # helmet and head boxes overlapping this much are the same head


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_cross_class(dets: list[Detection], iou_thr: float = CROSS_CLASS_IOU) -> list[Detection]:
    """Keep only the higher-scoring box where a helmet and a head box cover the same head.

    The HEF's NMS is per class, so an ambiguous head can come out as both 'helmet' and
    'head'; the tracker would make two people of it (the PC backend uses agnostic NMS for
    the same reason)."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: x.conf, reverse=True):
        if all(k.label == d.label or _iou(k.xyxy, d.xyxy) <= iou_thr for k in kept):
            kept.append(d)
    return kept


class HailoDetector:
    def __init__(self, hef_path: str, conf: float = 0.35, batch_size: int = 8, labels: list[str] | None = None):
        """labels: the HEF's class names in order (default: the hard-hat model's helmet/head)."""
        from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(params)
        self.model = self.vdevice.create_infer_model(hef_path)
        self.model.set_batch_size(batch_size)
        self.model.output().set_format_type(FormatType.FLOAT32)
        self.configured = self.model.configure()
        self.h, self.w = self.model.input().shape[:2]
        self.conf = conf
        self.batch_size = batch_size
        self.labels = list(labels) if labels else list(LABELS)
        self.ppe = self.labels == LABELS
        self.name = f"YOLOv8s-{'helmet' if self.ppe else 'site'} (HailoRT, Hailo-10H, batch {batch_size})"

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        H, W = img.shape[:2]
        s = min(self.w / W, self.h / H)
        nw, nh = int(round(W * s)), int(round(H * s))
        canvas = np.full((self.h, self.w, 3), 114, np.uint8)
        dx, dy = (self.w - nw) // 2, (self.h - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh))
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), s, dx, dy

    def infer(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        out: list[list[Detection]] = []
        for i in range(0, len(frames), self.batch_size):
            chunk = frames[i:i + self.batch_size]
            metas, bindings = [], []
            for f in chunk:
                x, s, dx, dy = self._letterbox(f)
                b = self.configured.create_bindings()
                b.input().set_buffer(np.ascontiguousarray(x))
                b.output().set_buffer(np.empty(self.model.output().shape, dtype=np.float32))
                bindings.append(b)
                metas.append((s, dx, dy))
            self.configured.run(bindings, 1000)
            for b, (s, dx, dy) in zip(bindings, metas):
                out.append(self._decode(b.output().get_buffer(), s, dx, dy))
        return out

    def _decode(self, nms_out, s: float, dx: int, dy: int) -> list[Detection]:
        # HAILO NMS by-class output: per class an array of [y1, x1, y2, x2, score], normalised.
        dets = []
        for cls, boxes in enumerate(nms_out):
            for y1, x1, y2, x2, score in np.asarray(boxes).reshape(-1, 5):
                if score < self.conf:
                    continue
                X1, Y1 = (x1 * self.w - dx) / s, (y1 * self.h - dy) / s
                X2, Y2 = (x2 * self.w - dx) / s, (y2 * self.h - dy) / s
                if cls < len(self.labels):
                    dets.append(Detection((float(X1), float(Y1), float(X2), float(Y2)), self.labels[cls], float(score)))
        # helmet and head are two states of one head; other classes are different objects
        return merge_cross_class(dets) if self.ppe else dets
