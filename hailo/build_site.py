"""Warehouse (site) detector -> Hailo-10H HEF with the Dataflow Compiler, with the same recipe as
the hard-hat detector (hailo/build_helmet.py): cut at the six YOLOv8 head convs, NMS on the
host, optimization level 1, 512 calibration images, activations clipped to the 0.01-99.99
percentiles.  The input can be the square 960x960 the PC prototype uses or a 16:9 544x960
variant (less work per 1280x720 camera frame).

Calibration images come only from the detector's TRAINING split (site_v3, 8 SmartSpaces training
scenes): a fixed random sample (seed 0), letterboxed exactly like the evaluation input.  No
evaluation scene is ever used.

    python hailo/build_site.py parse    [--onnx PATH] [--tag _r544]
    python hailo/build_site.py optimize [--tag _r544] [--calib 512]
    python hailo/build_site.py compile  [--tag _r544] [--performance]

Paths: SAYGUARD_WORK (default /mnt/d/ugen-work), HAILO_BUILD (~/hailo/build),
HAILO_MODEL_ZOO (~/hailo/hailo_model_zoo).  Environment: hailo/setup_dfc.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from hailo_sdk_client import ClientRunner  # import before TensorFlow (DFC GPU note)

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yolo_eval import letterbox  # noqa: E402

WORK = Path(os.environ.get("SAYGUARD_WORK", "/mnt/d/ugen-work"))
CALIB = WORK / "datasets" / "site_v3" / "images" / "train"
BUILD = Path(os.environ.get("HAILO_BUILD", Path.home() / "hailo" / "build"))
ZOO = Path(os.environ.get("HAILO_MODEL_ZOO", Path.home() / "hailo" / "hailo_model_zoo"))
NAME = "site"
HW = "hailo10h"
N_CLASSES = 6
ZOO_NMS = ZOO / "hailo_model_zoo/cfg/postprocess_config/yolov8s_nms_config.json"
END_NODES = [f"/model.22/cv{b}.{i}/cv{b}.{i}.2/Conv" for i in range(3) for b in (2, 3)]
STRIDES = (8, 16, 32)
CLIP = "pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])"


def input_hw(onnx_path: str) -> tuple[int, int]:
    import onnx
    dims = onnx.load(onnx_path).graph.input[0].type.tensor_type.shape.dim
    return int(dims[2].dim_value), int(dims[3].dim_value)


def meta_path(tag: str) -> Path:
    return BUILD / f"{NAME}{tag}.json"


def output_map(runner: ClientRunner) -> dict[str, str]:
    hn = runner.get_hn()
    hn = json.loads(hn) if isinstance(hn, str) else hn
    found = {}
    for lname, layer in hn["layers"].items():
        for orig in layer.get("original_names", []):
            if orig in END_NODES:
                found[orig] = lname.split("/")[-1]
    return found


def parse(onnx_path: str, tag: str) -> None:
    h, w = input_hw(onnx_path)
    runner = ClientRunner(hw_arch=HW)
    runner.translate_onnx_model(onnx_path, NAME, start_node_names=["images"], end_node_names=END_NODES,
                                net_input_shapes={"images": [1, 3, h, w]})
    runner.save_har(str(BUILD / f"{NAME}{tag}.har"))
    m = output_map(runner)
    print(json.dumps(m, indent=1))
    if len(m) != 6:
        sys.exit(f"expected 6 head outputs, found {len(m)}")
    meta_path(tag).write_text(json.dumps({"onnx": onnx_path, "h": h, "w": w}), encoding="utf-8")
    print(f"parsed {onnx_path}: input {h}x{w}")


def model_script(runner: ClientRunner, tag: str, h: int, w: int, n_calib: int) -> str:
    m = output_map(runner)
    nms = json.loads(ZOO_NMS.read_text())
    nms["classes"] = N_CLASSES
    nms["image_dims"] = [h, w]
    nms["bbox_decoders"] = [
        {"name": f"bbox_decoder{i}", "stride": s,
         "reg_layer": m[f"/model.22/cv2.{i}/cv2.{i}.2/Conv"],
         "cls_layer": m[f"/model.22/cv3.{i}/cv3.{i}.2/Conv"]}
        for i, s in enumerate(STRIDES)]
    nms_path = BUILD / f"{NAME}{tag}_nms_config.json"
    nms_path.write_text(json.dumps(nms, indent=1))
    cls = [d["cls_layer"] for d in nms["bbox_decoders"]]
    lines = ["normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])",
             *[f"change_output_activation({c}, sigmoid)" for c in cls],
             f'nms_postprocess("{nms_path}", meta_arch=yolov8, engine=cpu)',
             "model_optimization_flavor(optimization_level=1)",
             f"model_optimization_config(calibration, calibset_size={n_calib})",
             CLIP,
             "performance_param(optimize_for_power=True)"]
    script = "\n".join(lines) + "\n"
    (BUILD / f"{NAME}{tag}.alls").write_text(script)
    return script


def calib_set(n: int, h: int, w: int, src: Path = CALIB) -> np.ndarray:
    pool = sorted(src.glob("*.jpg"))
    if len(pool) < n:
        sys.exit(f"need {n} calibration images in {src}, found {len(pool)}")
    paths = random.Random(0).sample(pool, n)
    arr = np.empty((n, h, w, 3), np.float32)
    for k, p in enumerate(paths):
        arr[k] = letterbox(Image.open(p), h, w)[0]
    print(f"calibration: {arr.shape} from {src} (seed 0) mean {arr.mean():.1f}", flush=True)
    return arr


def optimize(tag: str, n: int, src: Path) -> None:
    meta = json.loads(meta_path(tag).read_text())
    runner = ClientRunner(har=str(BUILD / f"{NAME}{tag}.har"))
    script = model_script(runner, tag, meta["h"], meta["w"], n)
    print(script, flush=True)
    runner.load_model_script(script)
    runner.optimize(calib_set(n, meta["h"], meta["w"], src))
    runner.save_har(str(BUILD / f"{NAME}{tag}_optimized.har"))


def compile_hef(tag: str, performance: bool) -> None:
    runner = ClientRunner(har=str(BUILD / f"{NAME}{tag}_optimized.har"))
    if performance:
        runner.load_model_script("performance_param(compiler_optimization_level=max)", append=True)
        tag += "_perf"
    hef = runner.compile()
    (BUILD / f"{NAME}{tag}.hef").write_bytes(hef)
    runner.save_har(str(BUILD / f"{NAME}{tag}_compiled_model.har"))
    print("HEF", len(hef) / 1e6, "MB:", BUILD / f"{NAME}{tag}.hef")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["parse", "optimize", "compile"])
    ap.add_argument("--onnx", default=str(WORK / "models" / "site_yolov8s.onnx"))
    ap.add_argument("--tag", default="", help="suffix for this variant's files, e.g. _r544")
    ap.add_argument("--calib", type=int, default=512)
    ap.add_argument("--calib-dir", type=Path, default=CALIB)
    ap.add_argument("--performance", action="store_true")
    a = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    {"parse": lambda: parse(a.onnx, a.tag),
     "optimize": lambda: optimize(a.tag, a.calib, a.calib_dir),
     "compile": lambda: compile_hef(a.tag, a.performance)}[a.stage]()


if __name__ == "__main__":
    main()
