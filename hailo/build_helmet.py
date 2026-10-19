"""Helmet detector -> Hailo-10H HEF with the Dataflow Compiler (run in the WSL DFC venv).

Stages (each saves its own HAR so a later stage can be re-run alone):
  parse    ONNX -> helmet.har, cut at the six YOLOv8 head convs (NMS is added later on CPU)
  optimize model script + calibration images -> helmet_optimized.har (INT8); calibration is a
           fixed random sample (seed 0) of the validation split, never the test split
  compile  -> helmet{tag}[_perf].hef + ..._compiled_model.har (for the profiler); with --install
           the HEF is also copied to @work/models/helmet_yolov8s.hef, where detector.hef points

    python hailo/build_helmet.py parse
    python hailo/build_helmet.py optimize [--level 1] [--calib 512] [--extra "<alls line>"] [--tag _v2]
    python hailo/build_helmet.py compile [--tag _v2] [--performance] [--install]

Paths: SAYGUARD_WORK (default /mnt/d/ugen-work), HAILO_BUILD (~/hailo/build),
HAILO_MODEL_ZOO (~/hailo/hailo_model_zoo). Environment: hailo/setup_dfc.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

from hailo_sdk_client import ClientRunner  # import before TensorFlow (DFC GPU note)

import numpy as np
from PIL import Image

WORK = Path(os.environ.get("SAYGUARD_WORK", "/mnt/d/ugen-work"))       # same work dir as the app
ONNX = str(WORK / "models" / "helmet_yolov8s.onnx")
CALIB = WORK / "datasets" / "helmet" / "images" / "val"   # default calibration source
BUILD = Path(os.environ.get("HAILO_BUILD", Path.home() / "hailo" / "build"))
ZOO = Path(os.environ.get("HAILO_MODEL_ZOO", Path.home() / "hailo" / "hailo_model_zoo"))
NAME = "helmet"
HW = "hailo10h"
ZOO_NMS = ZOO / "hailo_model_zoo/cfg/postprocess_config/yolov8s_nms_config.json"
END_NODES = [f"/model.22/cv{b}.{i}/cv{b}.{i}.2/Conv" for i in range(3) for b in (2, 3)]
STRIDES = (8, 16, 32)


def output_map(runner: ClientRunner) -> dict[str, str]:
    """ONNX end node -> Hailo layer name (the HAR renames every layer)."""
    hn = runner.get_hn()
    hn = json.loads(hn) if isinstance(hn, str) else hn
    found = {}
    for lname, layer in hn["layers"].items():
        for orig in layer.get("original_names", []):
            if orig in END_NODES:
                found[orig] = lname.split("/")[-1]
    return found


def parse() -> None:
    runner = ClientRunner(hw_arch=HW)
    runner.translate_onnx_model(ONNX, NAME, start_node_names=["images"], end_node_names=END_NODES,
                                net_input_shapes={"images": [1, 3, 640, 640]})
    runner.save_har(str(BUILD / f"{NAME}.har"))
    m = output_map(runner)
    print(json.dumps(m, indent=1))
    if len(m) != 6:
        sys.exit(f"expected 6 head outputs, found {len(m)}")


def model_script(runner: ClientRunner, level: int, n_calib: int, extra: list[str] = (), tag: str = "") -> str:
    """Model Zoo yolov8s.alls with our layer names and a 2-class NMS config (+ `extra` lines)."""
    m = output_map(runner)
    nms = json.loads(ZOO_NMS.read_text())
    nms["classes"] = 2
    nms["bbox_decoders"] = [
        {"name": f"bbox_decoder{i}", "stride": s,
         "reg_layer": m[f"/model.22/cv2.{i}/cv2.{i}.2/Conv"],
         "cls_layer": m[f"/model.22/cv3.{i}/cv3.{i}.2/Conv"]}
        for i, s in enumerate(STRIDES)]
    nms_path = BUILD / f"{NAME}_nms_config.json"
    nms_path.write_text(json.dumps(nms, indent=1))
    cls = [d["cls_layer"] for d in nms["bbox_decoders"]]
    lines = ["normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])",
             *[f"change_output_activation({c}, sigmoid)" for c in cls],
             f'nms_postprocess("{nms_path}", meta_arch=yolov8, engine=cpu)',
             f"model_optimization_flavor(optimization_level={level})",
             # without this the DFC calibrates on its default 64 images whatever it is given
             f"model_optimization_config(calibration, calibset_size={n_calib})",
             "performance_param(optimize_for_power=True)", *extra]
    script = "\n".join(lines) + "\n"
    (BUILD / f"{NAME}{tag}.alls").write_text(script)
    return script


def calib_set(n: int, src: Path = CALIB) -> np.ndarray:
    """n images drawn at random (seed 0) from `src`, as raw 0-255 RGB at 640x640: the network
    normalizes on-chip (normalization1).  The dataset's images are all 640x640 already."""
    pool = sorted(src.glob("*.jpg"))
    if len(pool) < n:
        sys.exit(f"need {n} calibration images in {src}, found {len(pool)}")
    paths = random.Random(0).sample(pool, n)
    arr = np.empty((len(paths), 640, 640, 3), np.float32)
    for k, p in enumerate(paths):
        arr[k] = np.asarray(Image.open(p).convert("RGB").resize((640, 640)))
    print(f"calibration: {arr.shape} min {arr.min():.0f} max {arr.max():.0f} mean {arr.mean():.1f}", flush=True)
    return arr


def optimize(level: int, n: int, extra: list[str] = (), tag: str = "", src: Path = CALIB) -> None:
    runner = ClientRunner(har=str(BUILD / f"{NAME}.har"))
    script = model_script(runner, level, n, extra, tag)
    print(script, flush=True)
    runner.load_model_script(script)
    runner.optimize(calib_set(n, src))
    runner.save_har(str(BUILD / f"{NAME}{tag}_optimized.har"))


def compile_hef(tag: str = "", performance: bool = False, install: bool = False) -> None:
    runner = ClientRunner(har=str(BUILD / f"{NAME}{tag}_optimized.har"))
    if performance:
        # what `hailomz compile --performance` appends for a generic model script: the
        # allocator searches much longer for fewer contexts / higher FPS (the Model Zoo build)
        runner.load_model_script("performance_param(compiler_optimization_level=max)", append=True)
        tag += "_perf"
    hef = runner.compile()
    (BUILD / f"{NAME}{tag}.hef").write_bytes(hef)
    runner.save_har(str(BUILD / f"{NAME}{tag}_compiled_model.har"))
    print("HEF", len(hef) / 1e6, "MB:", BUILD / f"{NAME}{tag}.hef")
    if install:     # whichever variant was chosen becomes the HEF configs/site.yaml detector.hef names
        installed = WORK / "models" / "helmet_yolov8s.hef"
        shutil.copyfile(BUILD / f"{NAME}{tag}.hef", installed)
        print("installed as", installed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["parse", "optimize", "compile"])
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--calib", type=int, default=512, help="number of calibration images (calibset_size)")
    ap.add_argument("--calib-dir", type=Path, default=CALIB, help="folder of .jpg calibration images")
    ap.add_argument("--extra", action="append", default=[], help="extra model-script line (repeatable)")
    ap.add_argument("--tag", default="", help="suffix for this variant's .alls/.har/.hef files")
    ap.add_argument("--performance", action="store_true", help="compile: compiler_optimization_level=max")
    ap.add_argument("--install", action="store_true", help="compile: copy the HEF to @work/models/helmet_yolov8s.hef")
    a = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    {"parse": parse, "optimize": lambda: optimize(a.level, a.calib, a.extra, a.tag, a.calib_dir),
     "compile": lambda: compile_hef(a.tag, a.performance, a.install)}[a.stage]()


if __name__ == "__main__":
    main()
