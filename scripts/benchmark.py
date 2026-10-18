"""Throughput numbers for the write-up: what the PC prototype measures, and what
one UGen300 (Hailo-10H) should sustain according to Hailo's published figures.

The two are kept strictly apart.  PC rows are measured here (end-to-end
Ultralytics predict incl. pre/post-processing, FP16, 640x640 frames, the same
call the pipeline makes).  UGen300 rows are *estimates* from the Hailo Model Zoo
reference for YOLOv8s (configs/site.yaml: hailo_reference); our detector is that
architecture with a 2-class head, so Stage II replaces the estimate with a
measurement of the compiled HEF.  GenAI rows are Hailo's published zoo figures.

    python scripts/benchmark.py
    python scripts/benchmark.py --no-pc       # only regenerate the UGen300 tables (no GPU)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import load_config, resolve, work_dir  # noqa: E402

HAILO_SOURCE = "Hailo Model Zoo, Hailo-10H object detection, measured over PCIe Gen3 x4"
GENAI_SOURCE = "Hailo Model Zoo (GenAI / vision), Hailo-10H"
GENAI_REFERENCE = [
    {"model": "Whisper-Small", "role": "speech-to-text for spoken rules", "tps": 10.61, "ttft_s": None},
    {"model": "Qwen2.5-1.5B-Instruct", "role": "rule parsing, daily safety report", "tps": 7.35, "ttft_s": 0.37},
    {"model": "Qwen2-VL-2B-Instruct", "role": "VLM double-check of alerts", "tps": 7.04, "ttft_s": 1.02},
]
YOLO_WORLD_REFERENCE = {"model": "yolo_world_v2s", "role": "open-vocabulary objects (forklift, ...)",
                        "fps": 46.6}
# Typical output lengths in SayGuard, used only for the derived latency estimates.
TOKEN_ASSUMPTIONS = [
    ("Whisper-Small", "transcribe a ~5 s spoken rule", 25),
    ("Qwen2.5-1.5B-Instruct", "spoken rule -> rule JSON", 80),
    ("Qwen2-VL-2B-Instruct", "verify one alert (short yes/no + reason)", 16),
    ("Qwen2.5-1.5B-Instruct", "daily safety report", 350),
]


# ---------------------------------------------------------------- PC measurement

def load_frames(src: Path, n: int, size: int) -> tuple[list[np.ndarray], str]:
    """Real frames (one per clip, resized to size x size) so NMS sees real boxes;
    random noise only if no clip is readable."""
    import cv2

    frames = []
    for p in sorted(src.glob("*.mp4")) if src.is_dir() else []:
        cap = cv2.VideoCapture(str(p))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) // 2)
        ok, f = cap.read()
        cap.release()
        if ok:
            frames.append(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA))
        if len(frames) >= n:
            break
    if frames:
        return frames, f"{len(frames)} real frames from {src}"
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (size, size, 3), np.uint8) for _ in range(n)], "random noise frames"


def measure_pc(weights: Path, batches: list[int], imgsz: int, conf: float, warmup: int, iters: int,
               frames_src: Path) -> dict:
    import torch
    import ultralytics
    from ultralytics import YOLO

    cuda = torch.cuda.is_available()
    device, quantize = (0, 16) if cuda else ("cpu", None)
    model = YOLO(str(weights))
    frames, frames_desc = load_frames(frames_src, max(batches), imgsz)

    def run(batch: list[np.ndarray]) -> list:
        res = model.predict(batch, imgsz=imgsz, conf=conf, device=device, quantize=quantize, verbose=False)
        for r in res:            # the pipeline copies boxes to the CPU; include that cost
            r.boxes.data.cpu()
        return res

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    rows = []
    for bs in batches:
        batch = [frames[i % len(frames)] for i in range(bs)]
        for _ in range(warmup):
            run(batch)
        times, stages = [], {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        for _ in range(iters):
            t0 = time.perf_counter()
            res = run(batch)
            if cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            for k in stages:
                stages[k] += res[0].speed[k]
        mean = statistics.fmean(times)
        rows.append({"batch": bs, "ms_per_batch": round(mean, 2),
                     "ms_per_batch_p50": round(statistics.median(times), 2),
                     "ms_per_batch_p95": round(sorted(times)[max(0, math.ceil(0.95 * len(times)) - 1)], 2),
                     "ms_per_image": round(mean / bs, 2), "images_per_s": round(1000 * bs / mean, 1),
                     "stage_ms_per_image": {k: round(v / iters, 2) for k, v in stages.items()}})
        print(f"PC batch {bs:>2}: {mean:7.2f} ms/batch  {1000 * bs / mean:7.1f} img/s", flush=True)
    out = {
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "precision": "fp16" if quantize == 16 else "fp32",
        "weights": str(weights), "imgsz": imgsz, "conf": conf, "warmup": warmup, "iters": iters,
        "frames": frames_desc, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if cuda else None,
        "method": "wall clock around model.predict() + boxes to CPU, CUDA-synchronised",
        "rows": rows,
    }
    del model
    if cuda:
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- UGen300 estimate

def capacity_table(fps_b1: float, fps_b8: float, cameras: int, fps_list: list[float]) -> list[dict]:
    rows = []
    for f in fps_list:
        demand = cameras * f
        rows.append({"cameras": cameras, "fps_per_camera": f, "demand_fps": demand,
                     "load_batch8_pct": round(100 * demand / fps_b8, 1),
                     "load_batch1_pct": round(100 * demand / fps_b1, 1),
                     "headroom_fps_batch8": round(fps_b8 - demand, 1),
                     "headroom_pct_batch8": round(100 * (1 - demand / fps_b8), 1),
                     "max_cameras_batch8": int(fps_b8 // f), "max_cameras_batch1": int(fps_b1 // f)})
    return rows


def mixed_load(fps_b8: float, cameras: int, fps: float, ov_cameras: int, every_n: int) -> dict:
    """Detector on every camera plus YOLO-World every Nth frame on the cameras that have
    a spoken object rule, as a share of device time (assumes ideal time-slicing)."""
    det_pct = 100 * cameras * fps / fps_b8
    ov_fps = ov_cameras * fps / every_n
    ov_pct = 100 * ov_fps / YOLO_WORLD_REFERENCE["fps"]
    return {"cameras": cameras, "fps_per_camera": fps, "ov_cameras": ov_cameras, "ov_every_n_frames": every_n,
            "detector_pct": round(det_pct, 1), "yolo_world_fps": round(ov_fps, 2), "yolo_world_pct": round(ov_pct, 1),
            "total_pct": round(det_pct + ov_pct, 1)}


def genai_estimates() -> list[dict]:
    ref = {r["model"]: r for r in GENAI_REFERENCE}
    out = []
    for model, task, tokens in TOKEN_ASSUMPTIONS:
        r = ref[model]
        out.append({"model": model, "task": task, "output_tokens": tokens,
                    "est_seconds": round((r["ttft_s"] or 0.0) + tokens / r["tps"], 1)})
    return out


def pc_capacity(pc: dict | None, fps_list: list[float]) -> list[dict]:
    if not pc or not pc.get("rows"):
        return []
    best = max(pc["rows"], key=lambda r: r["images_per_s"])
    return [{"fps_per_camera": f, "max_cameras": int(best["images_per_s"] // f), "at_batch": best["batch"],
             "images_per_s": best["images_per_s"]} for f in fps_list]


# ---------------------------------------------------------------- markdown

def table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(str(c) for c in r) + " |" for r in rows)]


def render_md(rep: dict) -> str:
    L = ["# SayGuard throughput benchmark", "",
         f"Generated {rep['generated']} by `scripts/benchmark.py`. "
         "**Measured** = run on the PC prototype in this repo. **Reference** = Hailo's published "
         "Model Zoo figures (not measured by us). **Estimate** = derived from the reference numbers.", ""]
    pc = rep.get("pc")
    L += ["## 1. Detector on the PC prototype (measured)", ""]
    if pc:
        L += [f"Device: {pc['device']}, {pc['precision'].upper()}, `{Path(pc['weights']).name}`, "
              f"{pc['imgsz']}x{pc['imgsz']} input, {pc['frames']}; {pc['warmup']} warm-up + {pc['iters']} timed "
              f"iterations per batch size; {pc['method']}."
              + (f" Peak GPU memory {pc['peak_gpu_mem_mb']} MB." if pc.get("peak_gpu_mem_mb") else ""), ""]
        L += table(["Batch", "ms / batch (mean)", "p50", "p95", "ms / image", "images / s",
                    "pre / infer / post (ms per image)"],
                   [[r["batch"], r["ms_per_batch"], r["ms_per_batch_p50"], r["ms_per_batch_p95"], r["ms_per_image"],
                     r["images_per_s"], " / ".join(f"{v:.2f}" for v in r["stage_ms_per_image"].values())]
                    for r in pc["rows"]])
        if rep["pc_capacity"]:
            L += ["", "Cameras the PC could serve at its best measured throughput (measured):", ""]
            L += table(["FPS per camera", "Max cameras", "At batch", "images / s"],
                       [[f"{r['fps_per_camera']:g}", r["max_cameras"], r["at_batch"], r["images_per_s"]]
                        for r in rep["pc_capacity"]])
    else:
        L += ["Not measured in this run (`--no-pc`)."]
    h = rep["hailo_reference"]
    L += ["", "## 2. One UGen300 (Hailo-10H): detector capacity (estimate)", "",
          f"Reference: {h['model']} on Hailo-10H = **{h['fps_batch1']} FPS at batch 1, {h['fps_batch8']} FPS at "
          f"batch 8** ({HAILO_SOURCE}). SayGuard's detector is the same YOLOv8s at 640x640 with a 2-class head, so "
          "these figures are used as the estimate until Stage II measures the compiled HEF. Host-side decoding, "
          "NMS and tracking are not included.", ""]
    L += table(["Cameras", "FPS / camera", "Demand FPS", "Load @ batch 8", "Load @ batch 1",
                "Headroom @ batch 8", "Max cameras @ batch 8", "Max cameras @ batch 1"],
               [[r["cameras"], f"{r['fps_per_camera']:g}", f"{r['demand_fps']:g}", f"{r['load_batch8_pct']}%",
                 f"{r['load_batch1_pct']}%", f"{r['headroom_fps_batch8']} FPS ({r['headroom_pct_batch8']}%)",
                 r["max_cameras_batch8"], r["max_cameras_batch1"]] for r in rep["ugen300_capacity"]])
    m = rep["mixed_load"]
    L += ["", f"With spoken object rules on {m['ov_cameras']} cameras, YOLO-World runs every "
          f"{m['ov_every_n_frames']} frames there ({m['yolo_world_fps']} FPS at the reference "
          f"{YOLO_WORLD_REFERENCE['fps']} FPS): detector {m['detector_pct']}% + YOLO-World {m['yolo_world_pct']}% "
          f"= **{m['total_pct']}% of device time** (estimate; assumes ideal time-slicing, no model-switch "
          "overhead, and no GenAI model running at the same moment)."]
    L += ["", "## 3. Hailo-10H GenAI and vision models used by SayGuard (reference)", "",
          f"Source: {GENAI_SOURCE}. Not measured by us; the PC prototype runs the same models with PyTorch / Ollama.", ""]
    L += table(["Model", "Used for", "Throughput", "Time to first token"],
               [[r["model"], r["role"], f"{r['tps']} tokens/s", f"{r['ttft_s']} s" if r["ttft_s"] else "-"]
                for r in GENAI_REFERENCE]
               + [[YOLO_WORLD_REFERENCE["model"], YOLO_WORLD_REFERENCE["role"], f"{YOLO_WORLD_REFERENCE['fps']} FPS",
                   "-"]])
    L += ["", "Rough latency per task on the UGen300 (estimate = TTFT + output tokens / tokens per s; "
          "prompt/image encoding beyond the published TTFT is not modelled):", ""]
    L += table(["Model", "Task", "Output tokens (assumed)", "Est. seconds"],
               [[r["model"], r["task"], r["output_tokens"], r["est_seconds"]] for r in rep["genai_estimates"]])
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default="@work/models/helmet_yolov8s.pt")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 10])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--frames", default="@work/videos/raw", help="folder of clips to take test frames from")
    ap.add_argument("--cameras", type=int, default=10)
    ap.add_argument("--fps", type=float, nargs="+", default=[5, 10, 15])
    ap.add_argument("--ov-cameras", type=int, default=2, help="cameras with a spoken object rule")
    ap.add_argument("--no-pc", action="store_true", help="skip the PC measurement (keeps the last one)")
    ap.add_argument("--skip-resource-check", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    href = {"model": "yolov8s", "fps_batch1": 166, "fps_batch8": 252, **cfg.get("hailo_reference", {})}
    json_path = work_dir() / "results" / "benchmark.json"
    md_path = ROOT / "docs" / "results" / "benchmark.md"

    pc = None
    if a.no_pc:
        if json_path.exists():
            pc = json.loads(json_path.read_text(encoding="utf-8")).get("pc")
    else:
        if not a.skip_resource_check:
            from sayguard.resources import require
            require("benchmark (YOLOv8s)", gpu_gb=1.0, ram_gb=1.5)
        pc = measure_pc(resolve(a.weights), sorted(set(a.batches)), a.imgsz, a.conf, a.warmup, a.iters,
                        resolve(a.frames))

    every_n = int(cfg.get("open_vocab", {}).get("every_n_frames", 3))
    stream_fps = float(cfg.get("streams", {}).get("fps", 10))
    rep = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pc": pc,
        "pc_capacity": pc_capacity(pc, a.fps),
        "hailo_reference": {**href, "source": HAILO_SOURCE},
        "ugen300_capacity": capacity_table(float(href["fps_batch1"]), float(href["fps_batch8"]), a.cameras, a.fps),
        "mixed_load": mixed_load(float(href["fps_batch8"]), a.cameras, stream_fps, a.ov_cameras, every_n),
        "genai_reference": {"source": GENAI_SOURCE, "models": GENAI_REFERENCE, "vision": YOLO_WORLD_REFERENCE},
        "genai_estimates": genai_estimates(),
        "labels": {"pc": "measured on PC", "ugen300_capacity": "estimate from Hailo reference",
                   "mixed_load": "estimate from Hailo reference", "genai_reference": "Hailo reference",
                   "genai_estimates": "estimate from Hailo reference"},
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_md(rep), encoding="utf-8")
    print("wrote", json_path, "and", md_path)
    for r in rep["ugen300_capacity"]:
        print(f"UGen300 est. {r['cameras']} cams @ {r['fps_per_camera']:g} fps: {r['demand_fps']:g} FPS, "
              f"load {r['load_batch8_pct']}% (batch 8), max {r['max_cameras_batch8']} cams")


if __name__ == "__main__":
    main()
