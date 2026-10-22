"""Build a YOLO dataset for the site-mode detector from NVIDIA PhysicalAI-SmartSpaces scenes.

The warehouse scenes come with boxes for every person and vehicle in every camera, so a
detector can be trained on some scenes and measured on a scene it never saw.  Off-the-shelf
models were not usable there: on Warehouse_010 YOLO-World found 0 of 30 forklift boxes, COCO
YOLOv8s matched 7 as "truck", and took 39 % of the humanoid robots for people.

Classes:  0 person  1 forklift  2 pallet_truck  3 transporter  4 mobile_robot  5 humanoid_robot

Frames: every --step-th source frame of every camera (ffmpeg select, so output k == source
frame step*k and the ground truth lines up), scaled to --width.  Boxes are the "visible" 2D
boxes of the ground truth; boxes smaller than --min-px after scaling are dropped.

    python scripts/fetch_smartspaces.py train/Warehouse_000          # for each scene used
    python scripts/prepare_site_dataset.py --train train/Warehouse_000 train/Warehouse_004 ... \
        --val train/Warehouse_016

A scene may name cameras: train/Warehouse_016:8,9 uses only cameras 8 and 9, and
train/Warehouse_016:~8,9 all the others (so one scene can feed both splits).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import work_dir  # noqa: E402

NAMES = ["person", "forklift", "pallet_truck", "transporter", "mobile_robot", "humanoid_robot"]
TYPE_MAP = {"Person": 0, "Forklift": 1, "PalletTruck": 2, "Transporter": 3, "NovaCarter": 4,
            "FourierGR1T2": 5, "AgilityDigit": 5}


def load_gt(path: Path) -> dict[int, list[dict]]:
    """ground_truth.json -> {frame: [objects]}; the 2026 files spell keys with spaces."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(f): [{k.replace(" ", "_"): v for k, v in o.items()} for o in objs] for f, objs in raw.items()}


def parse_scene(spec: str) -> tuple[str, set[int] | None, bool]:
    """'train/Warehouse_016[:8,9 | :~8,9]' -> (scene, camera numbers or None, exclude?)."""
    scene, _, cams = spec.partition(":")
    scene = Path(scene).as_posix().strip("/").removeprefix("./")
    if not cams:
        return scene, None, False
    return scene, {int(x) for x in cams.lstrip("~").split(",") if x}, cams.startswith("~")


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg            # bundled with the ultralytics/imageio stack
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    return exe


def extract(video: Path, out_dir: Path, step: int, width: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "f%05d.jpg"
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-threads", "2", "-i", str(video),
                    "-vf", f"select='not(mod(n\\,{step}))',scale={width}:-2", "-fps_mode", "passthrough",
                    "-q:v", "3", "-start_number", "0", str(pattern)], check=True)
    return sorted(out_dir.glob("f*.jpg"))


def build_split(scenes: list[str], split: str, out: Path, step: int, width: int, min_px: float,
                stats: Counter) -> int:
    img_dir, lbl_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_img = 0
    for spec in scenes:
        scene, nums, exclude = parse_scene(spec)
        sdir = work_dir() / "datasets" / "smartspaces" / scene
        gt = load_gt(sdir / "ground_truth.json")
        videos = sorted((sdir / "videos").glob("Camera_*.mp4"))
        if nums is not None:
            videos = [v for v in videos if (int(v.stem.split("_")[1]) in nums) != exclude]
        if not videos:
            sys.exit(f"no videos in {sdir / 'videos'} (run scripts/fetch_smartspaces.py {scene})")
        tmp = out / "_frames" / scene.replace("/", "_")
        with ThreadPoolExecutor(max_workers=4) as pool:
            frames = dict(zip(videos, pool.map(lambda v: extract(v, tmp / v.stem, step, width), videos)))
        for video, files in frames.items():
            cam = video.stem
            for k, f in enumerate(files):
                src = k * step
                objs = gt.get(src, [])
                import cv2
                h, w = cv2.imread(str(f)).shape[:2]
                sx = w / 1920.0
                lines = []
                for o in objs:
                    cls = TYPE_MAP.get(o["object_type"])
                    box = o.get("2d_bounding_box_visible", {}).get(cam)
                    if cls is None:
                        stats[f"skipped type {o['object_type']}"] += 1
                        continue
                    if not box:
                        continue
                    x1, y1, x2, y2 = (v * sx for v in box)
                    if y2 - y1 < min_px or x2 - x1 < min_px / 3:
                        stats["dropped small"] += 1
                        continue
                    lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                    stats[f"{split}:{NAMES[cls]}"] += 1
                name = f"{scene.replace('/', '_')}_{cam}_{src:05d}"
                shutil.move(str(f), img_dir / f"{name}.jpg")
                (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                n_img += 1
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"{split}: {spec} done ({n_img} images so far)", flush=True)
    return n_img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--step", type=int, default=90, help="source frames between samples (90 = every 3 s)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--min-px", type=float, default=10.0)
    ap.add_argument("--name", default="site", help="dataset folder under @work/datasets (replaced)")
    a = ap.parse_args()
    # every scene scripts/eval_site.py uses, never trained on: validation W012/W020 (settings), test
    # W010/W013/W015/W018/W019/W022, and development scenes W009/W017 (twins of training/test rooms)
    held_out = {f"Warehouse_{n:03d}" for n in (9, 10, 12, 13, 15, 17, 18, 19, 20, 22)}
    for s in a.train + a.val:
        if Path(parse_scene(s)[0]).name in held_out:
            sys.exit(f"{s}: {sorted(held_out)} are site-mode evaluation scenes; keep them out of training")

    out = work_dir() / "datasets" / a.name
    if out.exists():
        shutil.rmtree(out)
    stats: Counter = Counter()
    n_tr = build_split(a.train, "train", out, a.step, a.width, a.min_px, stats)
    n_va = build_split(a.val, "val", out, a.step, a.width, a.min_px, stats)
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)), encoding="utf-8")
    info = {"train_scenes": a.train, "val_scenes": a.val, "step": a.step, "width": a.width,
            "images": {"train": n_tr, "val": n_va}, "boxes": dict(sorted(stats.items())),
            "source": "NVIDIA PhysicalAI-SmartSpaces MTMC_Tracking_2026 (CC BY 4.0)"}
    (out / "stats.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
