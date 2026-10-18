"""Convert the keremberke/hard-hat-detection COCO export (Roboflow "Hard Hats" v2,
CC BY 4.0) into an Ultralytics YOLO dataset with two classes:

    0 helmet   (head wearing a hard hat)
    1 head     (bare head = no hard hat)

Images are hard-linked (same NTFS volume) instead of copied.

    python scripts/prepare_dataset.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import work_dir  # noqa: E402

NAMES = ["helmet", "head"]
CAT_MAP = {"hardhat": 0, "no-hardhat": 1}
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_split(src_dir: Path, out: Path, split: str) -> Counter:
    ann = json.loads((src_dir / "_annotations.coco.json").read_text(encoding="utf-8"))
    cats = {c["id"]: CAT_MAP.get(c["name"]) for c in ann["categories"]}
    by_img: dict[int, list] = {}
    for a in ann["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    img_dir, lbl_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    for im in ann["images"]:
        W, H = im["width"], im["height"]
        lines = []
        for a in by_img.get(im["id"], []):
            cls = cats.get(a["category_id"])
            if cls is None:
                continue
            x, y, w, h = a["bbox"]
            if w < 2 or h < 2:
                continue
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
            stats[NAMES[cls]] += 1
        src = src_dir / im["file_name"]
        if not src.exists():
            stats["missing_image"] += 1
            continue
        link_or_copy(src, img_dir / im["file_name"])
        (lbl_dir / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        stats["images"] += 1
        stats["background_images"] += int(not lines)
    return stats


def main() -> None:
    raw = work_dir() / "datasets" / "raw" / "keremberke" / "ex"
    out = work_dir() / "datasets" / "helmet"
    report = {}
    for src_name, split in SPLITS.items():
        s = convert_split(raw / src_name, out, split)
        report[split] = dict(s)
        print(split, dict(s))
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: helmet\n  1: head\n", encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", out / "data.yaml")


if __name__ == "__main__":
    main()
