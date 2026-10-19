"""Download one scene of NVIDIA's PhysicalAI-SmartSpaces dataset (MTMC_Tracking_2026).

Synthetic warehouses rendered in Omniverse / Isaac Sim: time-synchronized, calibrated 1080p30
cameras with overlapping views, and ground truth whose object_id is the same in every camera.
SayGuard uses it to show and measure the multi-camera features (one identity per worker on the
floor plan, distances in metres, one alert per person).

Licence: CC BY 4.0 (https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces). Credit:
"NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2026), CC BY 4.0". The data is never committed;
it goes to @work/datasets/smartspaces/<split>/<scene>/.  Depth maps (terabytes) are skipped.

    python scripts/fetch_smartspaces.py train/Warehouse_010                 # metadata + all videos
    python scripts/fetch_smartspaces.py train/Warehouse_010 --no-videos     # calibration, GT, map
    python scripts/fetch_smartspaces.py train/Warehouse_010 --cameras 0 1 2 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import work_dir  # noqa: E402

REPO = "nvidia/PhysicalAI-SmartSpaces"
SUBSET = "MTMC_Tracking_2026"
META = ("calibration.json", "map.png", "ground_truth.json")


def scene_dir(scene: str) -> Path:
    return work_dir() / "datasets" / "smartspaces" / scene


def fetch(filename: str, tries: int = 5) -> Path:
    """One file from the dataset repo into the local mirror (resumes; skips files already there)."""
    from huggingface_hub import hf_hub_download

    for i in range(tries):
        try:
            return Path(hf_hub_download(REPO, filename, repo_type="dataset",
                                        local_dir=work_dir() / "datasets" / "smartspaces_hf"))
        except Exception as e:          # flaky network: back off and retry
            if i == tries - 1:
                raise
            print(f"  retry {i + 1}/{tries - 1} after {type(e).__name__}: {e}", flush=True)
            time.sleep(5 * (i + 1))
    raise AssertionError("unreachable")


def list_videos(scene: str) -> list[str]:
    from huggingface_hub import HfApi

    items = HfApi().list_repo_tree(REPO, path_in_repo=f"{SUBSET}/{scene}/videos", repo_type="dataset")
    return sorted(i.path for i in items if i.path.endswith(".mp4"))


def link(src: Path, dst: Path) -> None:
    """Expose the mirrored file under the plain scene folder (hard link, or copy across drives)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scene", help="e.g. train/Warehouse_010 (see the dataset card's scene table)")
    ap.add_argument("--no-videos", action="store_true", help="only calibration, ground truth and map")
    ap.add_argument("--no-gt", action="store_true", help="skip ground_truth.json (~280 MB)")
    ap.add_argument("--cameras", type=int, nargs="*", help="camera numbers to fetch (default: all)")
    a = ap.parse_args()

    out = scene_dir(a.scene)
    for name in META:
        if name == "ground_truth.json" and a.no_gt:
            continue
        try:
            p = fetch(f"{SUBSET}/{a.scene}/{name}")
        except Exception as e:      # the real-world test scenes ship no ground truth
            print(f"{name}: not available ({type(e).__name__})")
            continue
        link(p, out / name)
        print(f"{name}: {p.stat().st_size / 1e6:.1f} MB")
    if not a.no_videos:
        vids = list_videos(a.scene)
        if a.cameras is not None:
            wanted = {f"Camera_{c:04d}.mp4" for c in a.cameras}
            vids = [v for v in vids if v.rsplit("/", 1)[-1] in wanted]
        for v in vids:
            p = fetch(v)
            link(p, out / "videos" / p.name)
            print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB", flush=True)
    print("scene folder:", out)


if __name__ == "__main__":
    main()
