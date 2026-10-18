"""Discover and download free stock clips from Mixkit for the multi-camera demo.

Only clips whose JSON-LD says copyrightNotice == "Free" (Mixkit Stock Video Free
License) are kept; "Mixkit Restricted License" items are skipped.  The raw clips
are NOT committed to git (redistributing stock footage is not allowed); this
script plus the generated manifest lets anyone re-download the same files.

    python scripts/fetch_videos.py discover
    python scripts/fetch_videos.py download [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import work_dir  # noqa: E402

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")
CATEGORIES = [
    "construction", "construction-site", "construction-worker", "worker", "workers", "factory",
    "factory-worker", "warehouse", "industry", "industrial", "engineer", "architect", "builder",
    "carpenter", "welder", "welding", "mechanic", "technician", "helmet", "safety", "forklift",
    "building", "workshop", "manufacturing", "logistics", "labor", "job", "work", "tools", "crane",
]
LD_RE = re.compile(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', re.S)


def http_get(url: str, retries: int = 6, timeout: int = 30) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network is flaky here; back off and retry
            last = e
            time.sleep(2 + 2 * i)
    raise RuntimeError(f"GET failed after {retries} tries: {url}: {last}")


def iter_video_objects(node):
    if isinstance(node, dict):
        if node.get("@type") == "VideoObject" and node.get("contentUrl"):
            yield node
        for v in node.values():
            yield from iter_video_objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from iter_video_objects(v)


def discover(out: Path) -> list[dict]:
    found: dict[str, dict] = {}
    for cat in CATEGORIES:
        for page in (1, 2, 3):
            url = f"https://mixkit.co/free-stock-video/{cat}/" + (f"?page={page}" if page > 1 else "")
            try:
                html = http_get(url).decode("utf-8", "ignore")
            except RuntimeError as e:
                print("skip", e)
                continue
            n_before = len(found)
            for block in LD_RE.findall(html):
                try:
                    data = json.loads(block)
                except json.JSONDecodeError:
                    continue
                for vo in iter_video_objects(data):
                    m = re.search(r"/videos/(\d+)/", vo["contentUrl"])
                    if not m or vo.get("copyrightNotice") != "Free":
                        continue
                    vid = m.group(1)
                    found.setdefault(vid, {
                        "id": vid,
                        "title": vo.get("name", ""),
                        "description": vo.get("description", ""),
                        "page": vo.get("@id", "").split("#")[0],
                        "license": "Mixkit Stock Video Free License (https://mixkit.co/license/#videoFree)",
                        "content_url": vo["contentUrl"],
                        "category": cat,
                    })
            print(f"{cat} p{page}: +{len(found) - n_before} (total {len(found)})")
            time.sleep(1)
    items = sorted(found.values(), key=lambda x: int(x["id"]))
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    return items


KEYWORDS = re.compile(
    r"construct|worker|engineer|helmet|hard ?hat|factory|warehouse|forklift|site|"
    r"build|industr|workshop|manufactur|weld|crane|scaffold|labor|technician|safety",
    re.I)
SKIP = re.compile(r"aerial|drone|time.?lapse|close.?up|blueprint|hands? |hand$|"
                  r"plan|laptop|tablet|meeting|night sky|sunset|motorcycl|motorbike|bike|wheelie|"
                  r"desert|street|driving|road|car |cars|rider|racing|sport", re.I)


def download(manifest: Path, raw_dir: Path, limit: int) -> None:
    items = json.loads(manifest.read_text(encoding="utf-8"))
    picked = [x for x in items
              if KEYWORDS.search(x["title"] + " " + x["description"])
              and not SKIP.search(x["title"])]
    print(f"{len(picked)} relevant of {len(items)} free clips; downloading up to {limit}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch_one(x: dict) -> None:
        dst = raw_dir / f"mixkit-{x['id']}.mp4"
        if dst.exists() and dst.stat().st_size > 100_000:
            return
        for q in ("720", "1080"):
            url = f"https://assets.mixkit.co/videos/{x['id']}/{x['id']}-{q}.mp4"
            try:
                dst.write_bytes(http_get(url, timeout=120))
                print(f"ok  {dst.name}  {q}p  {dst.stat().st_size / 1e6:.1f} MB  {x['title']}", flush=True)
                return
            except RuntimeError as e:
                print("miss", q, e, flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch_one, picked[:limit]))
    (raw_dir / "manifest.json").write_text(json.dumps(picked[:limit], ensure_ascii=False, indent=1),
                                           encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["discover", "download", "all"])
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()
    vids = work_dir() / "videos"
    manifest = vids / "mixkit_candidates.json"
    vids.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("discover", "all"):
        discover(manifest)
    if a.cmd in ("download", "all"):
        download(manifest, vids / "raw", a.limit)


if __name__ == "__main__":
    main()
