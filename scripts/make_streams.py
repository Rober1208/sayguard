"""Turn the selected stock clips into ten 60-second "camera" streams.

The demo reads camXX.mp4 files as if they were RTSP cameras, so every stream gets
the same CCTV-like format: 960 px wide, 10 fps (the pipeline's processing rate),
no audio, short GOP so the looping reader restarts cleanly.  Clips shorter than
60 s are looped with -stream_loop.  Mixkit's free license requires no credit,
but the attribution table is written anyway so the sources are traceable.

    python scripts/make_streams.py            # skips streams that already exist
    python scripts/make_streams.py --force --cams cam03 cam07
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import resolve, work_dir  # noqa: E402

SECONDS = 60
VF = "scale=960:-2,fps=10"
ENCODE = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", "20",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
LICENSE = "Mixkit Stock Video Free License"
LICENSE_URL = "https://mixkit.co/license/#videoFree"


def probe_duration(path: Path) -> float | None:
    exe = shutil.which("ffprobe")
    if not exe or not path.exists():
        return None
    r = subprocess.run([exe, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, timeout=60)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def plan_input(start: float, duration: float | None, seconds: int = SECONDS) -> tuple[float, int]:
    """(seek, extra loops) so that `seconds` of video are available after the seek."""
    if not duration:
        return max(0.0, start), 0
    if duration >= seconds:
        return round(min(max(0.0, start), duration - seconds), 2), 0
    return 0.0, math.ceil(seconds / duration) - 1


def ffmpeg_cmd(src: Path, dst: Path, start: float, loops: int, seconds: int = SECONDS) -> list[str]:
    cmd = [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if loops:
        cmd += ["-stream_loop", str(loops)]
    if start > 0:
        cmd += ["-ss", f"{start:.2f}"]
    return cmd + ["-i", str(src), "-t", str(seconds), "-vf", VF, *ENCODE, str(dst)]


def encode(cmd: list[str], dst: Path) -> None:
    """Write to a .part file first so an interrupted run never leaves a 'finished' stream."""
    part = dst.with_name(dst.stem + ".part.mp4")
    r = subprocess.run([*cmd[:-1], str(part)], capture_output=True, text=True)
    if r.returncode != 0:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed ({r.returncode}): {r.stderr.strip()[-800:]}")
    os.replace(part, dst)


def load_manifest(raw: Path) -> dict[str, dict]:
    p = raw / "manifest.json"
    return {str(x["id"]): x for x in json.loads(p.read_text(encoding="utf-8"))} if p.exists() else {}


def attribution_md(rows: list[dict], manifest: dict[str, dict]) -> str:
    def cell(s: str) -> str:
        return str(s).replace("|", "\\|").strip()

    lines = [
        "# Demo video sources",
        "",
        f"The demo camera streams are cut from free stock clips on [Mixkit](https://mixkit.co/), used under the "
        f"[{LICENSE}]({LICENSE_URL}). Each stream is {SECONDS} s starting at the listed offset, scaled to 960 px "
        f"wide, 10 fps, audio removed (clips shorter than {SECONDS} s are looped).",
        "",
        "The footage itself is not redistributed in this repository; `scripts/fetch_videos.py`, "
        "`scripts/score_clips.py` and `scripts/make_streams.py` re-create the streams.",
        "",
        "| Camera | Title | Start (s) | Mixkit page | License |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        m = manifest.get(str(r["clip_id"]), {})
        page = m.get("page") or r.get("page", "")
        lines.append(f"| {r['cam']} | {cell(m.get('title') or r['title'])} | {float(r.get('start_s') or 0):g} | "
                     f"{page} | {LICENSE} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selection", default="@work/videos/selection.json")
    ap.add_argument("--raw", default="@work/videos/raw")
    ap.add_argument("--out", default="@work/videos/streams")
    ap.add_argument("--cams", nargs="*", help="only these cameras (default: all)")
    ap.add_argument("--force", action="store_true", help="re-encode existing streams")
    ap.add_argument("--dry-run", action="store_true", help="print the ffmpeg commands only")
    a = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")
    sel_path, raw, out = resolve(a.selection), resolve(a.raw), resolve(a.out)
    rows = json.loads(sel_path.read_text(encoding="utf-8"))
    manifest = load_manifest(raw)
    out.mkdir(parents=True, exist_ok=True)

    failed = []
    for r in rows:
        if a.cams and r["cam"] not in a.cams:
            continue
        src = Path(r["file"]) if Path(r["file"]).is_absolute() else raw / r["file"]
        dst = out / f"{r['cam']}.mp4"
        if dst.exists() and not a.force:
            print(f"{r['cam']}: exists, skip ({dst.name})")
            continue
        if not src.exists():
            print(f"{r['cam']}: missing source {src}")
            failed.append(r["cam"])
            continue
        start, loops = plan_input(float(r.get("start_s") or 0), probe_duration(src) or r.get("duration_s"))
        cmd = ffmpeg_cmd(src, dst, start, loops)
        if a.dry_run:
            print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
            continue
        try:
            encode(cmd, dst)
        except RuntimeError as e:
            print(f"{r['cam']}: {e}")
            failed.append(r["cam"])
            continue
        got = probe_duration(dst)
        warn = "" if got is None or got >= SECONDS - 1 else f"  WARNING: only {got:.1f}s"
        print(f"{r['cam']}: {src.name} @ {start:g}s (loops {loops}) -> {dst.name} "
              f"{dst.stat().st_size / 1e6:.1f} MB{warn}", flush=True)

    if a.dry_run:
        return
    md = attribution_md(rows, manifest)
    (out / "ATTRIBUTION.md").write_text(md, encoding="utf-8")
    docs = ROOT / "docs" / "VIDEO_SOURCES.md"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(md, encoding="utf-8")
    print("wrote", out / "ATTRIBUTION.md", "and", docs)
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
