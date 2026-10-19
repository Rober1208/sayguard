"""Cut one SmartSpaces scene into frame-locked demo streams for site mode.

Site mode fuses every camera of a scene on one floor plan, and the ground truth is indexed by
source frame, so each stream keeps exactly every Nth source frame: output frame k is source
frame N*k in every camera.  That is why frames are picked by index (select on n, timestamps
rewritten afterwards) rather than with fps=, which picks by timestamp and can slip by one, and
why each encode also checks that the decoder delivered every source frame without errors.
The file sources pace each camera by the rate OpenCV reports (the average rate), so streams
are made exactly constant-rate, and cameras whose average rates differ are reported: one of
them would play faster than the others and drift out of the lock.
Output: @work/videos/site/<scene>/Camera_XXXX.mp4 (1280 px, 10 fps for N = 3, H.264, no
audio) plus streams.json, which records the mapping (frame k = source frame step*k) for the
site pipeline.

Source: NVIDIA PhysicalAI-SmartSpaces MTMC_Tracking_2026, CC BY 4.0 (scripts/fetch_smartspaces.py).

    python scripts/make_site_streams.py train/Warehouse_010                 # every downloaded camera
    python scripts/make_site_streams.py train/Warehouse_010 --cameras 0 1 2 --step 3 --width 1280
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import work_dir  # noqa: E402

SOURCE = "NVIDIA PhysicalAI-SmartSpaces MTMC_Tracking_2026 (CC BY 4.0)"
ENCODE = ["-an", "-sn", "-dn", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", "20",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
# below-normal priority keeps the desktop (and the demo, if it is running) responsive; on POSIX
# main() lowers its own priority instead, which the children inherit
RUN = {"capture_output": True, "text": True, "encoding": "utf-8", "errors": "replace",
       **({"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS} if os.name == "nt" else {})}
DECODE_STATS = re.compile(r"Input stream #0:\d+ \(video\): (\d+) packets read.*?; (\d+) frames decoded; "
                          r"(\d+) decode errors")


def exe(name: str) -> str:
    return shutil.which(name) or name


def probe(path: Path, decode: bool = False) -> dict | None:
    """Width, height, frame rates and frame count of the first video stream.  The count comes from
    reading every packet (decode=False) or decoding every frame, never from the container index
    alone; "indexed" is the index's count, so a file cut short shows up as frames < indexed."""
    cmd = [exe("ffprobe"), "-v", "error", "-select_streams", "v:0",
           "-count_frames" if decode else "-count_packets",
           "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,"
           "nb_read_packets", "-of", "json", str(path)]
    try:
        r = subprocess.run(cmd, timeout=600, **RUN)
        s = json.loads(r.stdout)["streams"][0]
        frames = int(s["nb_read_frames" if decode else "nb_read_packets"])
        indexed = int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else frames
        avg = s.get("avg_frame_rate", "0/0")
        return {"width": int(s["width"]), "height": int(s["height"]), "fps": Fraction(s["r_frame_rate"]),
                "avg_fps": Fraction(avg) if not avg.endswith("/0") else None,
                "frames": frames, "indexed": indexed}
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, IndexError, ZeroDivisionError):
        return None


def stable(paths: list[Path], wait: float = 2.0) -> set[Path]:
    """Files whose size did not change over `wait` seconds (a copy may still be landing)."""
    def sizes() -> dict[Path, int]:
        return {p: p.stat().st_size for p in paths if p.exists()}

    before = sizes()
    if before:
        time.sleep(wait)
    after = sizes()
    return {p for p, n in after.items() if n > 0 and before.get(p) == n}


def check_source(src: Path, complete: set[Path]) -> dict | None:
    """Probe info if the file is complete.  A faststart MP4 cut short still probes fine (its index
    is at the front) and ffmpeg stops at the cut without an error, hence the packet count."""
    info = probe(src) if src in complete else None
    return info if info and 0 < info["frames"] == info["indexed"] else None


def ffmpeg_cmd(src: Path, dst: Path, step: int, fps: Fraction, width: int, frames: int,
               threads: int) -> list[str]:
    """lt(n, step*frames) drops a trailing partial group so every camera ends on the same frame,
    while the whole input is still decoded (so the decoder statistics cover every frame).  The
    selected frames keep their 1/source-fps durations, which the muxer uses for the last frame
    (OpenCV then reads 10.002 fps); fps= after setpts, where every timestamp already sits on its
    grid slot, only fixes the durations and never drops or repeats a frame."""
    rate = f"{fps.numerator}/{fps.denominator}"
    vf = (f"select='not(mod(n\\,{step}))*lt(n\\,{step * frames})',"
          f"setpts=N*{fps.denominator}/({fps.numerator}*TB),fps={rate},scale={width}:-2")
    return [exe("ffmpeg"), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "level+verbose", "-y",
            "-filter_threads", "1", "-threads", str(threads), "-i", str(src), "-map", "0:v:0", "-vf", vf,
            "-fps_mode", "passthrough", *ENCODE, "-threads", str(threads), str(dst)]


def encode(cmd: list[str], dst: Path, src_frames: int) -> str:
    """Write a .part file first so an interrupted run never leaves a 'finished' stream, and refuse
    the result if the decoder skipped or broke a frame (that would shift every later frame).
    Returns a warning when this ffmpeg does not report decoder statistics (older than 6.1)."""
    part = dst.with_name(dst.stem + ".part.mp4")
    r = subprocess.run([*cmd[:-1], str(part)], **RUN)
    m = DECODE_STATS.search(r.stderr)
    problem = None
    if r.returncode != 0:
        errors = [ln for ln in r.stderr.splitlines() if "[error]" in ln or "[fatal]" in ln]
        problem = f"ffmpeg failed ({r.returncode}): " + ("\n".join(errors) or r.stderr)[-800:]
    elif m and (int(m[2]) != src_frames or int(m[3])):
        problem = f"decoder delivered {m[2]}/{src_frames} frames with {m[3]} errors"
    if problem:
        part.unlink(missing_ok=True)
        raise RuntimeError(problem)
    os.replace(part, dst)
    return "" if m else "; WARNING: no decoder statistics, frame alignment unchecked"


def previous(out: Path) -> dict:
    """The last streams.json: it vouches for streams whose source is no longer available."""
    try:
        return json.loads((out / "streams.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def verify(dst: Path, step: int, width: int, src: dict | None, prev: dict) -> tuple[dict | None, str]:
    """(info, '') if dst has the requested width, source fps / step and exactly floor(source
    frames / step) decodable frames.  Without the source, the stream must be listed in the last
    streams.json for this step and width (an earlier run checked it against the source); its
    format alone cannot show which source frames it holds."""
    info = probe(dst, decode=True)
    if not info:
        return None, "unreadable"
    if info["width"] != width:
        return None, f"width {info['width']} != {width}"
    if src is None:
        if (dst.stem not in prev.get("cameras", ()) or (prev.get("step"), prev.get("width")) != (step, width)
                or float(info["fps"]) != prev.get("fps") or info["frames"] < prev.get("frames", 0)):
            return None, "source unavailable, and the last streams.json does not list it for these settings"
        return info, ""
    if info["fps"] != src["fps"] / step:
        return None, f"{info['fps']} fps != {src['fps'] / step}"
    if info["frames"] != src["frames"] // step:
        return None, f"{info['frames']} frames != {src['frames'] // step}"
    return info, ""


def make(src_path: Path, dst: Path, src: dict | None, a: argparse.Namespace, may_encode: bool,
         prev: dict) -> tuple[str, dict | None, str]:
    """(status, output info, message) for one camera; status is ok, skipped or failed.  --force
    never discards a stream whose source is unavailable, since it could not be made again."""
    if dst.exists() and not (a.force and may_encode and src):
        info, why = verify(dst, a.step, a.width, src, prev)
        if info:
            return "ok", info, "exists, verified" + ("" if src else " by the last streams.json (no source)")
        if not may_encode or src is None:
            return "failed", None, f"existing output fails verification ({why})"
    if src is None:
        state = "incomplete (still downloading?)" if src_path.exists() else "missing"
        return "skipped", None, f"source {state}: {src_path}"
    t0 = time.time()
    try:
        cmd = ffmpeg_cmd(src_path, dst, a.step, src["fps"] / a.step, a.width, src["frames"] // a.step,
                         a.threads)
        note = encode(cmd, dst, src["frames"])
    except (RuntimeError, OSError) as e:      # OSError: dst open elsewhere (a running demo) on Windows
        return "failed", None, str(e)
    info, why = verify(dst, a.step, a.width, src, prev)
    if not info:
        return "failed", None, f"output fails verification ({why})"
    return "ok", info, f"encoded in {time.time() - t0:.0f}s, {dst.stat().st_size / 1e6:.1f} MB{note}"


def number(x: Fraction) -> int | float:
    return int(x) if x.denominator == 1 else float(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scene", help="e.g. train/Warehouse_010, as given to fetch_smartspaces.py")
    ap.add_argument("--cameras", type=int, nargs="*", help="camera numbers (default: every downloaded one)")
    ap.add_argument("--step", type=int, default=3, help="keep every Nth source frame (3: 30 -> 10 fps)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--jobs", type=int, default=4, help="parallel ffmpeg processes")
    ap.add_argument("--threads", type=int, help="threads per ffmpeg (default: 3/4 of the CPUs over --jobs)")
    ap.add_argument("--force", action="store_true", help="re-encode outputs that already verify OK")
    a = ap.parse_args()
    if a.step < 1 or a.width < 2 or a.width % 2:
        sys.exit("--step must be >= 1 and --width a positive even number")
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        sys.exit("ffmpeg / ffprobe not found on PATH")
    if os.name != "nt":
        os.nice(10)
    a.jobs = max(1, a.jobs)
    a.threads = max(1, a.threads or (os.cpu_count() or 4) * 3 // (4 * a.jobs))

    src_dir = work_dir() / "datasets" / "smartspaces" / a.scene / "videos"
    out = work_dir() / "videos" / "site" / Path(a.scene).name
    if a.cameras:
        wanted = list(dict.fromkeys(f"Camera_{c:04d}" for c in a.cameras))
    else:
        wanted = sorted(p.stem for p in src_dir.glob("Camera_*.mp4"))
    if not wanted:
        sys.exit(f"no Camera_*.mp4 in {src_dir}")
    out.mkdir(parents=True, exist_ok=True)
    # outputs made by an earlier run (other --cameras) are re-verified so streams.json lists them too
    earlier = sorted(p.stem for p in out.glob("Camera_*.mp4")
                     if not p.stem.endswith(".part") and p.stem not in wanted)
    names = wanted + earlier
    complete = stable([src_dir / f"{n}.mp4" for n in names])
    prev = previous(out)

    results: dict[str, tuple[str, dict | None, str]] = {}
    print(f"{a.scene}: {len(wanted)} camera(s), step {a.step}, width {a.width}, "
          f"{a.jobs} job(s) x {a.threads} thread(s) -> {out}", flush=True)
    with ThreadPoolExecutor(a.jobs) as pool:
        sources = dict(zip(names, pool.map(lambda n: check_source(src_dir / f"{n}.mp4", complete), names)))
        futures = {pool.submit(make, src_dir / f"{n}.mp4", out / f"{n}.mp4", sources[n], a, n in wanted,
                               prev): n for n in names}
        for f in as_completed(futures):
            n = futures[f]
            results[n] = f.result()
            status, info, msg = results[n]
            frames = f", {info['frames']} frames {info['width']}x{info['height']}" if info else ""
            print(f"{n}: {status}{frames} - {msg}", flush=True)

    ok = {n: r[1] for n, r in sorted(results.items()) if r[0] == "ok"}
    if not ok:
        sys.exit("no verified streams")
    counts = {n: i["frames"] for n, i in ok.items()}
    # each verified stream runs at its source fps / step, so one output rate means one source rate
    rates = {i["fps"] for i in ok.values()}
    sizes = {(i["width"], i["height"]) for i in ok.values()}
    problems = [n for n, r in results.items() if r[0] != "ok"]
    if len(set(counts.values())) > 1:
        print("WARNING: frame counts differ, streams.json uses the shortest:", counts)
        problems.append("frame counts")
    # streams made before fps= was added have a short last frame and read as 10.002 fps; alone
    # that is harmless, mixed with exact ones the file sources would play cameras at two speeds
    if len({i["avg_fps"] for i in ok.values()}) > 1:
        old = [n for n, i in ok.items() if i["avg_fps"] != i["fps"]]
        print("WARNING: average frame rates differ between cameras; re-encode these with --force:", *old)
        problems.append("average frame rates")
    if len(rates) > 1 or len(sizes) > 1:
        sys.exit(f"cameras disagree on fps {rates} or size {sizes}; streams.json not written")
    (rate,), ((w, h),) = rates, sizes
    meta = {"scene": a.scene, "source_fps": number(rate * a.step), "step": a.step, "fps": number(rate),
            "width": w, "height": h, "frames": min(counts.values()), "cameras": list(ok), "source": SOURCE}
    (out / "streams.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {out / 'streams.json'}: {len(ok)} camera(s), {meta['frames']} frames at {meta['fps']} fps")
    if problems:
        sys.exit(f"not done: {', '.join(problems)}")


if __name__ == "__main__":
    main()
