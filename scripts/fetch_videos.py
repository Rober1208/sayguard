"""List the Mixkit clips behind the ten demo cameras and check they were downloaded.

Mixkit's User Terms forbid scripts or bots that mass-download items, so this script
never touches the network: `list` prints each clip's page and the name to save it
under, the clips are downloaded by hand in a browser, and `check` confirms they are in
the raw folder make_streams.py reads.  When all ten are there, `check` also writes
raw/manifest.json (titles and pages for score_clips.py and make_streams.py) and, if
none exists yet, selection.json: the camera picks score_clips.py made for the demo, so
make_streams.py rebuilds the same streams without re-running the detectors.

The clips are not committed: the Mixkit Stock Video Free License covers using them,
not redistributing the footage itself.

    python scripts/fetch_videos.py list
    python scripts/fetch_videos.py check [--rename] [--inbox ~/Downloads]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard.config import resolve  # noqa: E402

LICENSE = "Mixkit Stock Video Free License"
LICENSE_URL = "https://mixkit.co/license/#videoFree"
MIN_BYTES = 100_000
MP4_BOXES = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide"}


@dataclass(frozen=True)
class Clip:
    cam: str
    camera_name: str
    zone: str
    id: str
    title: str
    page: str
    duration_s: float
    reason: str
    start_s: float = 0.0
    creator: str | None = None  # not recorded when the clips were picked; the licence needs no credit

    @property
    def file(self) -> str:
        return f"mixkit-{self.id}.mp4"


# The clips score_clips.py picked for cam01-cam10 in configs/site.yaml (1280x720 downloads).
CLIPS = [
    Clip("cam01", "Site Gate", "entrance", "1442", "Man on the phone near a construction site",
         "https://mixkit.co/free-stock-video/man-on-the-phone-near-a-construction-site-1442/", 11.0,
         "worker on the phone at the site gate, hard hat on"),
    Clip("cam02", "Site Office", "office", "42641", "Young people working together in the office",
         "https://mixkit.co/free-stock-video/young-people-working-together-in-the-office-42641/", 10.05,
         "office staff without hard hats: office is exempt from the hard-hat rule"),
    Clip("cam03", "Excavation Pit", "zone_a", "4069", "Excavation crane lifts a rock into a truck",
         "https://mixkit.co/free-stock-video/excavation-crane-lifts-a-rock-into-a-truck-4069/", 96.43,
         "excavator next to a worker: target for a spoken proximity rule"),
    Clip("cam04", "Scaffold", "zone_a", "1436", "Two construction workers",
         "https://mixkit.co/free-stock-video/two-construction-workers-1436/", 15.79,
         "two workers with hard hats"),
    Clip("cam05", "House Frame", "zone_b", "1459", "Construction workers at a house under construction",
         "https://mixkit.co/free-stock-video/construction-workers-at-a-house-under-construction-1459/", 9.0,
         "workers without hard hats (violation)"),
    Clip("cam06", "Welding Bay", "zone_b", "4125", "Welder working at a construction",
         "https://mixkit.co/free-stock-video/welder-working-at-a-construction-4125/", 13.67,
         "welder at a construction site"),
    Clip("cam07", "Warehouse Aisle", "warehouse", "4705", "Young worker doing inventory",
         "https://mixkit.co/free-stock-video/young-worker-doing-inventory-4705/", 15.45,
         "worker doing inventory without a hard hat (violation)"),
    Clip("cam08", "Loading Dock", "warehouse", "31346", "Courier worker preparing boxes on a loading truck",
         "https://mixkit.co/free-stock-video/courier-worker-preparing-boxes-on-a-loading-truck-31346/", 11.25,
         "courier loading boxes on a truck"),
    Clip("cam09", "Maintenance Shop", "workshop", "41937", "Mechanic pouring oil into an engine",
         "https://mixkit.co/free-stock-video/mechanic-pouring-oil-into-an-engine-41937/", 8.13,
         "mechanic without a hard hat (violation)"),
    Clip("cam10", "Site Walk", "zone_a", "4648", "Architect talking to businessmen",
         "https://mixkit.co/free-stock-video/architect-talking-to-businessmen-4648/", 17.64,
         "architect walking the site with visitors, hard hats on"),
]


def problem(path: Path) -> str | None:
    """Why `path` is not a usable clip; saving the web page instead of the video gives HTML."""
    if not path.is_file():
        return "missing"
    if path.stat().st_size < MIN_BYTES:
        return f"only {path.stat().st_size} bytes"
    with open(path, "rb") as f:
        if f.read(8)[4:8] not in MP4_BOXES:
            return "not an MP4 file"
    return None


def candidates(clip: Clip, dirs: list[Path]) -> list[Path]:
    """Downloads still under the browser's name: any MP4 whose name holds the clip id as a whole number."""
    pat = re.compile(rf"(?<!\d){clip.id}(?!\d)")
    found = {p.resolve() for d in dirs if d.is_dir() for p in d.glob("*.mp4")
             if p.name != clip.file and pat.search(p.stem) and problem(p) is None}
    return sorted(found)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def write_manifest(raw: Path) -> Path:
    """Merge rather than replace: score_clips.py takes titles from it for every clip in the folder."""
    path = raw / "manifest.json"
    items = {str(x["id"]): x for x in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
    for c in CLIPS:
        items.setdefault(c.id, {}).update(id=c.id, title=c.title, page=c.page, creator=c.creator,
                                           license=f"{LICENSE} ({LICENSE_URL})")
    write_json(path, sorted(items.values(), key=lambda x: str(x["id"]).zfill(12)))
    return path


def selection_rows() -> list[dict]:
    """The fields score_clips.py writes (minus its score), so make_streams.py reads either file."""
    return [{"cam": c.cam, "camera_name": c.camera_name, "zone": c.zone, "clip_id": c.id, "file": c.file,
             "title": c.title, "page": c.page, "duration_s": c.duration_s, "start_s": c.start_s,
             "reason": c.reason} for c in CLIPS]


def list_clips(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    print(f"Open each page, click Download (the demo used 720p; any resolution works) and save\n"
          f"the video into {raw} under the name shown.\n"
          f"Then run: python scripts/fetch_videos.py check\n")
    for c in CLIPS:
        print(f"{c.cam}  save as {c.file:<17} {c.title}\n       {c.page}")
    print(f"\nLicence: {LICENSE} ({LICENSE_URL}); no credit required. Do not commit the clips.")


def check(raw: Path, selection: Path, inbox: list[Path], rename: bool, force: bool) -> int:
    print(f"raw folder: {raw}")
    bad = 0
    for c in CLIPS:
        dst = raw / c.file
        why = problem(dst)
        found = candidates(c, [raw, *inbox]) if why else []
        if rename and len(found) == 1:
            src, same = found[0], found[0].parent == raw.resolve()
            raw.mkdir(parents=True, exist_ok=True)
            if same:
                src.replace(dst)
            else:
                shutil.copy2(src, dst)
            print(f"{'renamed' if same else 'copied':<8} {c.cam}  {src} -> {dst.name}")
            why, found = problem(dst), []
        if why is None:
            print(f"ok       {c.cam}  {dst.name}  {dst.stat().st_size / 1e6:.1f} MB")
            continue
        bad += 1
        print(f"{'MISSING' if why == 'missing' else 'BAD':<8} {c.cam}  {dst.name}"
              f"{'' if why == 'missing' else ': ' + why + ', download it again'}\n         {c.page}")
        if len(found) == 1:
            print(f"         found {found[0]}  (re-run with --rename to use it)")
        elif found:
            print(f"         found {len(found)} files with this id, rename one by hand: "
                  + ", ".join(p.name for p in found))
    if bad:
        print(f"\n{bad} of {len(CLIPS)} clips missing or unusable: "
              f"download them from the pages above into {raw}.")
        return 1
    print("wrote", write_manifest(raw))
    if selection.exists() and not force:
        print(f"kept {selection} (from score_clips.py or an earlier check; --force replaces it)")
    else:
        write_json(selection, selection_rows())
        print("wrote", selection)
    print(f"all {len(CLIPS)} clips present; next: python scripts/make_streams.py")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["list", "check"])
    ap.add_argument("--raw", default="@work/videos/raw", help="folder make_streams.py reads the clips from")
    ap.add_argument("--selection", default="@work/videos/selection.json")
    ap.add_argument("--inbox", action="append", default=[],
                    help="also look here for clips saved under another name, e.g. Downloads")
    ap.add_argument("--rename", action="store_true",
                    help="give such a clip its expected name in the raw folder, replacing a missing or "
                         "broken one (an --inbox file is copied)")
    ap.add_argument("--force", action="store_true", help="replace an existing selection.json")
    a = ap.parse_args()
    raw = resolve(a.raw)
    if a.cmd == "list":
        list_clips(raw)
    else:
        sys.exit(check(raw, resolve(a.selection), [Path(p).expanduser() for p in a.inbox], a.rename, a.force))


if __name__ == "__main__":
    main()
