English | [繁體中文](VIDEO_SOURCES.zh-TW.md)

# Demo video sources

SayGuard uses two kinds of footage, one per mode: ten stock clips for hard-hat mode, and
synchronized, calibrated multi-camera warehouse scenes for warehouse mode. Neither is in this
repository; the scripts below fetch or check the footage and turn it into demo streams.

## Hard-hat mode: ten Mixkit clips

The ten demo camera streams are cut from free stock clips on [Mixkit](https://mixkit.co/), used under the [Mixkit Stock Video Free License](https://mixkit.co/license/#videoFree). The licence asks for no credit; the clips are listed here so the sources are traceable. Each stream is 60 s starting at the listed offset, scaled to 960 px wide, 10 fps, audio removed (clips shorter than 60 s are looped).

| Camera | Name | Mixkit clip | Save as | Start (s) | License |
|---|---|---|---|---|---|
| cam01 | Site Gate | [Man on the phone near a construction site](https://mixkit.co/free-stock-video/man-on-the-phone-near-a-construction-site-1442/) | `mixkit-1442.mp4` | 0 | Mixkit Stock Video Free License |
| cam02 | Site Office | [Young people working together in the office](https://mixkit.co/free-stock-video/young-people-working-together-in-the-office-42641/) | `mixkit-42641.mp4` | 0 | Mixkit Stock Video Free License |
| cam03 | Excavation Pit | [Excavation crane lifts a rock into a truck](https://mixkit.co/free-stock-video/excavation-crane-lifts-a-rock-into-a-truck-4069/) | `mixkit-4069.mp4` | 0 | Mixkit Stock Video Free License |
| cam04 | Scaffold | [Two construction workers](https://mixkit.co/free-stock-video/two-construction-workers-1436/) | `mixkit-1436.mp4` | 0 | Mixkit Stock Video Free License |
| cam05 | House Frame | [Construction workers at a house under construction](https://mixkit.co/free-stock-video/construction-workers-at-a-house-under-construction-1459/) | `mixkit-1459.mp4` | 0 | Mixkit Stock Video Free License |
| cam06 | Welding Bay | [Welder working at a construction](https://mixkit.co/free-stock-video/welder-working-at-a-construction-4125/) | `mixkit-4125.mp4` | 0 | Mixkit Stock Video Free License |
| cam07 | Warehouse Aisle | [Young worker doing inventory](https://mixkit.co/free-stock-video/young-worker-doing-inventory-4705/) | `mixkit-4705.mp4` | 0 | Mixkit Stock Video Free License |
| cam08 | Loading Dock | [Courier worker preparing boxes on a loading truck](https://mixkit.co/free-stock-video/courier-worker-preparing-boxes-on-a-loading-truck-31346/) | `mixkit-31346.mp4` | 0 | Mixkit Stock Video Free License |
| cam09 | Maintenance Shop | [Mechanic pouring oil into an engine](https://mixkit.co/free-stock-video/mechanic-pouring-oil-into-an-engine-41937/) | `mixkit-41937.mp4` | 0 | Mixkit Stock Video Free License |
| cam10 | Site Walk | [Architect talking to businessmen](https://mixkit.co/free-stock-video/architect-talking-to-businessmen-4648/) | `mixkit-4648.mp4` | 0 | Mixkit Stock Video Free License |

### Getting the clips

The footage is not in this repository: the licence covers using the clips, not redistributing them. Mixkit's terms also forbid scripts or bots that mass-download items, so nothing here downloads them for you. Download the ten clips by hand:

1. `python scripts/fetch_videos.py list` prints the pages above, the name to save each clip under and the raw folder (`<work dir>/videos/raw`; the work dir is `D:/ugen-work` unless `SAYGUARD_WORK` says otherwise).
2. Open each page in a browser, click Download (the demo used the 720p file; any resolution works, the streams are scaled to 960 px) and save the video into the raw folder as the name in "Save as", e.g. `mixkit-1442.mp4`.
3. `python scripts/fetch_videos.py check` reports missing or broken files. If the browser kept Mixkit's own file name (or the file under the expected name turns out to be a saved web page), leave the downloaded video in the raw folder or point `--inbox` at the download folder and add `--rename`: a file whose name contains the clip id is given the expected name (copied when it comes from `--inbox`). Once all ten are present, `check` writes `raw/manifest.json` and, unless one already exists, `videos/selection.json` with the camera assignment above (`--force` replaces it).
4. `python scripts/make_streams.py` turns them into `videos/streams/cam01.mp4` … `cam10.mp4`.

To try other footage, save any Mixkit clip under the Stock Video Free License (not the Restricted License) into the raw folder as `mixkit-<id>.mp4` and run `python scripts/score_clips.py`: it runs the demo's detectors over every clip in the folder and rewrites `selection.json` with the ten best fits for the cameras in `configs/site.yaml`.

## Warehouse mode: NVIDIA PhysicalAI-SmartSpaces scenes

Warehouse mode uses **NVIDIA PhysicalAI-SmartSpaces, MTMC_Tracking_2026**: synthetic
warehouses filmed by time-synchronized, calibrated 1080p 30 fps cameras with overlapping views,
with floor plans and ground truth, licensed **CC BY 4.0**
([dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)). Credit:
"NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2026), CC BY 4.0"; changes: every 3rd frame kept
(10 fps), scaled to 1280 px, SayGuard overlays and face pixelation added. The live demo
(`configs/warehouse.yaml`) plays scene `train/Warehouse_010`; the other scenes below are used
for evaluation only. Every SmartSpaces scene used, including the detector-training scenes, is
listed with its role in [LICENSES.md](LICENSES.md#nvidia-physicalai-smartspaces-scenes-used).

### Scenes made into streams

| Scene | Role | Cameras | Stream frames (10 fps) | Config |
|---|---|---|---|---|
| `train/Warehouse_010` | live demo and test | 11 | 3000 | `configs/warehouse.yaml` |
| `train/Warehouse_013` | test | 10 | 3000 | `configs/eval/warehouse_013.yaml` |
| `train/Warehouse_015` | test | 10 | 3000 | `configs/eval/warehouse_015.yaml` |
| `train/Warehouse_018` | test | 3 | 3000 | `configs/eval/warehouse_018.yaml` |
| `train/Warehouse_019` | test | 3 | 3000 | `configs/eval/warehouse_019.yaml` |
| `val/Warehouse_022` | test | 4 | 3090 | `configs/eval/warehouse_022.yaml` |
| `train/Warehouse_012` | validation | 8 | 3000 | `configs/eval/warehouse_012.yaml` |
| `val/Warehouse_020` | validation: the site-mode settings were chosen here; 16 cameras, outside the 12-camera spec | 16 | 3000 | `configs/eval/warehouse_020.yaml` |
| `train/Warehouse_009` | development only (twin of training scene W008) | 12 | 3000 | `configs/eval/warehouse_009.yaml` |
| `train/Warehouse_017` | development only (twin of test scene W018) | 3 | 3000 | `configs/eval/warehouse_017.yaml` |

Test scenes were never trained or tuned on and are buildings distinct from the training scenes.
The detector-training scenes are not turned into streams: `scripts/prepare_site_dataset.py`
samples frames straight from the downloaded videos.

### Making the streams

Both scripts take the scene by its dataset path and write under the work directory
(`D:/ugen-work` unless `SAYGUARD_WORK` says otherwise). Downloading needs `huggingface_hub`;
converting needs `ffmpeg` and `ffprobe` on PATH.

```bash
python scripts/fetch_smartspaces.py train/Warehouse_010     # calibration, floor plan, ground truth, all videos
python scripts/make_site_streams.py train/Warehouse_010     # frame-locked 10 fps streams
```

1. `fetch_smartspaces.py` puts `calibration.json`, `map.png`, `ground_truth.json` and
   `videos/Camera_XXXX.mp4` into `@work/datasets/smartspaces/<split>/<scene>/`. Depth maps are
   skipped; downloads resume and retry. `--no-videos` fetches only calibration, ground truth and
   map, `--no-gt` skips the ground truth, `--cameras 0 1 2` fetches only those cameras.
2. `make_site_streams.py` writes `@work/videos/site/<scene>/Camera_XXXX.mp4` (1280×720, 10 fps,
   H.264, no audio) and `streams.json`. Options: `--cameras`, `--step` (default 3: 30 → 10 fps),
   `--width` (default 1280), `--jobs` (parallel ffmpeg processes, default 4), `--threads`
   (per process), `--force` (re-encode outputs that already verify).

How `make_site_streams.py` makes the streams, and why:

- **Frame lock.** Site mode fuses every camera on one floor plan, and the ground truth is indexed
  by source frame, so output frame k must be source frame 3k in every camera. Frames are picked
  by index (`select` on the frame number, timestamps rewritten afterwards), not with `fps=`,
  which picks by timestamp and can slip by one frame. A trailing partial group is dropped so
  every camera ends on the same frame.
- **Checks.** A source that is still being copied (size changing) or was cut short (fewer
  packets than the container index lists) is skipped. Each encode must show that the decoder
  delivered every source frame with no decode errors (ffmpeg 6.1 or newer reports this; with an
  older ffmpeg the script warns that alignment is unchecked), and the output must have exactly
  ⌊source frames / step⌋ frames at source fps / step and the requested width. Output goes to a
  `.part.mp4` first, so an interrupted run never leaves a finished-looking stream.
- **Constant rate.** File sources are paced by the frame rate OpenCV reports, so every stream is
  encoded at an exact constant rate; cameras whose average rates differ are reported, because one
  of them would play faster and drift out of the lock.
- **`streams.json`** records the scene, source fps, step, fps, frame size, frame count (the
  shortest camera) and camera list. `scripts/eval_site.py` reads step and fps from it to line
  every stream frame up with its ground-truth frame.
- Re-runs verify existing outputs and skip them; a stream whose source is no longer on disk is
  kept only if the last `streams.json` lists it for the same settings.

In the live demo the server plays the files frame-locked (`SyncedFileGroup` in
`sayguard/sources.py`): one reader takes frame k of every file and publishes them together, and
all files rewind together at the end of the shortest one.

An evaluation scene is prepared with the same two commands and its own path, e.g.
`python scripts/fetch_smartspaces.py val/Warehouse_022` and
`python scripts/make_site_streams.py val/Warehouse_022`. Each `configs/eval/warehouse_0XX.yaml`
repeats its commands in its header; the evaluation itself is described in
[SITE_MODE.md](SITE_MODE.md).
