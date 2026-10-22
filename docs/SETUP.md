English | [繁體中文](SETUP.zh-TW.md)

# Setup, reproduction and repository layout

For developers who want to run the prototype or re-run the measurements. The product overview is in the [README](../README.md).

## Run the prototype (Windows 11, NVIDIA GPU)

You need:

- Python 3.12 (64-bit), git, and `ffmpeg` / `ffprobe` on `PATH` (for example `winget install Gyan.FFmpeg`)
- an NVIDIA GPU with about 8 GB of memory (the prototype was measured on an RTX 5060 Ti 16 GB)
- [Ollama](https://ollama.com), which serves the local LLM

Heavy artifacts (datasets, weights, videos, the alert database) live in a work directory
outside the repo: `SAYGUARD_WORK`, default `D:/ugen-work`. In the configs, paths that start
with `@work/` resolve there. Point it somewhere else before running anything, for example:

```powershell
$env:SAYGUARD_WORK = "E:/sayguard-work"         # this terminal; setx SAYGUARD_WORK E:\sayguard-work keeps it
```

```powershell
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
ollama serve                                   # in its own terminal (Stage II: hailo-ollama on the UGen300)
ollama pull qwen2.5:1.5b
```

Models from their publishers, into the work directory:

```powershell
$W = $env:SAYGUARD_WORK
hf download openai/whisper-small --local-dir "$W/models/hf/whisper-small"
python -c "import clip, os; clip.load('ViT-B/32', device='cpu', download_root=os.path.join(r'$W', 'models', 'clip'))"   # -> models/clip/ViT-B-32.pt
python -c "from ultralytics import YOLO; YOLO('yolov8s-worldv2.pt')"; Move-Item yolov8s-worldv2.pt "$W/models/"
hf download Qwen/Qwen2-VL-2B-Instruct --local-dir "$W/models/hf/Qwen2-VL-2B-Instruct"   # optional: the VLM check
```

`models/clip/ViT-B-32.pt` is shared by YOLO-World and the yes/no check. The trained weights
(hard-hat detector, yes/no head, site detector) are not distributed; the steps below train them.

The unit tests need none of this (no GPU, no models, no work directory):

```powershell
python -m pytest -q tests
```

### Hard-hat mode

```powershell
# Hard Hats v2 (Roboflow Universe, CC BY 4.0), mirrored on Hugging Face
hf download keremberke/hard-hat-detection --repo-type dataset --local-dir "$W/datasets/raw/keremberke"
foreach ($s in "train", "valid", "test") { Expand-Archive "$W/datasets/raw/keremberke/data/$s.zip" "$W/datasets/raw/keremberke/ex/$s" }
python scripts/prepare_dataset.py              # Hard Hats v2 -> YOLO format
python scripts/train_detector.py --epochs 40   # -> @work/models/helmet_yolov8s.pt (+ .onnx for the Hailo DFC)
python scripts/train_probe.py                  # yes/no head -> @work/models/helmet_probe.npz
python scripts/fetch_videos.py list            # 10 Mixkit pages: download each clip by hand (VIDEO_SOURCES.md)
python scripts/fetch_videos.py check           # all saved? -> writes the camera selection
python scripts/make_streams.py                 # transcode the 10 looping demo streams
python -m sayguard serve                       # configs/site.yaml -> http://127.0.0.1:8000 (?lang=en or ?lang=zh)
```

`start_demo.cmd` starts Ollama and the hard-hat demo in one step. It uses `SAYGUARD_WORK` (default
`D:\ugen-work`) and falls back to `python` and `ollama` on `PATH`.

### Warehouse mode

```powershell
# 1. Demo scene: calibration, floor plan, ground truth (only evaluation needs it; --no-gt skips it), 11 videos
python scripts/fetch_smartspaces.py train/Warehouse_010
python scripts/make_site_streams.py train/Warehouse_010    # frame-locked 10 fps streams + streams.json

# 2. Site detector (weights are not distributed). Training scenes only:
#    prepare_site_dataset.py refuses every evaluation and development scene (009, 010, 012, 013, 015, 017-020, 022).
python scripts/fetch_smartspaces.py train/Warehouse_000    # likewise 002, 004, 006, 008, 011, 014, 016
python scripts/prepare_site_dataset.py --name site_v3 --train train/Warehouse_000 train/Warehouse_002 train/Warehouse_004 train/Warehouse_006 train/Warehouse_008 train/Warehouse_011 train/Warehouse_014 "train/Warehouse_016:~9,10" --val "train/Warehouse_016:9,10"
python scripts/train_detector.py --data "@work/datasets/site_v3/data.yaml" --name site_v8s_v3 --out site_yolov8s.pt --imgsz 960 --batch 16 --epochs 20

# 3. Run
python -m sayguard serve --config configs/warehouse.yaml   # http://127.0.0.1:8000
```

`start_demo_warehouse.cmd` does the same for warehouse mode once the streams exist.

The dataset has 8,000 training images from 8 SmartSpaces training scenes in 5 synthetic
buildings (Warehouse_000/002/004/006 are one building). The measured weights
(`@work/models/site_yolov8s.pt`) were trained in two stages. The first stage ran 30 epochs
from COCO YOLOv8s on Warehouse_000/004/008/014, with 016 for validation. The last command
above then continued from that checkpoint, passed as `--model`. Written as it is, the command
starts from COCO YOLOv8s instead.

The dashboard takes the camera list and the example rules under "Say a rule" from the
config (`cameras:` and `examples:` in `configs/warehouse.yaml`).

## Reproduce the numbers

```powershell
python scripts/eval_detector.py              # hard-hat detector, test split -> detector_metrics.json
python scripts/benchmark.py                  # throughput tables -> benchmark.md
python scripts/eval_rules.py                 # rule parsing, hard-hat mode (typed sentences) -> rule_parsing.json (needs Ollama)
python scripts/eval_rules.py --site          # rule parsing, warehouse mode (typed sentences) -> rule_parsing_site.json
python scripts/eval_spoken_rules.py          # the same held-out sentences, spoken (synthetic voices, site noise) -> rule_parsing_spoken.json (needs Ollama, edge-tts)
python scripts/eval_report.py                # report actions, 20 synthetic days -> report_actions.json (needs Ollama)
python scripts/train_probe.py                # yes/no head -> probe_train.json
python scripts/eval_vlm.py --per-class 100   # generative VLM on the same crops -> vlm_check.json
python scripts/eval_yesno.py                 # yes/no methods on 193 held-out crops -> yesno_check.json (earlier comparison table)

python scripts/eval_site.py --all-zones                                              # Warehouse_010 -> site_eval.json
python scripts/eval_site.py --config configs/eval/warehouse_018.yaml --all-zones     # -> site_eval_warehouse_018.json
```

Notes on these scripts:

- **Where results go.** Results are written to `docs/results/` and to `@work/results/`.
- **Warehouse scenes.** Each `configs/eval/warehouse_0XX.yaml` lists the three commands for
  its scene in its header: fetch, make streams, evaluate. Warehouse_020 and _022 are in the
  dataset's `val/` split; the others are in `train/`.
- **`eval_site.py` options.**
  - `--dets-cache dets.pkl` saves the detector output on the first run and replays it
    afterwards, so later runs need no GPU.
  - `--tag NAME` writes an experiment only to `@work/results/` and leaves the published
    files alone.
- **Spoken rules.** `eval_spoken_rules.py` also needs `pip install edge-tts` (not in
  `requirements.txt`; the voices come from Microsoft's online service), Ollama with
  `qwen2.5:1.5b` and the Whisper-Small download above. A GPU helps; the published run was on a
  Colab L4. `--limit N` runs only N sentences per set, as a quick check, but still overwrites
  the published `rule_parsing_spoken.json`.
- **Results kept as files.** Three published results are stored as they were recorded, not
  regenerated by these scripts:
  - `rule_parsing_site_v1_first_run.json`, the first run of warehouse sentence set 1. The
    parser was improved on those sentences afterwards, and set 1 now scores 100 % (LLM + checker).
  - The frozen warehouse run in `docs/results/frozen/`, made before the two alert-logic fixes.
  - The AI-judge scores of the report actions. Judging is not part of the scripts: the judged
    run is `report_actions_judged.json`, with the prompt, batches and verdicts in
    `report_actions_judging/`. Rerunning `eval_report.py` writes a new `report_actions.json`
    without scores.
- **Hailo builds and emulator checks** are in `hailo/`; see the next section.

## Rebuild the Hailo files

Runs in WSL Ubuntu 22.04 on the CPU: the TensorFlow version the Dataflow Compiler needs has no
kernels for RTX 50-series GPUs. The warehouse detector's steps ran on a Colab T4 instead, which that
TensorFlow supports. The compiler wheel (5.4.0) comes from the Hailo Developer Zone. Results are
written to `docs/results/`: `hailo_int8.json`, `pc_baseline.json`, `clip_probe_int8.json`,
`site_same_nms.json`, `site_r544_same_nms.json` and `helmet_onnx_same_nms.json`. The compile
figures of the two warehouse HEFs are in `site_hef.json`.
What the build does and what it measured: [HAILO_PORT.md](HAILO_PORT.md).

```bash
bash hailo/setup_dfc.sh                       # Dataflow Compiler + Hailo Model Zoo in a WSL venv

# Hard-hat detector -> HEF (512 calibration images from the validation split + activation clipping)
python hailo/build_helmet.py parse
python hailo/build_helmet.py optimize --calib 512 --tag _c512clip \
  --extra "pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])"
python hailo/build_helmet.py compile --tag _c512clip --performance --install   # -> @work/models/helmet_yolov8s.hef

# Warehouse detector -> HEF (512 calibration frames from its training scenes only). The 960x960
# compile takes about two hours; copy site_perf.hef to @work/models/site_yolov8s.hef
python hailo/build_site.py parse
python hailo/build_site.py optimize --calib 512
python hailo/build_site.py compile --performance
# The cameras' 16:9 shape: export the same weights with Ultralytics at imgsz=(544, 960), then
python hailo/build_site.py parse --onnx site_yolov8s_r544.onnx --tag _r544
python hailo/build_site.py optimize --tag _r544 --calib 512
python hailo/build_site.py compile --tag _r544 --performance

# Float vs INT8 with one evaluator (same letterbox, the HEF's NMS settings, pycocotools)
python hailo/eval_same_nms.py site --har ~/hailo/build/site_optimized.har
python hailo/eval_same_nms.py site --tag _r544 --har ~/hailo/build/site_r544_optimized.har
python hailo/eval_same_nms.py helmet --onnx-only

# Float vs INT8 accuracy on the 2,001 test images; PyTorch baseline (SayGuard venv)
python hailo/eval_int8.py --har ~/hailo/build/helmet_c512clip_optimized.har
python hailo/eval_pc_baseline.py

# CLIP encoder: crops and PyTorch features (SayGuard venv), then quantize and emulate (WSL)
python hailo/clip_probe_int8.py dump
hailomz parse clip_vit_b_32_image_encoder --hw-arch hailo10h
python hailo/clip_probe_int8.py emulate --har clip_vit_b_32_image_encoder.har
```

## Layout

```
sayguard/            pipeline, rule engine + parser, detectors (Ultralytics / HailoRT), tracking,
                     privacy, alert store, alert verifier (CLIP probe / VLM), ASR, report, server
  site.py            warehouse mode: camera calibration, floor plan, zones
  sitetrack.py       warehouse mode: cross-camera fusion of workers, vehicles from ground contacts
  site_pipeline.py   warehouse mode: the live loop (shared with scripts/eval_site.py)
  site_view.py       warehouse mode: camera tiles and the live floor plan
web/                 dashboard (vanilla JS, works offline)
scripts/             dataset prep, training, evaluation, clip selection, SmartSpaces download and streams, benchmark
hailo/               Hailo Dataflow Compiler flow: ONNX -> INT8 -> HEF, emulator accuracy checks
configs/site.yaml    hard-hat mode: ten cameras, zones, models, default rules
configs/warehouse.yaml  warehouse mode: 11 calibrated cameras of Warehouse_010, floor zones, example rules
configs/eval/        one config per evaluation scene except Warehouse_010 (zones drawn on each scene's camera-coverage map before any result was seen)
docs/                FACTS.md (every published number), SITE_MODE.md (warehouse mode end to end),
                     results/, Hailo port plan, licences, video sources, demo script, pitch outline
tests/               unit tests (rules, parser, site rules, cross-camera tracking, tracker, events,
                     report, verifier, ASR decode, synced sources, privacy, pipeline metrics, long
                     runs, API guard) and the warehouse rule sentences
start_demo.cmd       Windows launcher for the hard-hat demo
start_demo_warehouse.cmd  Windows launcher for the warehouse demo
```
