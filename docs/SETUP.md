English | [繁體中文](SETUP.zh-TW.md)

# Setup, reproduction and repository layout

For developers who want to run the prototype or re-run the measurements. The product overview is in the [README](../README.md).

## Run the prototype (Windows 11, NVIDIA GPU)

Heavy artifacts (datasets, weights, videos, the alert database) live in a work directory
outside the repo: `SAYGUARD_WORK`, default `D:/ugen-work`. In the configs, paths that start
with `@work/` resolve there. You also need `ffmpeg` and `ffprobe` on `PATH`, and
[Ollama](https://ollama.com).

```powershell
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
ollama serve                                   # in its own terminal (Stage II: hailo-ollama on the UGen300)
ollama pull qwen2.5:1.5b
```

Model files expected under the work directory:

- `models/hf/whisper-small`
- `models/yolov8s-worldv2.pt`
- `models/clip/ViT-B-32.pt`, shared by YOLO-World and the yes/no check
- optional: `models/hf/Qwen2-VL-2B-Instruct`, for the VLM check

### Hard-hat mode

```powershell
python scripts/prepare_dataset.py              # Hard Hats v2 -> YOLO format
python scripts/train_detector.py --epochs 40   # -> @work/models/helmet_yolov8s.pt (+ .onnx for the Hailo DFC)
python scripts/train_probe.py                  # yes/no head -> @work/models/helmet_probe.npz
python scripts/fetch_videos.py list            # 10 Mixkit pages: download each clip by hand (VIDEO_SOURCES.md)
python scripts/fetch_videos.py check           # all saved? -> writes the camera selection
python scripts/make_streams.py                 # transcode the 10 looping demo streams
python -m sayguard serve                       # configs/site.yaml -> http://127.0.0.1:8000 (?lang=en or ?lang=zh)
```

`start_demo.cmd` starts Ollama and the hard-hat demo in one step. Its paths are set for the
authors' machine.

### Warehouse mode

```powershell
# 1. Demo scene: calibration, floor plan, ground truth (only evaluation needs it; --no-gt skips it), 11 videos
python scripts/fetch_smartspaces.py train/Warehouse_010
python scripts/make_site_streams.py train/Warehouse_010    # frame-locked 10 fps streams + streams.json

# 2. Site detector (weights are not distributed). Training scenes only:
#    prepare_site_dataset.py refuses Warehouse_010, _018 and _020.
python scripts/fetch_smartspaces.py train/Warehouse_000    # likewise 002, 004, 006, 008, 011, 014, 016
python scripts/prepare_site_dataset.py --name site_v3 --train train/Warehouse_000 train/Warehouse_002 train/Warehouse_004 train/Warehouse_006 train/Warehouse_008 train/Warehouse_011 train/Warehouse_014 "train/Warehouse_016:~9,10" --val "train/Warehouse_016:9,10"
python scripts/train_detector.py --data "@work/datasets/site_v3/data.yaml" --name site_v8s_v3 --out site_yolov8s.pt --imgsz 960 --batch 16 --epochs 20

# 3. Run
python -m sayguard serve --config configs/warehouse.yaml   # http://127.0.0.1:8000
```

`start_demo_warehouse.cmd` does the same for warehouse mode once the streams exist. Its paths
are also set for the authors' machine.

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
kernels for RTX 50-series GPUs. The compiler wheel (5.4.0) comes from the Hailo Developer Zone.
Results are written to `docs/results/hailo_int8.json` and `docs/results/clip_probe_int8.json`.
What the build does and what it measured: [HAILO_PORT.md](HAILO_PORT.md).

```bash
bash hailo/setup_dfc.sh                       # Dataflow Compiler + Hailo Model Zoo in a WSL venv

# Hard-hat detector -> HEF (512 calibration images from the validation split + activation clipping)
python hailo/build_helmet.py parse
python hailo/build_helmet.py optimize --calib 512 --tag _c512clip \
  --extra "pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])"
python hailo/build_helmet.py compile --tag _c512clip --performance --install   # -> @work/models/helmet_yolov8s.hef

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
                     report, verifier, ASR decode, synced sources) and the warehouse rule sentences
start_demo.cmd       Windows launcher for the hard-hat demo
start_demo_warehouse.cmd  Windows launcher for the warehouse demo
```
