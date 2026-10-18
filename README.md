# SayGuard

**One ASUS UGen300 · ten cameras · safety rules you can say.**

SayGuard is our ASUS UGen AI League entry (Battlefield Lightning · Workplace AI). One
UGen300 (Hailo-10H) watches every camera on a site for hard-hat compliance, and the safety
officer adds new rules by speaking them — "Warn me if anyone is within 3 m of a forklift
in the warehouse" — without collecting data or retraining a model.

This repository is the **Stage I prototype**: the full pipeline runs on a PC GPU with the
same networks that the Hailo Model Zoo provides for Hailo-10H, so Stage II re-targets it
onto the UGen300 (see [docs/HAILO_PORT.md](docs/HAILO_PORT.md)).

## What it does

- **Ten streams, one detector.** Frames from all cameras are batched into one YOLOv8s
  hard-hat detector (fine-tuned on 19,745 CC BY 4.0 images), tracked per camera with
  ByteTrack, and checked against the active rules ten times a second per camera.
- **Rules you can say.** Whisper-Small transcribes Mandarin or English; Qwen2.5-1.5B
  turns the sentence into a rule (JSON schema); a deterministic checker overrides any
  number, distance, time or object the sentence states explicitly; the officer confirms
  before it goes live.
- **No retraining for new hazards.** Objects named in a rule ("forklift", "ladder") are
  found by YOLO-World from the word itself; distances are estimated from the worker's
  head size.
- **Fewer false alarms.** An alert needs the condition to persist (default 2 s) across a
  track's recent frames; Qwen2-VL-2B then double-checks the crop and filters false alarms.
- **Daily safety report** written on-device by the LLM from the alert log.
- **Privacy.** Everything stays on site; faces are pixelated in every tile and snapshot.

## Run the prototype (Windows 11, NVIDIA GPU)

```bash
# heavy artifacts live outside the repo; override with SAYGUARD_WORK
python -m venv venv && venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/prepare_dataset.py            # Hard Hats v2 -> YOLO format
python scripts/train_detector.py --epochs 40 # -> models/helmet_yolov8s.pt (+ .onnx for Hailo DFC)
python scripts/eval_detector.py              # test-split metrics -> docs/results/
python scripts/fetch_videos.py all           # free Mixkit clips (not redistributed)
python scripts/score_clips.py && python scripts/make_streams.py   # pick + transcode 10 demo streams
ollama serve & ollama pull qwen2.5:1.5b      # or hailo-ollama on the UGen300
python -m sayguard serve                     # http://127.0.0.1:8000
```

Model folders expected under the work dir: `models/hf/whisper-small`,
`models/hf/Qwen2-VL-2B-Instruct`, `models/yolov8s-worldv2.pt`.

## Layout

```
sayguard/            pipeline, rule engine + parser, detectors (Ultralytics / HailoRT), tracking,
                     privacy, alert store, VLM verifier, ASR, report, FastAPI server
web/                 dashboard (vanilla JS, works offline)
scripts/             dataset prep, training, evaluation, clip selection, benchmark
configs/site.yaml    cameras, zones, models, default rules
docs/                Hailo port plan, results, licences, demo script, pitch outline
tests/               rule engine / parser / report unit tests
```

## Results (Stage I, PC prototype)

![SayGuard dashboard](docs/img/dashboard_en.png)

| | |
|---|---|
| Hard-hat detector, held-out test split | mAP50 **0.939**, precision 0.913, recall 0.899 |
| Ten cameras at 10 fps | 105 frames/s through one batched detector; ≈40 % of one UGen300 (estimate) |
| Spoken rules, 20 held-out sentences | LLM alone 40 % → **LLM + checker 95 %** exactly right, 0.34 s |
| VLM second opinion, held-out crops | filters 89 % of wrong flags, confirms 94 % of real violations |

Details and raw numbers: [docs/results/](docs/results/README.md). Reproduce with
`scripts/eval_detector.py`, `scripts/benchmark.py`, `scripts/eval_rules.py`, `scripts/eval_vlm.py`.

## Licences

SayGuard is released under the [GNU AGPL-3.0](LICENSE) (it builds on Ultralytics, which is
AGPL-3.0). Data, models and dependencies are listed in [docs/LICENSES.md](docs/LICENSES.md).
Demo footage: [docs/VIDEO_SOURCES.md](docs/VIDEO_SOURCES.md).
