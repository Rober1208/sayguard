English | [繁體中文](LICENSES.zh-TW.md)

# Third-party data, models and code

The competition rules require entrants to hold the rights to every image, video, model
and dataset they use. Everything below is used under its published license; nothing
restricted is redistributed in this repository.

SayGuard itself: Copyright (C) 2026 逼寶隊, released under the GNU AGPL-3.0 ([LICENSE](../LICENSE)).

## Data

| Item | Source | License | Use |
|---|---|---|---|
| Hard Hats v2 (19,745 images, classes `hardhat` / `no-hardhat`) | Roboflow Universe "hard-hats-fhbh5", mirrored as `keremberke/hard-hat-detection` on Hugging Face | CC BY 4.0 (attribution) | Training/validation/test of the hard-hat detector (classes renamed `helmet` / `head` by `scripts/prepare_dataset.py`); crops for the alert yes/no check; calibration images for the Hailo INT8 builds (never the test split) |
| Demo camera footage, hard-hat mode | Mixkit (see `docs/VIDEO_SOURCES.md`) | Mixkit Stock Video Free License | Simulated CCTV streams; clips are **not** committed: they are downloaded by hand from the listed pages (`scripts/fetch_videos.py list` / `check`) |
| NVIDIA PhysicalAI-SmartSpaces, MTMC_Tracking_2026 (synthetic warehouses: synchronized calibrated videos, calibration, floor plans, 2D/3D ground truth) | [huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) | CC BY 4.0 (attribution) | Warehouse mode: demo footage, evaluation, and training data for the site detector; every scene used and its role are listed below. Not committed; `scripts/fetch_smartspaces.py` downloads single scenes |

Faces are pixelated locally, on site (today the PC prototype; the UGen300 in Stage II), in every
dashboard tile and saved snapshot (`sayguard/privacy.py`): from each detected head in hard-hat mode,
and from the top of each detected person box in warehouse mode (taller for a box cut off at the
bottom or a crouching worker). A face the detector misses is not pixelated; how often that
happens is not measured.

### Hard Hats v2

Credit: "Hard Hats v2, Roboflow Universe (https://universe.roboflow.com/roboflow-universe-projects/hard-hats-fhbh5),
CC BY 4.0", used through its Hugging Face mirror https://huggingface.co/datasets/keremberke/hard-hat-detection.
Changes: classes renamed `helmet` / `head` and labels converted from COCO to YOLO format
(`scripts/prepare_dataset.py`).

### NVIDIA PhysicalAI-SmartSpaces: scenes used

Credit: "NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2026), CC BY 4.0". Scenes are named by
their dataset path: `train/` and `val/` are NVIDIA's folders, not SayGuard's roles. SayGuard's
own role for each scene is in the second column (the evaluation protocol is in
[SITE_MODE.md](SITE_MODE.md), the numbers in [FACTS.md](FACTS.md)).

| Scene | SayGuard role | What was used |
|---|---|---|
| `train/Warehouse_010` | Live demo (`configs/warehouse.yaml`) and **test** scene. Not a blind test: the tracker was developed while watching it and its zones are the demo config's | 11 camera videos, calibration, floor plan; ground truth only to score |
| `train/Warehouse_013`, `train/Warehouse_015`, `train/Warehouse_018`, `train/Warehouse_019`, `val/Warehouse_022` | **Test**: the detector was never trained on them and the measured settings were not chosen on them; floor plans distinct from the training scenes. W018 is the twin of development scene W017, which was used to develop the alert logic | camera videos, calibration, floor plan; ground truth only to score |
| `train/Warehouse_012` | **Validation** | camera videos, calibration, floor plan; ground truth only to score |
| `val/Warehouse_020` | **Validation**: the measured site-mode settings (worker confidence filter, vehicle confidence, forklift distance pad, 40 px "in view" height) were chosen on this scene only. Same building as training scenes W000-W007, 16 cameras (outside the 12-camera spec) | camera videos, calibration, floor plan; ground truth only to score |
| `train/Warehouse_009`, `train/Warehouse_017` | **Development only**: W009 is the twin of training scene W008 (same building, same cameras), W017 the twin of test scene W018 | camera videos, calibration, floor plan; ground truth only to score |
| `train/Warehouse_000`, `_002`, `_004`, `_006`, `_008`, `_011`, `_014`, `_016` | **Site detector training**: 8,000 images from 8 scenes in 5 buildings (W000, 002, 004 and 006 are one building; W016: cameras 0-8 for training, its other cameras for the training-time validation split); never used for evaluation | frames sampled from the camera videos, labels from the ground-truth visible 2D boxes (`scripts/prepare_site_dataset.py`) |
| `train/Warehouse_001`, `_003`, `_005`, `_007`, `val/Warehouse_021` | Not used for training or evaluation | calibration, floor plan and ground truth only; no video |

Changes made (CC BY 4.0 asks for them to be indicated):

- Demo and evaluation streams: every 3rd source frame kept (30 → 10 fps), scaled to 1280 px
  wide, re-encoded as H.264 without audio (`scripts/make_site_streams.py`); in the dashboard,
  SayGuard overlays and face pixelation are added.
- Detector training: sampled frames saved as JPEG at 1280 px wide, with YOLO labels converted
  from the ground truth (`scripts/prepare_site_dataset.py`).
- The ground truth never enters the live pipeline; `scripts/eval_site.py` uses it only to score.

## Models

| Model | License |
|---|---|
| YOLOv8s (COCO pre-trained) and our fine-tuned weights (hard-hat detector, 2 classes; warehouse site detector, 6 classes), YOLO-World v2-S weights | AGPL-3.0 (Ultralytics) |
| CLIP ViT-B/32: text encoder (YOLO-World class embeddings) and image encoder (alert yes/no check); Hailo Model Zoo `clip_vit_b_32_image_encoder` for Hailo-10H | MIT (OpenAI) |
| Whisper-Small | MIT (OpenAI) |
| Qwen2.5-1.5B-Instruct | Apache-2.0 (Alibaba Qwen) |
| Qwen2-VL-2B-Instruct (optional VLM check) | Apache-2.0 (Alibaba Qwen) |

Our fine-tuned weights are not in this repository (they live in `@work/models/`). The site
detector weights are trained on SmartSpaces frames, so anyone sharing them should carry the
CC BY 4.0 credit line above.

## Code dependencies

Ultralytics (AGPL-3.0), FastAPI/Starlette (MIT), Uvicorn (BSD-3),
OpenCV (Apache-2.0), PyTorch (BSD-3), Transformers (Apache-2.0), Ollama (MIT),
opencc-python-reimplemented (Apache-2.0), huggingface_hub (Apache-2.0, `scripts/fetch_smartspaces.py`),
motmetrics (MIT, `scripts/eval_site.py`), pycocotools (BSD-2-Clause, `hailo/eval_int8.py`),
NumPy (BSD-3), PyYAML (MIT), Pydantic (MIT), HTTPX (BSD-3), psutil (BSD-3), Accelerate (Apache-2.0),
Pillow (MIT-CMU), python-multipart (Apache-2.0), ftfy (Apache-2.0), regex (Apache-2.0), the Ultralytics
CLIP fork (MIT), pytest (MIT), onnxruntime (MIT, `hailo/eval_same_nms.py`), edge-tts (LGPL-3.0, only
`scripts/eval_spoken_rules.py`), Hailo Model Zoo (MIT); Hailo Dataflow Compiler and HailoRT (Hailo license, downloaded from the
Hailo Developer Zone, not redistributed here). FFmpeg (LGPL or GPL depending on the build) is
run as a separate program by the stream and dataset scripts and is not redistributed.

Because the detector code path uses Ultralytics (AGPL-3.0), this repository is released
under the AGPL-3.0 as well (see `LICENSE`), which keeps the combination compliant. Note that the
competition's IP clause (winners grant ASUS/Intel/Hailo a licence to the entry) does not
change third-party licences.
