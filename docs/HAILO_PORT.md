English | [繁體中文](HAILO_PORT.zh-TW.md)

# Stage II: moving SayGuard onto the ASUS UGen300 (Hailo-10H)

Every job in the prototype maps to a Hailo-10H model that already exists in the Hailo Model Zoo
or the Hailo GenAI Model Zoo, so Stage II is a re-target, not a redesign. The host keeps the
glue: decoding camera streams, tracking, floor-plan fusion (warehouse mode), the rule
engine, the web UI.

**Where it stands.** Stage I runs on the PC GPU; nothing has run on a UGen300 yet. Only the
hard-hat detector is compiled to a HEF (not yet run on hardware). The CLIP image encoder was
quantized but not compiled. Both were checked in Hailo's bit-exact emulator
([below](#done-in-stage-i-with-the-hailo-toolchain-hailo)). **The
warehouse-mode detector has no HEF yet:** `configs/warehouse.yaml` already names
`@work/models/site_yolov8s.hef`, but that file has not been built
([details](#warehouse-mode-detector-no-hef-yet)). Only the detector has a HailoRT backend in
the code today (`sayguard/detectors/hailo_backend.py`); YOLO-World, Whisper and the CLIP
encoder still run through PyTorch/transformers on the PC.

| Function | PC prototype (Stage I) | UGen300 (Stage II) | Hailo reference (Hailo-10H) |
|---|---|---|---|
| Hard-hat detection, all cameras | YOLOv8s fine-tuned (2 classes), Ultralytics/CUDA | same weights → ONNX (opset 11) → INT8 HEF with the Hailo Dataflow Compiler (`hailo/build_helmet.py`), NMS by HailoRT on the host; `sayguard/detectors/hailo_backend.py`. **HEF built, not yet run on hardware** | yolov8s: 166 FPS (batch 1), 252 FPS (batch 8) |
| Warehouse detection, all cameras (warehouse mode) | YOLOv8s fine-tuned (6 classes, 960 px), Ultralytics/CUDA | the same flow with its own ONNX, input size, 6-class NMS config and calibration frames → `site_yolov8s.hef`; same backend (class order from `detector.labels`). **HEF not built yet** | the yolov8s figures are for a 640 px input; nothing measured at 960 px |
| Objects named in spoken rules | YOLO-World v2-S (Ultralytics) | `yolo_world_v2s` HEF + host-side text embeddings | 46.6 FPS |
| Speech → text | Whisper-Small (transformers) | Whisper-Small (Hailo GenAI Model Zoo) | 10.6 tokens/s |
| Spoken rule → rule JSON (Mandarin or English, zone names); daily report plan (picks listed issues and vetted fixes) | Qwen2.5-1.5B-Instruct via the Ollama API, a JSON-schema `format` on every call | Qwen2.5-1.5B-Instruct via **hailo-ollama** (same API: change `llm.base_url`) | 7.35 tokens/s, 0.37 s to first token |
| Alert double-check (yes/no check) | CLIP ViT-B/32 image encoder + 2×512 linear head | `clip_vit_b_32_image_encoder` HEF (quantized and checked in the emulator; not compiled yet); the head is one host-side matrix multiply | 81.5 FPS (batch 1) |
| Optional VLM check | Qwen2-VL-2B-Instruct (transformers) | Qwen2-VL-2B-Instruct via the HailoRT GenAI VLM API (hailo-ollama serves text LLMs only) | Hailo GenAI Model Zoo (see [results/benchmark.md](results/benchmark.md)) |

FPS/TPS figures: Hailo Model Zoo (public models, HAILO10H, compiled with DFC v5.4.0,
measured over PCIe Gen3 x4) and Hailo GenAI Model Zoo. The UGen300 USB version may be
somewhat lower than PCIe; Stage II measures it on the real device.

## Budget

**Hard-hat mode, 10 cameras.** Ten cameras at a 10 frames/s target each go through one
batched YOLOv8s (measured on the PC: about 105 frames/s, 31-40 ms per 10-frame batch). At the
nominal 100 frames/s that is an estimated **40 % of the batch-8 capacity (252 FPS)** (not
measured on the device), leaving room for YOLO-World on the cameras that have
object rules and for the alert double-check (one CLIP encoder pass per alert, 81.5 FPS at
batch 1). Language models only run on demand (a spoken rule, a report), never per frame.
Estimates for the UGen300 from the reference speeds ([results/benchmark.md](results/benchmark.md),
not measured): a spoken rule → rule JSON in about 11.3 s, Whisper transcription of a 5 s rule in
about 2.4 s. All LLM accuracy in this repository was measured with Ollama `qwen2.5:1.5b` on the
PC GPU; the Hailo GenAI build (a different quantization) is not measured.

The dashboard shows this estimate live ("Est. UGen300 load") from the measured per-camera
frame rate and the yolov8s reference in the config (`hailo_reference`).

**Warehouse mode.** The deployment spec is one UGen300 per area of ≤ 12 cameras, a coverage
spec from the test scenes, not a compute figure. On the PC,
11 cameras at 10 fps go through the 6-class site detector at 960 px in about 63-95 ms per
11-frame batch (RTX 5060 Ti). Whether one Hailo-10H sustains 12 cameras at 960 px is **not
measured** (planned for Stage II, week 1): the Model Zoo figures are for a 640 px input, a 960 px
input is more work per frame, and there is no HEF to measure yet. The "Est. UGen300 load" readout in warehouse mode
uses the same 640 px yolov8s reference (`configs/warehouse.yaml`), so it understates the load.

## Done in Stage I with the Hailo toolchain (`hailo/`)

No device yet, but both hard-hat-mode vision networks already went through the real Hailo
flow: the detector was compiled to a HEF, the CLIP encoder was quantized, and both were checked in
Hailo's bit-exact emulator; full results in
[results/README.md](results/README.md#hailo-10h-compiled-and-checked-bit-exactly-in-hailos-emulator-hailo).

1. Environment (WSL Ubuntu 22.04, CPU only: the TensorFlow the DFC pins has no kernels for
   RTX 50-series GPUs): `bash hailo/setup_dfc.sh` with the Dataflow Compiler 5.4.0 wheel from
   the Hailo Developer Zone; it also clones the Hailo Model Zoo.
2. Detector, the shipped variant (512 calibration images plus activation clipping):
   ```bash
   python hailo/build_helmet.py parse
   python hailo/build_helmet.py optimize --calib 512 --tag _c512clip \
     --extra "pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])"
   python hailo/build_helmet.py compile --tag _c512clip --performance --install
   ```
   The ONNX is cut at the six YOLOv8 head convolutions and gets the Model Zoo `yolov8s`
   model script with a 2-class NMS config; calibration is a fixed random sample (seed 0) of
   the validation split, never the test split. `--performance` is what
   `hailomz compile --performance` adds; `--install` copies the HEF to
   `@work/models/helmet_yolov8s.hef`.
3. Detector accuracy: `python hailo/eval_int8.py --har ~/hailo/build/helmet_c512clip_optimized.har`
   (float vs bit-exact INT8 emulation on the 2,001 test images; writes `hailo_int8.json`) and
   `python hailo/eval_pc_baseline.py` (SayGuard venv; PyTorch, same evaluator and NMS settings).
4. CLIP encoder: `python hailo/clip_probe_int8.py dump` (SayGuard venv), then in WSL
   `hailomz parse clip_vit_b_32_image_encoder --hw-arch hailo10h` and
   `python hailo/clip_probe_int8.py emulate --har clip_vit_b_32_image_encoder.har`
   (writes `clip_probe_int8.json`).

What the emulator measured (`docs/results/hailo_int8.json`, `docs/results/clip_probe_int8.json`):

| Check | Result |
|---|---|
| Hard-hat detector, 2,001 test images: float Hailo graph → INT8 | mAP50 0.8927 → 0.889 (drop 0.0037); mAP50-95 0.5662 → 0.56 (drop 0.0062) |
| CLIP image features vs PyTorch (cosine) | float Hailo graph 0.9999; INT8 0.9634 |
| Yes/no check on INT8 features, head trained on PC features | accuracy 94.3 %, real violations kept 92.9 % (on PC features: 95.3 %, 94.9 %) |
| Yes/no check on INT8 features, head retrained on INT8 features | accuracy 95.3 %; wrong flags filtered 96.8 %, real violations kept 93.9 % |

The detector mAP50 here (score ≥ 0.2 NMS, pycocotools) is a different evaluation from the 0.939
of the Ultralytics validation (score ≥ 0.001): compare 0.889 only with 0.893.

## Warehouse-mode detector: no HEF yet

`configs/warehouse.yaml` already carries `detector.hef: "@work/models/site_yolov8s.hef"` and
`detector.labels` (the six classes in output order; `hailo_backend.py` uses them and skips the
helmet/head merge for any other label set). The HEF itself has not been built.
`scripts/train_detector.py` already exports `@work/models/site_yolov8s.onnx` (opset 11), but
`hailo/build_helmet.py` is written for the hard-hat model: the ONNX path, the 640×640 input
and the 2-class NMS config are fixed in the script; only the calibration folder is a flag
(`--calib-dir`). Building `site_yolov8s.hef` needs a variant of it with the site ONNX, a
960×960 input, a 6-class NMS config and calibration frames from the training scenes (never
an evaluation scene), followed by the same kind of float-vs-INT8 check before any warehouse
number can be quoted for the chip.

## Stage II on the UGen300

1. Install HailoRT 5.x and the UGen300 USB/PCIe driver on the host (Windows 11 or Linux).
2. Hard-hat mode: in `configs/site.yaml` set `detector.backend: hailo` (`detector.hef` already
   points to `@work/models/helmet_yolov8s.hef`). Warehouse mode: the same in
   `configs/warehouse.yaml`, once `site_yolov8s.hef` exists.
3. Validate `sayguard/detectors/hailo_backend.py` (written against HailoRT 5.x, not yet run
   on hardware): NMS output format, letterbox scaling, batch-8 throughput.
4. Double-check on the chip: run the CLIP image-encoder HEF through HailoRT and retrain the
   2×512 head on the chip's own INT8 features (in the emulator this gives 95.3 % accuracy, with
   93.9 % of real violations kept).
5. Start hailo-ollama with Qwen2.5-1.5B and set `llm.base_url` to it. Check that it honours
   the JSON-schema `format` field: the rule parser (`sayguard/rules/parser.py`) and the
   report's action menu (`sayguard/report.py`) send one with every request. Whisper-Small
   from the Hailo GenAI Model Zoo.
6. Measure: per-camera fps, end-to-end alert latency, device load, power; the accuracy and
   latency of the Hailo GenAI Qwen2.5-1.5B on the rule-parsing and report tests; in warehouse mode
   (week 1), also whether 12 cameras at 960 px fit on one device.
