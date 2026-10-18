# Stage II: moving SayGuard onto the ASUS UGen300 (Hailo-10H)

Every network in the prototype was chosen because it already exists in the Hailo Model
Zoo for Hailo-10H, so Stage II is a re-target, not a redesign. The host keeps the
glue: decoding camera streams, tracking, the rule engine, the web UI.

| Function | PC prototype (Stage I) | UGen300 (Stage II) | Hailo reference (Hailo-10H) |
|---|---|---|---|
| Hard-hat detection, all cameras | YOLOv8s fine-tuned, Ultralytics/CUDA | same weights → ONNX (opset 11) → HEF with the Hailo Dataflow Compiler, 2 classes, NMS on chip; `sayguard/detectors/hailo_backend.py` | yolov8s: 166 FPS (batch 1), 252 FPS (batch 8) |
| Objects named in spoken rules | YOLO-World v2-S (Ultralytics) | `yolo_world_v2s` HEF + host-side text embeddings | 46.6 FPS |
| Speech → text | Whisper-Small (transformers) | Whisper-Small (Hailo GenAI zoo) | 10.61 tokens/s |
| Text → rule JSON, daily report | Qwen2.5-1.5B-Instruct via Ollama API | Qwen2.5-1.5B-Instruct via **hailo-ollama** (same API: change `llm.base_url`) | 7.35 tokens/s, TTFT 0.37 s |
| Alert double-check | Qwen2-VL-2B-Instruct (transformers) | Qwen2-VL-2B-Instruct (Hailo GenAI zoo) | 7.04 tokens/s, TTFT 1.02 s |

FPS/TPS figures: Hailo Model Zoo (public models, HAILO10H, compiled with DFC v5.4.0,
measured over PCIe Gen3 x4) and Hailo GenAI Model Zoo. The UGen300 USB version may be
somewhat lower than PCIe; Stage II measures it on the real device.

## Budget for 10 cameras

YOLOv8s at 10 processed frames/s per camera = 100 FPS demand → **~40 % of the batch-8
capacity (252 FPS)**, leaving room for YOLO-World on the cameras that have object rules
and for the occasional VLM check (≈1–1.5 s each, only on alerts). Language models only run
on demand (a spoken rule, a report), never per frame.

The dashboard shows this estimate live ("Est. UGen300 load") from the measured per-camera
frame rate.

## Steps

1. `python scripts/train_detector.py` already exports `helmet_yolov8s.onnx` (opset 11).
2. Hailo DFC (Linux x86, Hailo Developer Zone account): parse → optimize with ~500
   calibration images from `datasets/helmet/images/val` → compile for `hailo10h` using the
   Model Zoo `yolov8s` NMS config with `classes: 2`.
3. Install HailoRT 5.x + the UGen300 USB/PCIe driver on the host (Windows 11 or Linux).
4. In `configs/site.yaml`: `detector.backend: hailo`, `detector.hef: @work/models/helmet_yolov8s.hef`.
5. Start hailo-ollama with Qwen2.5-1.5B and set `llm.base_url` to it.
6. Validate `hailo_backend.py` (written against HailoRT 5.x, not yet run on hardware):
   decode format of the NMS output, letterbox scaling, batch 8 throughput.
7. Measure: per-camera fps, end-to-end alert latency, device load, power.
