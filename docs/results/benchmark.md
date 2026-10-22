# SayGuard throughput benchmark

Generated 2026-10-18T13:14:21 by `scripts/benchmark.py`. **Measured** = run on the PC prototype in this repo. **Reference** = Hailo's published Model Zoo figures (not measured by us). **Estimate** = derived from the reference numbers.

## 1. Detector on the PC prototype (measured)

Device: NVIDIA GeForce RTX 5060 Ti, FP16, `helmet_yolov8s.pt`, 640x640 input, 10 real frames from D:\ugen-work\videos\raw; 10 warm-up + 50 timed iterations per batch size; wall clock around model.predict() + boxes to CPU, CUDA-synchronised. Peak GPU memory 243.8 MB.

| Batch | ms / batch (mean) | p50 | p95 | ms / image | images / s | pre / infer / post (ms per image) |
|---|---|---|---|---|---|---|
| 1 | 7.45 | 6.76 | 11.17 | 7.45 | 134.2 | 0.85 / 5.20 / 1.14 |
| 4 | 14.84 | 14.34 | 16.86 | 3.71 | 269.6 | 0.99 / 1.80 / 0.83 |
| 8 | 25.71 | 25.52 | 27.46 | 3.21 | 311.2 | 0.97 / 1.66 / 0.53 |
| 10 | 32.34 | 31.99 | 35.82 | 3.23 | 309.2 | 0.97 / 1.67 / 0.55 |

Cameras the PC could serve at its best measured throughput (measured):

| FPS per camera | Max cameras | At batch | images / s |
|---|---|---|---|
| 5 | 62 | 8 | 311.2 |
| 10 | 31 | 8 | 311.2 |
| 15 | 20 | 8 | 311.2 |

## 2. One UGen300 (Hailo-10H): detector capacity (estimate)

Reference: yolov8s on Hailo-10H = **166 FPS at batch 1, 252 FPS at batch 8** (Hailo Model Zoo, Hailo-10H object detection, measured over PCIe Gen3 x4). SayGuard's detector is the same YOLOv8s at 640x640 with a 2-class head, so these figures are used as the estimate until Stage II measures the compiled HEF. Host-side decoding, NMS and tracking are not included.

| Cameras | FPS / camera | Demand FPS | Load @ batch 8 | Load @ batch 1 | Headroom @ batch 8 | Max cameras @ batch 8 | Max cameras @ batch 1 |
|---|---|---|---|---|---|---|---|
| 10 | 5 | 50 | 19.8% | 30.1% | 202.0 FPS (80.2%) | 50 | 33 |
| 10 | 10 | 100 | 39.7% | 60.2% | 152.0 FPS (60.3%) | 25 | 16 |
| 10 | 15 | 150 | 59.5% | 90.4% | 102.0 FPS (40.5%) | 16 | 11 |

With spoken object rules on 2 cameras, YOLO-World runs every 3 frames there (6.67 FPS at the reference 46.6 FPS): detector 39.7% + YOLO-World 14.3% = **54.0% of device time** (estimate; assumes ideal time-slicing, no model-switch overhead, and no GenAI model running at the same moment).

## 3. Hailo-10H GenAI and vision models used by SayGuard (reference)

Source: Hailo Model Zoo (GenAI / vision), Hailo-10H. Not measured by us; the PC prototype runs the same models with PyTorch / Ollama.

| Model | Used for | Throughput | Time to first token |
|---|---|---|---|
| Whisper-Small | speech-to-text for spoken rules | 10.61 tokens/s | - |
| Qwen2.5-1.5B-Instruct | rule parsing, daily safety report | 7.35 tokens/s | 0.37 s |
| Qwen2-VL-2B-Instruct | optional VLM backend for the double-check (not the default) | 7.04 tokens/s | 1.02 s |
| CLIP ViT-B/32 image encoder | alert double-check (shipped; 512->2 yes/no head on the host) | 81.5 FPS | - |
| yolo_world_v2s | open-vocabulary objects (forklift, ...) | 46.6 FPS | - |

Rough latency per task on the UGen300 (estimate = TTFT + output tokens / tokens per s; prompt/image encoding beyond the published TTFT is not modelled):

| Model | Task | Output tokens (assumed) | Est. seconds |
|---|---|---|---|
| Whisper-Small | transcribe a ~5 s spoken rule | 25 | 2.4 |
| Qwen2.5-1.5B-Instruct | spoken rule -> rule JSON | 80 | 11.3 |
| Qwen2-VL-2B-Instruct | optional VLM backend: verify one alert (short yes/no + reason) | 16 | 3.3 |
| Qwen2.5-1.5B-Instruct | daily safety report | 350 | 48.0 |

Shipped double-check: one CLIP encoder pass per crop, about 12 ms at the reference 81.5 FPS (estimate until Stage II compiles CLIP to a HEF and measures it).
