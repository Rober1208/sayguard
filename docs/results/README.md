# Stage I results

Everything here was measured on the PC prototype (Windows 11, i5-14400, RTX 5060 Ti 16 GB)
on 2026-10-18 unless marked **reference** (Hailo's published figures) or **estimate**
(derived from them). Raw files sit next to this page.

## Hard-hat detector — held-out test split (`detector_metrics.json`)

YOLOv8s fine-tuned on Roboflow Hard Hats v2 (CC BY 4.0), 39 epochs, 1.25 h. Test split:
2,001 images the model never saw.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All | 0.913 | 0.899 | **0.939** | 0.595 |
| Hard hat | 0.919 | 0.896 | 0.945 | 0.546 |
| Bare head (violation) | 0.907 | 0.901 | 0.934 | 0.643 |

11.1 M parameters, 28.6 GFLOPs, 22.5 MB; ONNX (opset 11) exported for the Hailo DFC.

## Ten cameras through one detector (live run, `live_state` in the log)

| Metric | Value |
|---|---|
| Cameras × processed fps | 10 × 10.0 |
| Frames analysed per second | 105 |
| Detector batch | 10 frames, 31–40 ms (PC) |
| Estimated UGen300 load | 100 / 252 FPS = **39.7 %** (estimate, YOLOv8s batch 8 reference) |
| Full stack on the GPU (detector + YOLO-World + Whisper + Qwen2.5 + Qwen2-VL) | 7.6 GB VRAM, ~11 % utilisation, 45 W |
| YOLO-World (spoken object rules) | 12 ms per frame |

Capacity table and PC batch sweep: `benchmark.md`.

## Spoken rules (`rule_parsing.json`, `scripts/eval_rules.py`)

A rule counts as correct only if every stated field is right: type, object, count or
distance, time window, duration (when stated) and the set of cameras it applies to.

| Parser | Dev set (32 sentences, used to build the extractor) | **Held-out (20 sentences, never used for tuning)** |
|---|---|---|
| LLM alone (Qwen2.5-1.5B) | 25 % | **40 %** |
| Rule extractor alone | 100 % | 90 % |
| **LLM + checker (shipped)** | 97 % | **95 %** |

Parse latency with the LLM: 0.34 s mean. Speech-to-text (Whisper-Small, GPU): 0.4–0.8 s
per rule after warm-up; site words ("A區", "挖土機") are passed as a decoder prompt —
without it "A區" was transcribed as "H".

## VLM second opinion (`vlm_check.json`, `scripts/eval_vlm.py`)

Qwen2-VL-2B, production prompt, crops cut exactly like an alert crop from held-out test images.

| Crops | VLM answer | Meaning |
|---|---|---|
| 94 people wearing a hard hat | 89.4 % "wearing" | a wrong "no hard hat" flag would be filtered |
| 99 bare heads | 93.9 % "not wearing" | a real violation is confirmed |
| All | 91.7 % accuracy, 0.16 s mean (PC) | |

Because ~6 % of real violations would be judged "wearing", filtered alerts are kept in the
feed (dimmed) rather than deleted.

## Daily report

Default "hybrid" mode: every number, table and the summary are computed from the alert log,
and the on-device LLM drafts only the three recommended actions. In testing, Qwen2.5-1.5B
writing the whole report passed a numbers-in-stats check but still misattributed counts and
invented terms, so full-LLM mode is opt-in (`build_report(..., mode="llm")`).

## Not measured yet (Stage II)

Anything on the UGen300 itself: HEF throughput over USB, GenAI latency on hailo-ollama,
device power, end-to-end alert latency on the device.
