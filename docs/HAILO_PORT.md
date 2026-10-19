English | [繁體中文](HAILO_PORT.zh-TW.md)

# Porting SayGuard to the UGen300

**Every model in SayGuard was picked because Hailo already runs it on the Hailo-10H. Stage II is
a port, not a rebuild.**

The PC prototype and the UGen300 version share one design. A host (a mini PC) decodes the camera
streams, tracks workers, builds the floor plan, runs the rule engine and serves the dashboard.
The UGen300 runs the neural networks. Stage I ran everything on a PC GPU and nothing has run on a
UGen300 yet, but the hard-hat detector and the double-check's image encoder have already been
through Hailo's own toolchain, with their INT8 accuracy checked.

## Why these models

| Job in SayGuard | Model | Hailo-10H reference |
|---|---|---|
| Find people, hard hats and vehicles on every camera | YOLOv8s, fine-tuned (hard-hat: 2 classes; warehouse: 6 classes) | 166 FPS at batch 1, 252 FPS at batch 8 |
| Double-check every hard-hat alert | CLIP ViT-B/32 image encoder + a small yes/no head | 81.5 FPS (batch 1) |
| Find objects named in a spoken rule (excavator, forklift…) | YOLO-World v2-S | 46.6 FPS |
| Speech to text | Whisper-Small | 10.6 tokens/s |
| Spoken sentence → rule; daily report plan | Qwen2.5-1.5B-Instruct | 7.35 tokens/s, first token after 0.37 s |

Reference figures: Hailo Model Zoo and Hailo GenAI Model Zoo.

- **YOLOv8s: one detector for every camera.** Frames from all streams go through one batched
  network. Our fine-tuning only changes the number of classes, so the weights reuse Hailo's
  own YOLOv8s build recipe.
- **CLIP for the double-check, not a chat model.** The check only has to answer yes or no, so a
  frozen image encoder plus one small matrix multiply on the host is enough: 15 ms per alert on
  the PC.
- **Qwen2.5-1.5B through the Ollama API.** On the UGen300, hailo-ollama serves the same API, so
  moving the LLM is a change of one setting (`llm.base_url`). A small model is enough because code
  checks every rule it drafts and does all the counting in the report.
- **Whisper-Small** is available ready-made in the Hailo GenAI Model Zoo.

## Already done on the Hailo toolchain

We do not have the device yet, but the two vision models of hard-hat mode have gone through
Hailo's Dataflow Compiler 5.4.0 with the Hailo Model Zoo recipes. We measured what INT8 costs in
Hailo's emulator, which reproduces the chip's integer arithmetic bit for bit. The scripts are in
`hailo/`.

- **Hard-hat detector compiled to a HEF**, the file format the Hailo-10H runs. The fine-tuned
  YOLOv8s was exported to ONNX, quantized to INT8 with 512 calibration images and compiled. Box
  decoding and NMS ship inside the HEF, as in Hailo's own YOLOv8s build, and the config already
  points to the file.
- **INT8 costs almost nothing.** On the same 2,001 test images, the float Hailo graph scores 0.893
  mAP50 and the INT8 version 0.889. The float graph matches the PyTorch model, so the conversion
  itself is exact.
- **CLIP encoder quantized.** We used the Model Zoo's `clip_vit_b_32_image_encoder`. With the
  yes/no head retrained on INT8 features, the check stays at 95.3 % accuracy, the same as on the
  PC.
- **Host code written.** A HailoRT backend for the detectors (`sayguard/detectors/hailo_backend.py`)
  is written against HailoRT 5.x; switching is `detector.backend: hailo` in the config. The rule
  parser and the report already work with any Ollama-compatible server.

| Emulator check (Hailo-10H) | Before | INT8 |
|---|---|---|
| Hard-hat detector, mAP50 on 2,001 test images | 0.893 (float) | **0.889** |
| Hard-hat detector, mAP50-95 | 0.566 (float) | 0.560 |
| CLIP image features vs PyTorch (cosine similarity) | 0.9999 (float) | 0.9634 |
| Double-check accuracy (head retrained on INT8 features) | 95.3 % (PC) | **95.3 %** |
| Real violations kept by the double-check | 94.9 % (PC) | 93.9 % |

Both detector figures use the NMS settings built into the HEF, so compare 0.889 with 0.893, not
with the 0.939 in the README (a different evaluation).

## Load on one UGen300

- **Hard-hat mode, 10 cameras.** Ten cameras at 10 frames/s each make 100 frames/s. On the PC, one
  batched YOLOv8s analyses every one of them live. Against Hailo's reference of 252 FPS at batch
  8, that is about **40 % of one UGen300** (estimate). The rest is left for YOLO-World on cameras
  with object rules and for the double-check, which runs once per alert, not per frame. The
  dashboard shows this estimate live ("Est. UGen300 load").
- **Language models run on demand.** Whisper and Qwen2.5 only run when someone speaks a rule or a
  report is due, so they add no steady load. From the reference speeds we estimate about 2.4 s to
  transcribe a 5-second spoken rule, about 11.3 s to turn it into a rule, and about 48 s for a
  daily report written entirely by the LLM. A rule takes 0.2–0.5 s on the PC, so the device is
  slower, but an officer adds a few rules a day and confirms each one anyway.

## Stage II port plan (Nov 6 – Dec 4)

1. **Detectors on the chip (week 1).** Install HailoRT and the UGen300 driver on the host, run the
   hard-hat HEF through the HailoRT backend and measure throughput over USB. Build the warehouse
   detector's HEF with the same flow (6 classes, 960 px input, calibration frames from training
   scenes only), check its INT8 accuracy in the emulator, and measure whether 12 cameras fit on
   one box.
2. **Double-check on the chip.** Compile the quantized CLIP encoder to a HEF and retrain the
   yes/no head on the chip's own features, as already done in the emulator.
3. **Speech and language on the chip.** Switch to the Hailo GenAI Model Zoo builds of
   Whisper-Small and Qwen2.5-1.5B, serve Qwen2.5 through hailo-ollama and point `llm.base_url` at
   it. Re-run the same spoken-rule and report tests on the device.
4. **End-to-end test at the ASUS test space.** One host and one UGen300 run the whole pipeline; we
   measure frame rate per camera, alert latency, device load and power.

## Risks and how we handle them

| Risk | What we do |
|---|---|
| **The HailoRT backend has not run on hardware.** It is written against the HailoRT 5.x API. | First task of week 1. The HEF's accuracy is already known from the emulator, so only the plumbing is left to check: output format, image scaling, batching. |
| **USB may be slower than Hailo's PCIe reference figures.** | Hard-hat mode needs about 40 % of the reference capacity, so there is headroom. If it is tight, we lower the frame rate per camera or split the cameras across two boxes. |
| **The warehouse detector is heavier and has no HEF yet.** Its 960 px input is more work per frame than the 640 px reference, and the 12-camera limit comes from coverage in the test scenes, not from a compute measurement. | Build and measure it in week 1. If 12 cameras at 960 px do not fit, each box covers fewer cameras or the input gets smaller. |
| **The Hailo build of Qwen2.5-1.5B is quantized differently**, so the PC accuracy (95 % and 84–88 %) may not carry over. Rule parsing also needs hailo-ollama to honour the JSON-schema `format` field. | The checker corrects numbers, times and objects whatever the LLM says. If a reply is not valid JSON, the rule parser and the report fall back to their code-only paths. We re-run the same test sentences on the device. |
| **INT8 shifts the CLIP features** (cosine 0.9634). | Retrain the yes/no head on the chip's features: 95.3 % accuracy in the emulator. |
| **Vision and language models share one chip.** While Qwen2.5 drafts a rule or a report, the detectors get less time. | Language models run for seconds at a time and the detectors have headroom. Stage II measures the camera frame rate while a rule is being drafted. |

Every number and how it was measured: [docs/results](results/README.md). To rebuild the Hailo
files yourself, see [SETUP.md](SETUP.md#rebuild-the-hailo-files).
