English | [繁體中文](HAILO_PORT.zh-TW.md)

# Porting SayGuard to the UGen300

**Every model in SayGuard was picked because Hailo already runs it on the Hailo-10H. Stage II is
a port, not a rebuild.**

The PC prototype and the UGen300 version share one design. A host (a mini PC) decodes the camera
streams, tracks workers, builds the floor plan, runs the rule engine and serves the dashboard.
The UGen300 runs the neural networks. Stage I ran everything on a PC GPU and nothing has run on a
UGen300 yet, but both detectors are already compiled to Hailo-10H HEFs and the double-check's
image encoder is quantized (its HEF follows in Stage II), with all three checked for INT8 accuracy
in Hailo's emulator.

## Why these models

| Job in SayGuard | Model | Hailo-10H reference |
|---|---|---|
| Find people, hard hats and vehicles on every camera | YOLOv8s, fine-tuned (hard-hat: 2 classes; warehouse: 6 classes) | 166 FPS at batch 1, 252 FPS at batch 8, at 640×640 (no reference for the warehouse HEFs, which have larger inputs and several contexts; measured in Stage II) |
| Double-check every hard-hat alert | CLIP ViT-B/32 image encoder + a small yes/no head | 81.5 FPS (batch 1) |
| Find objects named in a spoken rule (excavator, forklift…) | YOLO-World v2-S | 46.6 FPS |
| Speech to text | Whisper-Small | 10.6 tokens/s |
| Spoken sentence → rule; daily report plan | Qwen2.5-1.5B-Instruct | 7.35 tokens/s, first token after 0.37 s |

Reference figures: Hailo Model Zoo and Hailo GenAI Model Zoo.

- **YOLOv8s: one detector for every camera.** Frames from all streams go through one batched
  network. Fine-tuning keeps the YOLOv8s architecture and changes only the classes (and, for the
  warehouse detector, the input size: 960×960 or 544×960), so both detectors are built with
  Hailo's own YOLOv8s recipe.
- **CLIP for the double-check, not a chat model.** The check only has to answer yes or no, so a
  frozen image encoder plus one small matrix multiply on the host is enough: 15 ms per alert on
  the PC.
- **Qwen2.5-1.5B through the Ollama API.** On the UGen300, hailo-ollama serves the same API, so
  moving the LLM is a change of one setting (`llm.base_url`). A small model is enough because code
  checks every rule it drafts and does all the counting in the report.
- **Whisper-Small** is available ready-made in the Hailo GenAI Model Zoo.

## Already done on the Hailo toolchain

We do not have the device yet, but both detectors and the double-check's image encoder have gone
through Hailo's Dataflow Compiler 5.4.0 with the Hailo Model Zoo recipes. We measured what INT8
costs in the compiler's quantized emulator, which models the chip's INT8 arithmetic; Stage II
confirms the accuracy on the device itself. The scripts are in `hailo/`.

- **Hard-hat detector compiled to a HEF**, the file format the Hailo-10H runs. The fine-tuned
  YOLOv8s was exported to ONNX, quantized to INT8 with 512 calibration images and compiled. Box
  decoding and NMS ship inside the HEF, as in Hailo's own YOLOv8s build, and the config already
  points to the file.
- **INT8 costs almost nothing.** On the same 2,001 test images, the float Hailo graph scores 0.893
  mAP50 and the INT8 version 0.889. The float graph matches the PyTorch model, so the conversion
  itself is exact.
- **Warehouse detector compiled to HEFs, at two input sizes.** The 6-class YOLOv8s went through the
  same recipe (512 calibration frames from its training scenes only, activation clipping). On its
  200 validation frames INT8 scores 0.947 mAP50, the same as float, at 960×960; at the cameras' 16:9
  shape, 544×960, it scores 0.951 with 43 % fewer pixels. That input is about 1.28 times the work of
  a 640×640 frame instead of 2.25 times. The network needs 96 layer controllers and one context on
  the chip holds 80, so the compiler splits it: 3 contexts at 960×960, 4 at 544×960. The chip
  runs the contexts in turn for every batch, so which build is faster is measured on the device.
  The builds ran on a cloud GPU (Colab T4), which the compiler's TensorFlow supports.
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
| Warehouse detector, mAP50 on 200 validation frames, 960×960 | 0.947 (float) | **0.947** |
| Warehouse detector, mAP50, 544×960 (16:9) | 0.948 (float) | **0.951** |
| CLIP image features vs PyTorch (cosine similarity) | 0.9999 (float) | 0.9634 |
| Double-check accuracy (head retrained on INT8 features) | 95.3 % (PC) | **95.3 %** |
| Real violations kept by the double-check | 94.9 % (PC) | 93.9 % |

All detector figures use the NMS settings built into the HEF, so compare 0.889 with 0.893, not
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
  daily report written entirely by the LLM. A rule takes 0.35–0.40 s on average on the PC (at most 0.64 s), so the device is
  slower, but an officer adds a few rules a day and confirms each one anyway.

## Stage II port plan (Nov 6 – Dec 4)

1. **Detectors on the chip (week 1).** Install HailoRT and the UGen300 driver on the host, run the
   detector HEFs through the HailoRT backend and measure throughput and latency over USB, at batch 1
   and 8. Measure warehouse mode with both of its HEFs (960×960 and 544×960, same accuracy, 3 and 4
   contexts) and whether 12 cameras fit on one box.
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
| **The HailoRT backend has not run on hardware.** It is written against the HailoRT 5.x API. | First task of week 1. The HEFs' accuracy is already known from the emulator, so only the plumbing is left to check: output format, image scaling, batching. |
| **USB may be slower than Hailo's PCIe reference figures.** | Hard-hat mode needs about 40 % of the reference capacity, so there is headroom. If it is tight, we lower the frame rate per camera or split the cameras across two boxes. |
| **The warehouse detector is heavier.** Its 960 px input is more work per frame than the 640 px reference, both of its HEFs run in several contexts (3 at 960×960, 4 at 544×960) that the chip swaps for every batch, and the 12-camera limit comes from coverage in the test scenes, not from a compute measurement. | Measure it in week 1, at batch 8 so that each swap is shared by eight frames. The 16:9 build (544×960) keeps the accuracy with 43 % less compute per frame (36.3 vs 64.0 GOPs), though its extra context may cost some of that; if 12 cameras still do not fit, each camera runs at 5 fps or a box covers fewer cameras. |
| **The Hailo build of Qwen2.5-1.5B is quantized differently**, so the PC accuracy (95 % and 84–88 %) may not carry over. Rule parsing also needs hailo-ollama to honour the JSON-schema `format` field. | The checker corrects numbers, times and objects whatever the LLM says. If a reply is not valid JSON, the rule parser and the report fall back to their code-only paths. We re-run the same test sentences on the device. |
| **INT8 shifts the CLIP features** (cosine 0.9634). | Retrained on INT8 features from the emulator, the yes/no head is back at 95.3 %; Stage II retrains it on the chip's own features. |
| **Vision and language models share one chip.** While Qwen2.5 drafts a rule or a report, the detectors get less time. | Language models run for seconds at a time and the detectors have headroom. Stage II measures the camera frame rate while a rule is being drafted. |

Every number and how it was measured: [docs/results](results/README.md). To rebuild the Hailo
files yourself, see [SETUP.md](SETUP.md#rebuild-the-hailo-files).
