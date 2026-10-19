English | [繁體中文](README.zh-TW.md)

# SayGuard

**One ASUS UGen300 per site area · every camera watched · safety rules you can say.**

SayGuard is our ASUS UGen AI League entry (Battlefield Lightning · Workplace AI). It watches
the cameras the site already has, plus a camera wherever a watched zone is seen by fewer than
two cameras, and the safety officer adds new rules by saying them, for example "Warn me if anyone is
within 3 m of a forklift". A new rule about known objects, zones, distances, counts and times
is one sentence, with no data collection and no retraining. A new object class may still need
training: open-vocabulary detection is available, but its accuracy is not measured. Stage II
moves all of this onto one UGen300 (Hailo-10H) next to the site's video recorder.

It has two modes:

- **Hard-hat mode** for construction sites (`configs/site.yaml`). It covers ten independent
  cameras, checks hard hats on every one and double-checks each hard-hat alert with a second
  model (the yes/no check).
- **Warehouse mode** (`configs/warehouse.yaml`). It fuses calibrated cameras on one floor
  plan, so each worker is counted once, not once per camera. Zones and forklift distances are
  in metres, and a worker gets one alert per entry, not one per camera.

This repository is the **Stage I prototype**. The whole pipeline runs locally on a PC GPU
(RTX 5060 Ti); nothing has run on a UGen300 yet. Every job maps to a Hailo-10H model, so Stage II
moves the pipeline onto the UGen300 (see [docs/HAILO_PORT.md](docs/HAILO_PORT.md)). So far only
the hard-hat detector is compiled to a HEF, and it has not run on hardware. Every measured
number on this page comes from [docs/FACTS.md](docs/FACTS.md). Most are taken from the raw
result files in [docs/results/](docs/results/README.md); the few observations from the running
prototype that have no result file are marked **live**. Where something has not been measured,
we say so.

![SayGuard dashboard](docs/img/dashboard_en.png)

## The two modes

### Hard-hat mode (construction sites)

- **Ten streams, one detector.** Frames from all cameras go as one batch into a YOLOv8s
  hard-hat detector, fine-tuned on Roboflow Hard Hats v2 (CC BY 4.0). Heads are tracked per
  camera by a small class-agnostic tracker, and the active rules are checked ten times a
  second per camera. The demo plays 10 looping video clips as its streams.
- **Objects named in a rule.** YOLO-World looks for an object named in a rule ("ladder",
  "excavator") from the word alone, without retraining. Distances to it are estimated from the
  worker's head size. Neither the open-vocabulary detection nor the head-size distance has been
  measured in this mode. On the warehouse demo scene YOLO-World found 0 of 30 forklift boxes,
  so a new object class may still need a trained detector.
- **Fewer false alarms.** An alert fires only if its condition lasts (2 s by default). A
  yes/no check then looks at the crop, using a frozen CLIP ViT-B/32 image encoder with a
  linear 512→2 head ("is this person wearing a hard hat?"). This is a decision, not a
  conversation, so no generative model is needed. A VLM is still available as an option
  (`verifier.backend: vlm`). See the comparison in `docs/results/vlm_check.json`.

### Warehouse mode (calibrated cameras on one floor plan)

- **Demo footage.** The demo uses 11 synchronized, calibrated cameras from NVIDIA
  PhysicalAI-SmartSpaces `Warehouse_010` (synthetic, CC BY 4.0), played frame-locked.
- **Each worker counted once, not once per camera.** Each camera's person box becomes a point
  on the floor: the point under the feet, or the point under the head when the feet are cut
  off. Every point carries its own uncertainty for that camera's view. `sayguard/sitetrack.py`
  fuses the points from all cameras into one worker.
- **Zones and distances in metres.** Zones are drawn once on the floor plan. A vehicle's
  position comes from where its box meets the ground (both bottom corners and the centre),
  so distances to forklifts are floor distances.
- **One alert per worker per entry, not one per camera.** The alert is shown in every camera
  that sees the worker. The officer speaks rules using zone names ("七號走道", "aisle seven",
  "碼頭區").
- **A detector trained for the site.** YOLOv8s with 6 classes: person, forklift, pallet
  truck, transporter, mobile robot and humanoid robot. It was fine-tuned on 8,000 frames from
  8 scenes in 5 synthetic buildings, never on an evaluation scene. Robots have their own
  classes, so a robot detected as a robot is not counted as a worker. Off-the-shelf models did
  not work on these scenes. On Warehouse_010, YOLO-World found 0 of 30 forklift boxes. COCO
  YOLOv8s matched only 7 of them as "truck" and took 39 % of the humanoid robots for people.
  (This was a one-off check noted in `scripts/prepare_site_dataset.py`, not a result file.)
- **Settings chosen on the validation scene only (Warehouse_020)**
  (`docs/results/w020_settings_analysis.json`).
  - A worker is shown once a camera sees them with confidence ≥ 0.7, and stays shown while
    confidence is ≥ 0.6.
  - Vehicles need confidence ≥ 0.6.
  - Forklift distances get a 0.1 m pad.
- **Alert logic** (`sayguard/rules/engine.py`). The first two points are the two alert-logic
  fixes. They were developed on non-test scenes (Warehouse_009, 012, 017 and 020) after the
  frozen run on the test scenes had been seen.
  - A new id counts as a worker who was just alerted (so no second alert) only if it sits on
    that worker while their id is still tracked, or, once their id is gone, if it appears
    near where they were alerted within 10 s or near where they were last seen within 3 s (a
    hand-over). Anyone else, such as the next worker down the same aisle, gets their own alert.
  - Presence rules alert again when a worker re-enters a zone after 10 s outside.
  - A cooldown spaces out reminders while someone stays in place.
- **No worker ids on screen.** Ids change within tens of seconds: 10.72 ids per person in the
  5-minute Warehouse_010 clip (IDF1 0.516). The hard-hat rule is not offered in this mode,
  because the site detector sees whole people and has no hard-hat class.

Both modes:

- **Daily safety report.** Code computes the numbers. The local LLM plans the actions
  (see below).
- **Privacy.** Everything stays on site. In the dashboard tiles and saved snapshots, faces
  are pixelated wherever a head (hard-hat mode) or a person (warehouse mode) is detected. How
  often a face is missed is not measured.

## How the local LLM is used

SayGuard runs a small LLM on site: Qwen2.5-1.5B-Instruct, served through the Ollama API
(Ollama on the PC today; hailo-ollama on the UGen300 in Stage II). It has two jobs:

1. **Rules.** It turns a spoken sentence into a rule, in Mandarin or English, using the
   site's own camera and zone names.
2. **The report plan.** It picks which issues the daily report leads with and a vetted fix
   for each.

A 1.5B model writes fluent text but is not reliable on facts, so both jobs follow one
principle:

> **The LLM proposes. Code checks. The officer confirms every rule.**

- A JSON schema constrains what the model can return.
- Code checks its answer against what the input actually states or what can be computed.
- No rule goes live until the officer confirms it. Nobody confirms the report plan, but it
  can only choose from the listed issues and vetted fixes.
- If the LLM is offline, a deterministic path takes over. For rules, the rule extractor alone
  gets 90 % of the held-out hard-hat-mode sentences and 68 % of the fresh warehouse set. For
  the report, template actions are used.
- The LLM never decides whether an alert fires (the detectors and the rule engine do),
  never writes a number into a report, and never activates a rule by itself.

**Where this was measured.** All LLM accuracy on this page was measured with Ollama
`qwen2.5:1.5b` on the PC GPU. The Hailo GenAI build of Qwen2.5-1.5B uses a different
quantization and has not been measured. Hailo's GenAI Model Zoo reference for Hailo-10H is
7.35 tokens/s, with 0.37 s to the first token. From it we estimate (not measured) that on the
UGen300 (Stage II) a spoken rule takes about 11.3 s to become rule JSON, and Whisper takes
about 2.4 s to transcribe a 5 s rule (`docs/results/benchmark.md`).

### Job 1: turn a spoken sentence into a rule

1. Whisper-Small transcribes the sentence in Mandarin or English. The site's zone and
   camera names are passed to it as a decoder prompt.
2. Qwen2.5 fills a rule schema (type, cameras, zones, object, threshold, time window,
   duration, severity). It is given the site's own camera and zone list.
3. The **rule extractor** (`sayguard/rules/parser.py`) is deterministic code. It reads what
   the sentence states explicitly: numbers, distances, times, the object word, rule-type
   keywords, and zone and camera names. The **checker** uses it: where the rule extractor
   found a field and the LLM disagrees, the extractor's value wins. A field that the extractor
   does not find, for example because of an unusual phrasing, keeps the LLM's value. So the
   checker overrides the numbers, times and objects it finds stated in the sentence, and it
   can miss some phrasings. Invented fields are dropped: a time window the sentence never
   mentions, or a threshold or object that the rule type does not use. The alert text is
   regenerated from the final fields.
4. The dashboard shows the draft, with a note for every override. The officer confirms it
   before the rule goes live.

Two **live** examples from the running server (no result file):

- "Alert me if more than two people are at the site gate after 10 pm": the LLM proposed the
  time window 22:00–00:00. The checker applied the stated convention, 22:00–06:00.
- "A區有人靠近挖土機三公尺內要立刻警告": the LLM proposed the object "excavation pit". The
  checker used "excavator".

On the PC, a parse with LLM + checker takes 0.35–0.40 s on average per rule on the held-out
sets (max 0.64 s).

| Held-out test set (typed sentences, Mandarin + English) | Sentences | LLM alone | Rule extractor alone | **LLM + checker (shipped)** | Latency, mean / max |
|---|---|---|---|---|---|
| Hard-hat mode (per-camera rules) | 20 | 40 % | 90 % | **95 %** | 347 / 469 ms |
| Warehouse mode (floor-plan rules), set 1, first run (before the fixes it exposed) | 25 | 16 % | 80 % | **84 %** | — |
| Warehouse mode (floor-plan rules), set 2, fresh (never used for tuning) | 25 | 24 % | 68 % | **88 %** | 395 / 638 ms |

How the test was run:

- The test sentences are typed text, so speech recognition is not part of these scores. Its
  accuracy on recorded speech is not measured.
- A sentence counts as correct only if every labelled field is right.
- The warehouse sentences (`tests/site_rule_sentences.json`) were written and labelled by AI
  agents, a writer and a reviewer that had never seen the parser code. Set 2 came from a new
  writer–reviewer pair.
- 84 % is the first run of set 1, its only blind result. The parser was then fixed on the
  phrasings that run exposed. Measured again after those fixes, set 1 scores 100 %, but it is
  no longer a test.
- The development sets were used to build the extractor, so their scores do not count as
  tests: hard-hat mode (32 sentences) 25 % / 100 % / 96.9 %, warehouse mode (15 sentences)
  20 % / 100 % / 100 %.

Neither half is enough alone. On the fresh warehouse set, the rule extractor alone gets 68 %
and the LLM alone 24 %. Together they get 88 %.

**Mandarin, English and the site's own names (part of this job).** Officers speak Mandarin or
English, and they name places the way people do: "七號走道", "aisle seven", or "碼頭" for 碼頭區.

- The LLM gets the site's camera list and its zone names in both languages.
- The rule extractor matches a zone name whether its number is written as digits, Chinese
  numerals or English words, and with or without a trailing 區.
- When the model answers in Simplified characters, they are converted to Taiwan Traditional
  (OpenCC). The report also replaces mainland terms (攝像頭 → 攝影機).

Language handling is measured as part of the rule scores above, because every test set mixes
Mandarin and English sentences. It is not scored on its own.

### Job 2: plan the daily report

- Code computes every number, table and the summary from the alert log.
- The LLM sees only the (camera, problem) pairs that actually happened that day, most
  frequent first, plus three vetted fixes for each kind of problem.
- It picks the 3 issues that matter most and one fix for each. It answers with issue and fix
  numbers in a JSON schema whose enums allow only the listed issues and fixes 0–2. So it
  cannot name a camera that had no problem, attach the wrong problem, or invent a number.
- Nobody confirms this plan before the report is written. That is why it can only choose
  listed issues and vetted fixes.
- If the LLM is offline or its answer does not fit the schema, template actions are used.

Results on 20 synthetic days (`scripts/eval_report.py`, Qwen2.5-1.5B through Ollama on the PC).
Each action was scored by 2 blind AI judges: independent judge agents, which are AI models, not
people. The judges did not know which approach wrote the action, and they were given each
camera's real problems that day. They agreed on "wrong problem" for 176 of 177 actions. The
judged run is `docs/results/report_actions_judged.json`; the judges' prompt, the blind batches
and every verdict are in `docs/results/report_actions_judging/`.

| How the three actions are produced | Actions | Wrong problem (both judges) | Sensible (both judges) | Keyword check: misattributed |
|---|---|---|---|---|
| LLM writes the actions freely (earlier prompt) | 60 | 20.0 % | 45.0 % | 13.3 % |
| LLM picks the issue and writes the action; code checks it (`actions="free"`) | 57 | 3.5 % | 52.6 % | 0.0 % |
| **LLM picks the issue and a vetted fix (`actions="menu"`, the default)** | 60 | **0.0 %** | **100.0 %** | 0.0 % |

- In the menu row the wording of every fix is ours, vetted in advance. There, "sensible" rates
  the LLM's choice of issue and fix, not its writing, so that column is not like-for-like with
  the rows above.
- In the `actions="free"` row, 14 of the 57 actions are template text that replaced the LLM's
  text. For the 43 actions the LLM wrote, both judges found the wrong problem in 4.7 % and
  called 37.2 % sensible.

## Results (Stage I, PC prototype)

| | Result | Raw file in `docs/results/` |
|---|---|---|
| Hard-hat detector, held-out test split (2,001 images) | mAP50 **0.9393** (Ultralytics val, score ≥ 0.001), precision 0.9132, recall 0.8986 | `detector_metrics.json` |
| Hard-hat mode, ten cameras at a 10 fps target, one batched detector | **live:** about 105 frames/s analysed, 31–40 ms per 10-frame batch. Estimated 40 % of one UGen300 at the nominal 100 frames/s (from Hailo's YOLOv8s reference, 252 FPS at batch 8; not measured) | none for the live run; estimate in `benchmark.md` |
| Rule parsing, hard-hat mode, 20 held-out typed sentences | LLM alone 40 % → **LLM + checker 95 %** (rule extractor alone 90 %), 347 ms mean | `rule_parsing.json` |
| Rule parsing, warehouse mode, 2 × 25 held-out typed sentences | set 1, first run: LLM alone 16 % → **LLM + checker 84 %**; fresh set 2: LLM alone 24 % → **LLM + checker 88 %** (rule extractor alone 80 % / 68 %) | `rule_parsing_site.json`, `rule_parsing_site_v1_first_run.json` |
| Daily report actions, 20 synthetic days, 2 blind AI judges per action | wrong problem 20.0 % (LLM writes freely) → **0.0 %** (LLM picks issue + vetted fix); sensible 45.0 % → 100.0 % (for vetted fixes the wording is ours, so the judges rated the choice) | `report_actions_judged.json` |
| Alert yes/no check (CLIP ViT-B/32 + linear head), 193 held-out crops | **95.3 %** correct. Passes 95.7 % of the 94 hard-hat crops (labelled dataset crops, our proxy for wrong "no hard hat" flags) and keeps 94.9 % of the 99 bare-head crops (violations); 15.2 ms mean per crop | `probe_train.json` |
| Hard-hat detector compiled to a HEF for Hailo-10H (not yet run on hardware) | INT8 mAP50 **0.889** vs 0.8927 float, mAP50-95 0.56 vs 0.5662, in Hailo's bit-exact emulator. This uses a different evaluation (score ≥ 0.2 NMS, pycocotools) from the 0.9393 above, so compare 0.889 only with 0.8927 | `hailo_int8.json` |
| CLIP encoder quantized to INT8 for Hailo-10H (Hailo's bit-exact emulator, not compiled) | **95.3 %** after retraining the head on INT8 features, with 93.9 % of violations kept (94.9 % on the PC); 94.3 % with the PC head | `clip_probe_int8.json` |
| Warehouse mode, head count on six test buildings the detector never trained on | mean error **0.05–1.74** people per frame with fusion vs 1.76–60.09 when per-camera counts are added up | `site_eval*.json` |
| Warehouse mode, alerts on the same six test buildings, after the two alert-logic fixes | door/dock zones: 53 of 53 events covered, 52 with an alert of their own (98 %), 0 of 52 alerts false. Forklift 3 m rule: 56 of 61 covered, 25 with an alert of their own (41 %), 1 of 26 false. Covered = an alert of its own or inside the cooldown of an alert for the same person (see below). Frozen run before the fixes: 50 of 53 and 54 of 61 covered, 0 of 26 and 1 of 24 alerts false | `site_eval*.json`, `site_by_zone_kind.json`, `frozen/` |

### Warehouse mode on six test buildings the detector never trained on

**Scene roles.**

- **Test:** Warehouse_010, 013, 015, 018, 019 and 022. The detector was never trained on
  these, and they are different buildings from the training scenes. Two caveats:
  - Warehouse_010 is the demo scene. The tracker was developed while watching it, and its
    zones come from the demo config (`configs/warehouse.yaml`), so they were not drawn blind.
  - Warehouse_017, used to develop the alert logic, is the twin of test scene Warehouse_018
    (same building and cameras).
- **Validation:** Warehouse_012 and 020. The confidence and distance settings above were
  chosen on Warehouse_020, which is the training building (scenes 000–007) seen by other
  cameras.
- **Development only:** Warehouse_009 and 017. Some SmartSpaces scenes are same-camera twins
  (same building and cameras): 009 is the twin of training scene 008, and 017 is the twin of
  test scene 018. Buildings were told apart by their floor plans: one building hosts training
  scenes 000–007 and validation scene 020, and no test scene shares a floor plan with a
  training scene.

**How it was measured.**

- The zones of the other five test scenes (Warehouse_013, 015, 018, 019 and 022;
  `configs/eval/*.yaml`) were drawn on camera-coverage maps by AI agents that had not seen any
  result. The same agents gave each zone its kind (door/dock, lane, work area or other).
  Warehouse_010 keeps the demo config's zones, and we assigned their kinds. Every zone's kind
  is listed in `docs/results/site_by_zone_kind.json`.
- `scripts/eval_site.py` runs the same fusion and rule code as the live server and scores it
  against NVIDIA's ground truth. The ground truth never enters the pipeline.
- An event counts as **covered** if the worker got an alert of their own for it, or if it fell
  inside the cooldown of an earlier alert for the same person and rule. By design, the
  cooldown suppresses a second alert in that case. **Strict recall** counts only the events
  that got an alert of their own.

| Test scene | Cameras | People seen by a camera (true mean) | Warehouse mode count (mean error) | Per-camera counts added (mean error) |
|---|---|---|---|---|
| Warehouse_010 | 11 | 24.97 | 26.56 (1.74) | 51.64 (26.67) |
| Warehouse_013 | 10 | 15.83 | 15.49 (0.55) | 39.99 (24.16) |
| Warehouse_015 | 10 | 22.0 | 22.91 (1.05) | 82.09 (60.09) |
| Warehouse_018 | 3 | 1.0 | 1.05 (0.05) | 2.76 (1.76) |
| Warehouse_019 | 3 | 2.07 | 1.78 (0.34) | 4.5 (2.45) |
| Warehouse_022 | 4 | 9.8 | 8.48 (1.35) | 13.51 (3.86) |

| Rule kind, all six test scenes | Covered | Strict recall (own alert) | False alerts | Frozen run: covered | Frozen: strict recall | Frozen: false alerts |
|---|---|---|---|---|---|---|
| Door / dock zones | 53 of 53 (100 %) | 52 of 53 (98 %) | 0 of 52 | 50 of 53 (94 %) | 26 of 53 (49 %) | 0 of 26 |
| Forklift 3 m | 56 of 61 (92 %) | 25 of 61 (41 %) | 1 of 26 | 54 of 61 (89 %) | 23 of 61 (38 %) | 1 of 24 |
| Narrow lanes / aisles | 98 of 118 (83 %) | 71 of 118 (60 %) | 3 of 77 (4 %) | 76 of 118 (64 %) | 39 of 118 (33 %) | 1 of 40 |
| Work areas | 323 of 338 (96 %) | 237 of 338 (70 %) | 26 of 362 (7 %) | 281 of 338 (83 %) | 147 of 338 (43 %) | 16 of 244 |
| Other zones | 43 of 48 (90 %) | 34 of 48 (71 %) | 6 of 44 (14 %) | 34 of 48 (71 %) | 21 of 48 (44 %) | 4 of 30 |

The headline numbers are from the final run, after two alert-logic fixes (re-appear /
hand-over and re-entry). The fixes were developed on non-test scenes (Warehouse_009, 012, 017
and 020) after the **frozen run** (`docs/results/frozen/`) on the test scenes had been seen.
The frozen run used the same detector and confidence settings and is the cleaner held-out
result. The fixes raised coverage, and false alerts rose with them: work areas from 16 of 244
to 26 of 362, lanes from 1 of 40 to 3 of 77, other zones from 4 of 30 to 6 of 44. Both runs are
published.

The forklift rule's strict recall (41 %) is low by design. Its 60 s cooldown deliberately does
not re-alert a worker who steps in and out of 3 m of a parked forklift, so those repeat events
are covered without an alert of their own.

On Warehouse_010 (the demo scene), positions had recall 0.928 and precision 0.873, with a
mean position error of 0.209 m. Worker-to-forklift distance had a median error of 0.2 m
(MAE 0.28 m).

**Limits.**

- **Validation scene Warehouse_020.** It has 16 cameras, outside the 12-camera spec, and the
  most people (49.02 on average). There the forklift rule covered 0.568 of its 95 events
  (strict recall 0.337), with 16 of 51 alerts false, and person recall was 0.519. Person
  detection there was 68 % for people 20–40 px tall and 92 % at 40–80 px, which is where the
  40 px figure in the spec below comes from.
- **Forklift detection varies by scene.** Forklifts were found in 0.986 of forklift-frames
  on Warehouse_010 (the demo scene), but only in 0.169 on Warehouse_018, 0.115 on
  Warehouse_019 and 0.793 on Warehouse_022. Why the detector misses them in those scenes is
  not established.
- **Worker ids are not stable.** They change within tens of seconds: 10.72 ids per person in
  5 minutes on Warehouse_010, 22.32 on Warehouse_020. IDF1 is 0.287–0.977 across the test
  scenes, which is why no id is shown.
- **All warehouse scenes are synthetic.** No real warehouse footage has been measured.
- **The warehouse detector has no Hailo HEF yet.**

**Deployment spec that follows from this:** one UGen300 per area of ≤ 12 cameras, and every
watched zone seen by ≥ 2 cameras at ≥ 40 px. The 12-camera limit is a coverage spec from the
test scenes. Whether one UGen300 can carry warehouse mode's compute load (the 960 px site
detector) is not measured; that is Stage II week 1. The dashboard's "Est. UGen300 load"
readout in warehouse mode uses the 640 px hard-hat reference, so it understates the load.

The full method, per-scene numbers and limits are in [docs/SITE_MODE.md](docs/SITE_MODE.md)
and [docs/results/](docs/results/README.md).

**Not measured yet (Stage II):**

- Anything on the UGen300 itself: HEF throughput and latency over USB, GenAI latency on
  hailo-ollama, device power and end-to-end alert latency.
- Warehouse mode's compute load on the UGen300 (960 px site detector).
- Rule-parsing and report-action accuracy with the Hailo GenAI build of Qwen2.5-1.5B. Every
  LLM figure above is from Ollama on the PC.
- Speech recognition accuracy on recorded speech. The rule-parsing test sets are typed.
- Open-vocabulary detection accuracy and head-size distance error in hard-hat mode.
- How often face pixelation misses a face.

Unit tests: 264 passing (`python -m pytest`).

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
python scripts/fetch_videos.py list            # 10 Mixkit pages: download each clip by hand (docs/VIDEO_SOURCES.md)
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

pip install motmetrics                       # needed by eval_site.py, not in requirements.txt
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
- **What a rerun cannot reproduce.**
  - The first run of warehouse sentence set 1 (`rule_parsing_site_v1_first_run.json`): the
    parser has since been fixed on those sentences.
  - The frozen warehouse run: it predates the two alert-logic fixes.
  - The blind AI-judge scores: the judge step is not scripted. The judged run is kept as
    `report_actions_judged.json`, with the prompt, batches and verdicts in
    `report_actions_judging/`. Rerunning `eval_report.py` rewrites `report_actions.json`
    without them.
- **Hailo builds and emulator checks** are in `hailo/`. See
  [docs/HAILO_PORT.md](docs/HAILO_PORT.md).

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
configs/eval/        one config per evaluation scene except Warehouse_010 (zones drawn on coverage maps by AI agents blind to results)
docs/                FACTS.md (every published number), SITE_MODE.md (warehouse mode end to end),
                     results/, Hailo port plan, licences, video sources, demo script, pitch outline
tests/               unit tests (rules, parser, site rules, cross-camera tracking, tracker, events,
                     report, verifier, ASR decode, synced sources) and the warehouse rule sentences
start_demo.cmd       Windows launcher for the hard-hat demo
start_demo_warehouse.cmd  Windows launcher for the warehouse demo
```

## Licences

SayGuard is released under the [GNU AGPL-3.0](LICENSE), because it builds on Ultralytics,
which is AGPL-3.0. Data, models and dependencies are listed in
[docs/LICENSES.md](docs/LICENSES.md).

- **Hard-hat mode footage:** free Mixkit clips, downloaded by hand and not redistributed
  ([docs/VIDEO_SOURCES.md](docs/VIDEO_SOURCES.md)).
- **Warehouse scenes:** "NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2026), CC BY 4.0",
  downloaded by `scripts/fetch_smartspaces.py` and never committed.
- **Models:** Qwen2.5-1.5B-Instruct is Apache-2.0. Whisper-Small and CLIP are MIT.
