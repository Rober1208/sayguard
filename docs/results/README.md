English | [繁體中文](README.zh-TW.md)

# Stage I results

Everything on this page was measured on the PC prototype (Windows 11, NVIDIA GeForce RTX 5060 Ti)
on 2026-10-18 and 2026-10-19, unless it is marked **reference** (Hailo's published figures) or
**estimate** (derived from them). Stage I runs everything on this PC; nothing has run on a UGen300
yet. The raw JSON files sit next to this page (the few live-run figures that no file records say so
where they appear); `frozen/` keeps the first warehouse-mode run (see "Frozen run vs final run").
[`../FACTS.md`](../FACTS.md) collects the headline numbers from these files.

Our rule: a feature is shown only where it was measured, and every section says what was not;
"Not measured yet" at the end collects the gaps.

**Where the local LLM is measured.** SayGuard gives one small language model
(Qwen2.5-1.5B-Instruct, served through the Ollama API: Ollama on the PC GPU in Stage I, hailo-ollama
on the UGen300 in Stage II) two jobs: turning a rule into structured form (Mandarin or English, with
the site's own zone names) and planning the daily report. For rules, **the LLM proposes, code
checks, the officer confirms.** The report plan is confirmed by no one, but the LLM can only choose
among the issues code lists and the vetted fixes we wrote.

| Job | The LLM | Code | Measured (section below) |
|---|---|---|---|
| Rule → rule JSON, hard-hat mode | drafts the rule as schema-constrained JSON | the deterministic checker overrides the numbers, distances, times and objects it finds stated in the sentence (it can miss phrasings); the officer confirms before the rule goes live | held-out, typed sentences: LLM alone 40 %, rule extractor alone 90 %, LLM + checker **95 %** ("Spoken rules — hard-hat mode") |
| Rule → floor-plan rule, warehouse mode | the same, with floor zones and the site detector's objects | the same checker, with zone names as people say them ("aisle seven", "七號走道"); the officer confirms | fresh held-out, typed sentences: LLM alone 24 %, rule extractor alone 68 %, LLM + checker **88 %** ("Spoken rules — warehouse floor plan") |
| Daily report plan | picks 3 issues from the (camera, problem) pairs code lists as having happened, and one of three vetted fixes for each; no one confirms the plan | computes every number, table and the summary, and supplies the text | actions on the wrong problem (2 blind AI judges): free text 20.0 % → menu **0.0 %** ("Daily report actions") |

All LLM accuracy on this page was measured with Ollama's qwen2.5:1.5b on the PC GPU. The Hailo
GenAI build of the same model (a different quantization) is not measured, and neither is its speed:
on the UGen300 (Stage II) a rule JSON is estimated at about 11.3 s (`benchmark.md`, estimate).

## Hard-hat detector — held-out test split (`detector_metrics.json`)

YOLOv8s fine-tuned on Roboflow Hard Hats v2 (CC BY 4.0), 2 classes. Test split: 2,001 images the
model was never trained on (4,479 hard hats, 1,038 bare heads). Ultralytics validation defaults
(confidence 0.001, IoU 0.7, 640 px, fp16); precision and recall are at the max-F1 confidence.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All | 0.9132 | 0.8986 | **0.9393** | 0.5945 |
| Hard hat | 0.9194 | 0.8964 | 0.9447 | 0.5464 |
| Bare head (violation) | 0.9069 | 0.9008 | 0.9338 | 0.6425 |

11.14 M parameters, 28.6 GFLOPs, 22.52 MB; 2.098 ms inference per image on the RTX 5060 Ti. ONNX
(opset 11) exported for the Hailo Dataflow Compiler.

## Ten cameras through one detector (live run)

Hard-hat mode on 10 looping demo streams, PC prototype.

| Metric | Value |
|---|---|
| Cameras × target fps per camera (hard-hat mode) | 10 × 10 |
| Frames analysed per second (measured) | about 105 |
| Detector batch | 10 frames, 31–40 ms (PC) |
| Estimated UGen300 load | about **40 %** (estimate: the nominal 100 frames/s, 10 cameras × 10 fps, against the YOLOv8s reference of 252 FPS at batch 8; `benchmark.md` 39.7 %) |
| Warehouse mode, site detector | one 11-frame batch at 960 px: about 63–95 ms (PC) |

No JSON file records these live-run figures. No UGen300 estimate is given here for warehouse mode:
the site detector's HEF has not been built, and the 960 px detector's load on the UGen300 is not
measured (Stage II week 1). The dashboard's "Est. UGen300 load" readout in warehouse mode uses the
same 640 px YOLOv8s reference, so it understates the load. Capacity table and PC batch sweep:
`benchmark.md` (written by `scripts/benchmark.py`).

## Spoken rules — hard-hat mode (`rule_parsing.json`, `scripts/eval_rules.py`)

Rules for hard-hat mode, where each rule applies to a set of cameras. The test sentences are typed;
speech recognition accuracy is not measured. A rule counts as correct only if every stated field is
right: type, object, count or distance, time window, duration (when stated) and the set of cameras
it applies to.

The checker is the deterministic rule extractor run on the same sentence: the numbers, times and
objects it finds stated override the LLM's draft. It can miss phrasings (the misses are listed in
the warehouse section below). With the LLM offline, the rule extractor alone gets 90 % here and
68 % on the fresh warehouse set.

| Parser | Dev set (32 sentences, used to build the extractor) | **Held-out (20 sentences, never used for tuning)** |
|---|---|---|
| LLM alone (Qwen2.5-1.5B) | 25 % | **40 %** |
| Rule extractor alone | 100 % | 90 % |
| **LLM + checker (shipped)** | 96.9 % | **95 %** |

Parse time with LLM + checker on the PC GPU: mean 352 ms (max 502) on the dev set, mean 347 ms
(max 469) on the held-out set. On the UGen300 (Stage II) these are estimates, not measurements:
about 11.3 s for the rule JSON and about 2.4 s for Whisper to transcribe a 5 s spoken rule
(`benchmark.md`). Two cases from the live server show why the checker exists: for "Alert me if more
than two people are at the site gate after 10 pm" the LLM proposed the time window 22:00–00:00 and
the checker applied the stated convention, 22:00–06:00; for "A區有人靠近挖土機三公尺內要立刻警告" the
LLM proposed the object "excavation pit" and the checker used "excavator". Site words ("A區",
"挖土機") are passed to Whisper-Small as a decoder prompt; without it "A區" came back as "H" in
testing.

## Spoken rules — warehouse floor plan (`rule_parsing_site.json`, `scripts/eval_rules.py --site`)

Warehouse rules name floor zones and the site detector's objects (forklift, pallet truck, AMR …) of
`configs/warehouse.yaml`. The sentences (`tests/site_rule_sentences.json`, Mandarin and English,
typed) were written and labelled by an AI writer agent and an AI reviewer agent that never saw the
parser code. A rule counts as correct only if type, object, threshold, time window, duration (when
stated) and the floor zones all match, and the rule covers every camera (a camera sees only part of
a zone; workers are placed on the floor).

| Parser | Site dev (15 sentences; may be used to fix the parser) | Held-out v1, first run (25) | Held-out v1 after the fixes (25; seen since) | **Held-out v2 (25; fresh, never used for tuning)** |
|---|---|---|---|---|
| LLM alone (Qwen2.5-1.5B) | 20 % | 16 % | 16 % | **24 %** |
| Rule extractor alone | 100 % | 80 % | 96 % | 68 % |
| **LLM + checker (shipped)** | 100 % | 84 % | 100 % | **88 %** |

Parse time with LLM + checker, mean (max): dev 363 ms (457), held-out v1 392 ms (515), held-out v2
395 ms (638). The first run of v1 is kept in `rule_parsing_site_v1_first_run.json`.

How the sets were used: v1's first, blind run gave 84 %; the parser was then fixed for the phrasings
it missed and v1 was measured again (100 %), so v1 is no longer clean. v2 was written afterwards by
a new pair of AI writer and reviewer agents, also blind to the parser, and measured once with the
fixed parser — **88 % is the honest number**. The parser fixes cover zone names with numerals or
words ("aisle 7", "aisle seven", "七號走道"), an optional trailing 區 ("碼頭" = "碼頭區"), distance
phrasing, linger verbs ("stays", "逗留"), "N or more people", "AMR"
written next to Chinese text, and making a rule that names a zone cover every camera.

On the fresh v2 set the rule extractor alone produced no rule for several rephrasings ("goes
above 12", "total occupancy … exceeds 40", "碼頭區的人數只要大於十"), and the checked LLM draft got
them right: this is why the LLM stays in the loop. LLM + checker still missed three v2 sentences: a
duration said as "half a minute" (「待了半分鐘」), "six or more people" in aisle seven (threshold),
and "晚上八點以後，倉庫裡只要看到人就通報我" (rule type and threshold). Every parsed rule is shown to the
officer to confirm before it goes live.

## Yes/no check on alerts: a decision, not a chat (`yesno_check.json`, `probe_train.json`)

"Is this person wearing a hard hat?" has two answers, so we compared non-generative classifiers
with the generative VLM on **the same 193 held-out crops** (94 wearing, 99 bare heads), cut exactly
like an alert crop. These are labelled dataset crops: the 94 hard-hat crops are a proxy for the
detector's real false "no hard hat" flags, so "wrong flag filtered" is not a measured false-flag
filter rate. Scripts: `scripts/eval_yesno.py`, `scripts/train_probe.py`, `scripts/eval_vlm.py`.

| Method | Accuracy | Wrong flag filtered | Real violation kept | PC time, one crop | PC time, batched | File |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 zero-shot (two text prompts) | 88.6 % | 86.2 % | 90.9 % | – | 3.49 ms | `yesno_check.json` |
| **CLIP ViT-B/32 frozen + linear head (shipped)** | **95.3 %** | 95.7 % | 94.9 % | **15.2 ms** (p95 24.7) | 3.49 ms | `probe_train.json` (batched time: `yesno_check.json`) |
| ResNet18 fine-tuned crop classifier | 95.3 % | 97.9 % | 92.9 % | – | 1.7 ms | `yesno_check.json` |
| Qwen2-VL-2B, production prompt (generative) | 91.7 % | 89.4 % | 93.9 % | 283 ms (p95 531) | – | `vlm_check.json` |

"One crop" is how an alert is checked: one call per crop after warm-up (`scripts/train_probe.py`
for the probe, `scripts/eval_vlm.py` for the VLM). "Batched" divides a 128-crop batch by its size
and is not comparable with one-crop times. `yesno_check.json` was written before the final runs of
the other two scripts: the shipped row comes from `probe_train.json` (the earlier probe row in
`yesno_check.json` reads 96.8 % / 93.9 %), the VLM row from `vlm_check.json`.

The shipped head (`scripts/train_probe.py`, 2,950 training crops, weight decay chosen on 578
validation crops, validation accuracy 0.95) is a 2×512 matrix on top of the CLIP image feature. We
chose it over ResNet18 (same accuracy) because it keeps more real violations, uses the CLIP weights
the open-vocabulary detector already loads, and a new yes/no question only needs a new head, not a
new network. It is also far smaller and faster than the VLM.

Because the yes/no check keeps 94.9 % of real violations, not all of them (about 5 % are filtered,
1 − 0.949, `probe_train.json`), filtered alerts stay in the feed (dimmed) rather than being
deleted. The VLM remains selectable (`verifier.backend: vlm`).

## Daily report actions (`report_actions.json`, `scripts/eval_report.py`)

Every number, table and the summary of the daily report are computed from the alert log by code
(default `mode="hybrid"`); the local LLM only plans the three recommended actions, and no one
confirms that plan (it can only choose listed issues and vetted fixes). In testing, Qwen2.5-1.5B
writing the whole report passed a numbers-in-stats check but still misattributed counts and
invented terms, so full-LLM mode is opt-in (`build_report(..., mode="llm")`).

How the actions are made (`sayguard/report.py`):

- **menu** (default, `actions="menu"`): code lists the (camera, problem) pairs that actually
  happened, most frequent first, plus the cameras with filtered false alarms. The LLM picks 3
  different issues and, for each, one of three vetted fixes for that kind of problem. Its reply is
  held to a JSON schema whose fields are enums (issue numbers from the list, fix 0–2), so it writes
  no text and cannot name a camera or a problem that did not happen.
- **structured** (`actions="free"`): the LLM picks 3 issues by number and writes one action for
  each; code fills in the camera and the problem, and an action that names another camera, a number
  not in the stats or another kind of problem gets the template text instead.
- **free text** (the earlier prompt, kept as the baseline): 3 markdown bullets from a few facts,
  checked only for invented numbers.

Test: 20 synthetic days (English and Chinese alternating), the same days for all three
approaches, Qwen2.5-1.5B (Ollama's qwen2.5:1.5b on the PC GPU).

**Keyword check** (automatic): an action names a camera when that camera's configured name appears
in its text (ignoring case), and is misattributed when it names a camera together with a kind of problem that camera
did not have that day.

| Approach | Days accepted | Actions | Name a camera | Misattributed |
|---|---|---|---|---|
| Free text (earlier prompt) | 20/20 (100 %) | 60 | 45 | 8 (13.3 %) |
| Structured (`actions="free"`) | 19/20 (95 %) | 57 | 57 | 0 (0.0 %) |
| **Menu (default)** | 20/20 (100 %) | 60 | 60 | **0 (0.0 %)** |

**Independent blind judges**: 2 AI judges (independent judge agents, not people) per action, blind
to which approach wrote it, given each camera's real problems that day. They agreed on "wrong
problem" for 176 of 177 actions.

| Approach | Actions | Wrong problem (both judges) | Wrong problem (either judge) | Sensible (both judges) | Names a camera |
|---|---|---|---|---|---|
| Free text | 60 | 20.0 % | 20.0 % | 45.0 % | 88.3 % |
| Structured | 57 | 3.5 % | 5.3 % | 52.6 % | 100.0 % |
| · text written by the LLM | 43 | 4.7 % | 7.0 % | 37.2 % | 100.0 % |
| · replaced by template text | 14 | 0.0 % | 0.0 % | 100.0 % | 100.0 % |
| **Menu (default)** | 60 | **0.0 %** | **0.0 %** | **100.0 %** | 100.0 % |

The two "names a camera" figures for free text measure different things: the keyword check counts
only a camera's configured name in the text (ignoring case; 45 of 60), while the judges also counted a camera referred
to in other words (53 of 60, 88.3 %; the two judges agreed on all 60). The 8 extra are all
Chinese-language actions that translate the camera's name, such as 焊接區 for Welding Bay
(`report_actions_judging/`). For menu actions "sensible" rates the choice of issue and fix, since
the wording is ours (vetted), so its 100.0 % is not like-for-like with the rows where the LLM wrote
the text.

The keyword check is too lenient: it found no misattribution in structured mode, where the judges
found 3.5–5.3 %. When the 1.5B model writes the text itself, both judges called only 37.2 % of its
actions sensible; choosing from vetted fixes removed wrong-problem actions on these days. So the
LLM's job is the part code cannot do — deciding which issues matter and which fix fits — and
nothing it outputs can be a wrong camera or a wrong number.

Limits: synthetic days, not a real site's log. The judges are AI agents, not safety officers, and
their prompt, the blind batches, the key and every verdict are in
[report_actions_judging/](report_actions_judging/README.md); the judged run is kept as
`report_actions_judged.json`. `scripts/eval_report.py` reproduces the keyword check only;
re-running it rewrites `report_actions.json` without the `independent_judges` block.

## Warehouse mode (`site_eval*.json`)

Warehouse mode fuses the calibrated cameras of one warehouse on its floor plan: each worker is
counted once, not once per camera, zones and distances are in metres on the floor, and a rule raises
one alert per worker per entry, not one per camera (how it works:
[`../SITE_MODE.md`](../SITE_MODE.md)). `scripts/eval_site.py` runs the exact live fusion
(`SiteFusion` + `RuleEngine.evaluate_site`) frame by frame over synchronized streams and scores it
against NVIDIA's ground truth (PhysicalAI-SmartSpaces, synthetic, CC BY 4.0), which
knows every person's floor position and identity in every camera. Ground truth never enters the
pipeline. Each scene: 3,000 frames at 10.0 fps (W022: 3,090).

### What is measured

- **People**: mean over frames of the true head count (people seen by at least one configured
  camera), of warehouse mode's worker count, and of the sum of per-camera person tracks (what a
  per-camera system counts); mean absolute error of each.
- **Positions**: workers matched to true people within 1 m on the floor: recall, precision, MOTA,
  mean position error, duplicates (share of person-frames with two worker ids on one person at
  once), IDF1, and how many ids each real person was split into.
- **Forklifts**: share of forklift-frames (a forklift visible in a configured camera) with a
  detected forklift whose ground contacts reach its footprint (2.5 m slack); measured
  worker-to-forklift distance minus the true gap to the forklift's 3D footprint (median absolute
  error, MAE, and mean signed error = bias).
- **Alerts**: a ground-truth event is a person within 3 m of a forklift for 2 s (rule "Keep 3 m
  from forklifts", 60 s cooldown) or inside a floor zone for 3 s (`--all-zones` adds a 3 s presence
  rule with a 60 s cooldown for every zone; on W010 the aisle-7 rule is the config's own, also 3 s
  and 60 s); gaps under 1 s do not split an event. An alert belongs to an event of the same rule if
  its worker is within 2.5 m of that event's person at alert time; an alert that belongs to no
  event is **false**.
  - **covered** = events with an alert of their own + events that start within the cooldown of an
    alert on the same person (no new alert by design: the cooldown spaces reminders);
  - **strict recall** = events with an alert of their own;
  - **per alerted event** = alerts on events ÷ events with an alert (1.0 = no repeats).
- **Per-camera mode** (baseline): the same detections and the same floor distances, but each
  camera's tracks alert on their own, with the duplicate guards SayGuard used per camera before
  (same-place IoU, 15 s per camera, one presence subject per camera). The only difference from
  warehouse mode is one identity per worker across cameras.

### Scene roles

SmartSpaces scenes come in pairs with the same building and cameras (000/001 … 006/007, 008/009,
017/018, 020/021), and W000–W007 and W020 are all one building. The site detector (YOLOv8s, 6
classes: person, forklift, pallet truck, transporter, mobile robot, humanoid robot) was fine-tuned on
8,000 images from 8 training scenes in 5 synthetic buildings (W000, 002, 004, 006, 008, 011, 014, and
W016 cameras 0–8; W016's cameras 9 and 10 were the training-time validation split) at 960 px, never
on an evaluation scene.

| Role | Scenes | Use |
|---|---|---|
| **Test** | W010, W013, W015, W018, W019, W022 | six test buildings the detector never trained on, distinct from the training buildings; the measured settings were chosen on W020, not on them; see the two caveats below |
| Validation | W012, W020 | the settings below were chosen on W020, which is the same building as training scenes W000–W007 with other cameras, and has 16 cameras (outside the 12-camera spec) |
| Development only | W009 (twin of training scene W008), W017 (twin of test scene W018) | not used as evidence |

Two test scenes are not fully independent:

- **W010 is the demo scene.** The tracker was developed while watching it (`sayguard/sitetrack.py`
  cites its ground truth), and its zones come from the demo config, not drawn blind. Its numbers are
  less independent than those of the other five test scenes.
- **W018 and the alert logic.** W017, used to develop the alert logic, is the same building and
  cameras as test scene W018. The two alert-logic fixes behind the final run were developed on
  W009, W012, W017 and W020 after the frozen run on the test scenes had been seen (see "Frozen run vs
  final run").

The zones of every scene except W010 (`configs/eval/*.yaml`) were drawn by AI agents that saw only
each scene's coverage map — where at least two cameras see a standing person ≥ 40 px tall — before
any result; the same agents labelled each zone's kind (`site_by_zone_kind.json`). W010 uses the
zones of the demo config, `configs/warehouse.yaml`, with kinds assigned by us.

### Settings chosen on the validation scene (W020) only

- A worker is shown (map, head count, rules) once a camera sees it with confidence ≥ 0.7, and kept
  while one still sees it with ≥ 0.6 (`SHOW_CONF`, `KEEP_CONF` in `sayguard/site_pipeline.py`);
  vehicles and robots need ≥ 0.6 (`OBJECT_CONF`).
- Forklift distance pad 0.1 m (`VEHICLE_PAD_M` in `sayguard/sitetrack.py`): the forklift distance
  bias on W020 went from -0.38 m to 0.01 m. The pads for the other vehicle classes are rough
  guesses, not measured.
- "In view" = at least 40 px tall at 1280×720: on W020 person detection was 68 % at 20–40 px and
  92 % at 40–80 px.

The grids behind these choices are in `w020_settings_analysis.json`.

The deployment spec: **one UGen300 per area of ≤ 12 cameras; every watched zone seen by ≥ 2 cameras
at ≥ 40 px.** The 12-camera figure is a coverage spec from the test scenes, not a measured compute
limit: warehouse-mode load on the UGen300 is not measured (Stage II week 1).

### Test scenes: people

| Scene | Cameras | True people (mean) | Warehouse mode (abs. error) | Sum of cameras (abs. error) |
|---|---|---|---|---|
| W010 | 11 | 24.97 | **26.56** (1.74) | 51.64 (26.67) |
| W013 | 10 | 15.83 | **15.49** (0.55) | 39.99 (24.16) |
| W015 | 10 | 22.0 | **22.91** (1.05) | 82.09 (60.09) |
| W018 | 3 | 1.0 | **1.05** (0.05) | 2.76 (1.76) |
| W019 | 3 | 2.07 | **1.78** (0.34) | 4.5 (2.45) |
| W022 | 4 | 9.8 | **8.48** (1.35) | 13.51 (3.86) |

Adding up cameras counts anyone two cameras see twice (82.09 for 22.0 people on W015); warehouse
mode's mean error stays between 0.05 and 1.74 people on the six test scenes.

### Test scenes: positions

| Scene | Recall | Precision | MOTA | Mean position error | Duplicates | IDF1 | Ids per person |
|---|---|---|---|---|---|---|---|
| W010 | 0.928 | 0.873 | 0.788 | 0.209 m | 2.35 % | 0.516 | 10.72 |
| W013 | 0.961 | 0.982 | 0.941 | 0.176 m | 1.55 % | 0.633 | 4.94 |
| W015 | 0.987 | 0.948 | 0.932 | 0.173 m | 3.35 % | 0.701 | 3.64 |
| W018 | 1.0 | 0.955 | 0.953 | 0.231 m | 0.0 % | 0.977 | 1.0 |
| W019 | 0.842 | 0.977 | 0.82 | 0.26 m | 1.45 % | 0.59 | 6.33 |
| W022 | 0.83 | 0.959 | 0.791 | 0.243 m | 1.2 % | 0.287 | 6.58 |

Positions are good (0.173–0.26 m mean error); identities are not: worker ids change within tens of
seconds (10.72 ids per person over the 5-minute W010 clip, about one per 28 s; IDF1 0.287 on W022,
0.516 on W010). That is why the dashboard does not show worker ids and why the alert logic
decides "same worker or someone else" itself (see "Frozen run vs final run").

### Test scenes: forklifts

| Scene | Forklift found (share of forklift-frames) | Distance error, median | MAE | Bias |
|---|---|---|---|---|
| W010 | 0.986 | 0.2 m | 0.28 m | -0.12 m |
| W018 | 0.169 | 0.45 m | 0.65 m | -0.02 m |
| W019 | 0.115 | 0.38 m | 0.44 m | -0.08 m |
| W022 | 0.793 | 0.47 m | 0.6 m | 0.43 m |

W013 and W015 have no forklift in view of their cameras. Forklift detection ranges from 0.986 (W010,
the demo scene) down to 0.169 and 0.115 (W018, W019); the causes of the misses are not established.

### Test scenes: alerts per rule

Kind = the zone kind used in the aggregate below ("–" for zones with no events). Covered is shown as
own alert + in cooldown = share of events.

| Scene | Rule / zone | Kind | Events | Warehouse mode: covered | Strict recall | Alerts (false) | Per alerted event | Per-camera: covered | Per-camera: alerts (false) |
|---|---|---|---|---|---|---|---|---|---|
| W010 | Keep 3 m from forklifts | forklift | 42 | 18 + 22 = 0.952 | 0.429 | 18 (0) | 1.0 | 0.905 | 23 (0) |
| W010 | No walking in aisle 7 (forklift parked) | lane | 9 | 8 + 1 = 1.0 | 0.889 | 8 (0) | 1.0 | 0.889 | 10 (0) |
| W010 | dock | door/dock | 22 | 21 + 1 = 1.0 | 0.955 | 21 (0) | 1.0 | 0.364 | 14 (0) |
| W010 | racks | work | 43 | 38 + 5 = 1.0 | 0.884 | 56 (13) | 1.13 | 0.186 | 31 (0) |
| W010 | open_floor | work | 48 | 42 + 6 = 1.0 | 0.875 | 130 (7) | 2.93 | 0.021 | 45 (0) |
| W013 | west_cross_aisle | lane | 17 | 11 + 5 = 0.941 | 0.647 | 12 (0) | 1.09 | 0.529 | 34 (0) |
| W013 | north_staging_area | work | 11 | 9 + 1 = 0.909 | 0.818 | 11 (2) | 1.0 | 0.455 | 25 (5) |
| W013 | south_open_floor | work | 36 | 26 + 9 = 0.972 | 0.722 | 29 (0) | 1.12 | 0.028 | 29 (0) |
| W013 | west_rack_line | – | 0 | – | – | 0 (0) | – | – | 0 (0) |
| W015 | north_cross_aisle | lane | 40 | 27 + 9 = 0.9 | 0.675 | 27 (0) | 1.0 | 0.25 | 22 (0) |
| W015 | south_staging_floor | work | 120 | 56 + 54 = 0.917 | 0.467 | 59 (3) | 1.0 | 0.075 | 40 (0) |
| W015 | west_rack_aisle | work | 46 | 38 + 7 = 0.978 | 0.826 | 48 (0) | 1.26 | 0.022 | 38 (0) |
| W015 | east_rack_aisle | other | 48 | 34 + 9 = 0.896 | 0.708 | 44 (6) | 1.12 | 0.125 | 40 (0) |
| W018 | Keep 3 m from forklifts | forklift | 8 | 4 + 4 = 1.0 | 0.5 | 5 (1) | 1.0 | 1.0 | 7 (1) |
| W018 | staging_west_walkway | – | 0 | – | – | 0 (0) | – | – | 3 (3) |
| W018 | loading_dock_staging | door/dock | 10 | 10 + 0 = 1.0 | 1.0 | 10 (0) | 1.0 | 0.5 | 15 (0) |
| W018 | north_open_floor | – | 0 | – | – | 0 (0) | – | – | 0 (0) |
| W019 | Keep 3 m from forklifts | forklift | 11 | 3 + 5 = 0.727 | 0.273 | 3 (0) | 1.0 | 0.727 | 4 (0) |
| W019 | dock_side_aisle | lane | 1 | 1 + 0 = 1.0 | 1.0 | 2 (1) | 1.0 | 1.0 | 7 (4) |
| W019 | dock_staging_area | door/dock | 21 | 21 + 0 = 1.0 | 1.0 | 21 (0) | 1.0 | 0.333 | 15 (0) |
| W019 | dock_approach_floor | – | 0 | – | – | 0 (0) | – | – | 6 (6) |
| W022 | Keep 3 m from forklifts | forklift | 0 | – | – | 0 (0) | – | – | 0 (0) |
| W022 | main_aisle_r1_r2 | lane | 51 | 24 + 12 = 0.706 | 0.471 | 28 (2) | 1.08 | 0.157 | 9 (1) |
| W022 | north_staging | work | 4 | 2 + 0 = 0.5 | 0.5 | 3 (1) | 1.0 | 0.75 | 3 (0) |
| W022 | picking_aisle_r2_r3 | work | 30 | 26 + 4 = 1.0 | 0.867 | 26 (0) | 1.0 | 0.333 | 22 (0) |

Given the same detections, per-camera mode covers far fewer zone events (W010 dock 0.364 against
1.0, W013 south_open_floor 0.028 against 0.972) and raises false alerts in zones nobody entered
(W019 dock_approach_floor: 6). Warehouse mode is not better everywhere: on the forklift rule
per-camera mode matches it on W018 and W019, and W022 north_staging is lower in warehouse mode (0.5
against 0.75).

### Aggregate by zone kind (six test scenes)

Counts from `site_by_zone_kind.json`. **Covered** = an alert of its own, or inside the cooldown of
an alert for the same person; **strict recall** = an alert of its own. The final run comes after two
alert-logic fixes that were developed on non-test scenes after the frozen run had been seen (see
"Frozen run vs final run"), so both runs are shown, false alerts included.

| Kind | Zones with events | Events | Covered: frozen → final | Strict recall: frozen → final | False alerts: frozen → final |
|---|---|---|---|---|---|
| Door / dock zones | W010 dock; W018 loading_dock_staging; W019 dock_staging_area | 53 | 50 (94 %) → **53 (100 %)** | 26 (49 %) → 52 (98 %) | 0 of 26 → 0 of 52 |
| Forklift 3 m rule | W010, W018, W019 | 61 | 54 (89 %) → **56 (92 %)** | 23 (38 %) → 25 (41 %) | 1 of 24 → 1 of 26 |
| Narrow lanes / aisles | W010 aisle 7; W013 west_cross_aisle; W015 north_cross_aisle; W019 dock_side_aisle; W022 main_aisle_r1_r2 | 118 | 76 (64 %) → **98 (83 %)** | 39 (33 %) → 71 (60 %) | 1 of 40 → 3 of 77 (4 %) |
| Work areas | W010 racks, open_floor; W013 north_staging_area, south_open_floor; W015 south_staging_floor, west_rack_aisle; W022 north_staging, picking_aisle_r2_r3 | 338 | 281 (83 %) → **323 (96 %)** | 147 (43 %) → 237 (70 %) | 16 of 244 → 26 of 362 (7 %) |
| Other zones | W015 east_rack_aisle | 48 | 34 (71 %) → 43 (90 %) | 21 (44 %) → 34 (71 %) | 4 of 30 → 6 of 44 (14 %) |

The forklift rule's strict recall stays at 41 % because its 60 s cooldown deliberately does not
re-alert a worker who steps in and out of 3 m of a parked forklift; "covered" counts those events,
strict recall does not. W010's zones come from the demo config, not drawn blind (see "Scene roles").

### Frozen run vs final run

`frozen/` holds the frozen-parameter run; the top-level `site_eval*.json` files are the final run.
Between the two the alert logic changed; the detector and the confidence settings stayed the same:

1. **Same worker or someone else.** A new worker id near a worker alerted within the cooldown
   counts as that worker only if it stands right on them while they are still tracked (a second id
   on one person) or, once their own id is gone, if it appears within 10 s near where they were
   alerted (`SITE_REAPPEAR_S`) or within 3 s near where they were last seen, the radius growing at
   walking pace (`SITE_HANDOVER_S`). Anyone else — say, the next worker down the same aisle — gets
   their own alert.
2. **Re-entry.** For presence rules, coming back into a zone after 10 s outside (`SITE_REENTRY_S`)
   is a new entry and alerts again, cooldown or not; the cooldown only spaces reminders while
   someone stays.

Both fixes were developed and checked only on W009, W012, W017 and W020, after the frozen run on the
test scenes had been seen (`site_preregistration.md`); W017 is the same building and cameras as test
scene W018. The test scenes were then re-run once.

Events covered, frozen → final: door/dock 94 % → 100 %, forklift 3 m 89 % → 92 %, lanes/aisles
64 % → 83 %, work areas 83 % → 96 %, other zones 71 % → 90 %. Alerts and false alerts rose with
the fixes: door/dock alerts 26 → 52 (none false), lanes 1 of 40 → 3 of 77 false, work areas 16 of
244 → 26 of 362, other zones 4 of 30 → 6 of 44 (table above). The frozen run was measured before
the fixes, so it is the cleaner held-out result; the final run describes the current code.
Alerting on re-entry also means busy work areas get repeat alerts (W010 open_floor: 130 alerts for
48 events, 2.93 per alerted event).

### Validation and development scenes

These scenes are not evidence (W020 chose the settings; W009 and W017 are twins of training or
test buildings), but they show where the method strains.

| Scene | Role | Cameras | True people | Warehouse mode (abs. error) | Sum of cameras (abs. error) | Recall | Precision | MOTA | Position error | Duplicates | IDF1 | Ids per person |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W012 | validation | 8 | 15.0 | 15.05 (0.31) | 30.78 (15.78) | 0.976 | 0.973 | 0.947 | 0.199 m | 2.43 % | 0.651 | 5.53 |
| W020 | validation | 16 | 49.02 | 42.1 (7.19) | 97.13 (48.11) | 0.519 | 0.605 | 0.168 | 0.348 m | 2.63 % | 0.158 | 22.32 |
| W009 | development | 12 | 13.0 | 13.99 (1.1) | 56.8 (43.8) | 0.977 | 0.909 | 0.874 | 0.174 m | 7.56 % | 0.485 | 10.0 |
| W017 | development | 3 | 1.75 | 1.17 (0.63) | 3.87 (2.12) | 0.617 | 0.924 | 0.554 | 0.237 m | 0.32 % | 0.7 | 8.0 |

Forklifts: W020 found in 0.802 of forklift-frames (distance error median 0.32 m, MAE 0.52 m, bias
0.01 m); W017 0.331 (median 0.3 m, MAE 0.46 m, bias 0.06 m); W012 and W009 have no forklift in view.

| Scene | Rule / zone | Events | Warehouse mode: covered | Strict recall | Alerts (false) | Per alerted event | Per-camera: covered | Per-camera: alerts (false) |
|---|---|---|---|---|---|---|---|---|
| W012 | west_aisle | 23 | 22 + 0 = 0.957 | 0.957 | 23 (0) | 1.05 | 0.304 | 17 (0) |
| W012 | north_staging_area | 41 | 27 + 14 = 1.0 | 0.659 | 28 (0) | 1.04 | 0.146 | 20 (0) |
| W012 | south_staging_area | 32 | 16 + 11 = 0.844 | 0.5 | 16 (0) | 1.0 | 0.312 | 15 (0) |
| W012 | west_rack_face | 0 | – | – | 0 (0) | – | – | 0 (0) |
| W020 | Keep 3 m from forklifts | 95 | 32 + 22 = 0.568 | 0.337 | 51 (16) | 1.09 | 0.411 | 73 (19) |
| W020 | pallet_row_aisle | 18 | 11 + 4 = 0.833 | 0.611 | 12 (1) | 1.0 | 0.222 | 22 (2) |
| W020 | pallet_staging_bays | 28 | 19 + 2 = 0.75 | 0.679 | 28 (6) | 1.16 | 0.5 | 32 (1) |
| W020 | conveyor_work_cell | 13 | 9 + 1 = 0.769 | 0.692 | 24 (6) | 2.0 | 0.692 | 18 (0) |
| W020 | conveyor_side_walkway | 3 | 3 + 0 = 1.0 | 1.0 | 3 (0) | 1.0 | 1.0 | 8 (0) |
| W009 | north_walkway | 53 | 32 + 12 = 0.83 | 0.604 | 36 (2) | 1.06 | 0.132 | 60 (0) |
| W009 | north_door | 5 | 3 + 1 = 0.8 | 0.6 | 6 (2) | 1.33 | 0.6 | 15 (5) |
| W009 | fenced_staging | 16 | 16 + 0 = 1.0 | 1.0 | 16 (0) | 1.0 | 0.812 | 30 (0) |
| W009 | pallet_block | 0 | – | – | 0 (0) | – | – | 11 (11) |
| W017 | Keep 3 m from forklifts | 32 | 6 + 17 = 0.719 | 0.188 | 6 (0) | 1.0 | 0.906 | 17 (0) |
| W017 | north_approach_aisle | 0 | – | – | 0 (0) | – | – | 1 (1) |
| W017 | dock_doors_apron | 0 | – | – | 0 (0) | – | – | 5 (5) |
| W017 | dock_staging_area | 30 | 10 + 0 = 0.333 | 0.333 | 10 (0) | 1.0 | 0.2 | 15 (0) |

W020 is the hardest scene: 16 cameras (outside the deployment spec), the most people (49.02 on
average), and the same building as the training scenes; MOTA falls to 0.168 and 16 of 51 forklift
alerts are false. Which of these differences causes the drop is not established. On W017 per-camera
mode covers more forklift events than warehouse mode (0.906 against 0.719).

### Limits

- **Synthetic footage only.** No real warehouse has been measured; real cameras would also need
  their own calibration.
- **Test scenes not fully independent.** W010 is the demo scene: the tracker was developed while
  watching it and its zones come from the demo config. W017, used to develop the alert logic, is
  the same building and cameras as test scene W018, and the two alert-logic fixes were made after
  the frozen run on the test scenes had been seen.
- **No retraining after testing.** No detector retraining was possible without using up a test
  scene, so the misses stay in these numbers: forklifts were found in only 0.169 and 0.115 of
  forklift-frames on W018 and W019 (0.793 on W022, 0.986 on W010); their causes are not established.
- **Identity is short-lived** (test IDF1 0.287–0.977, up to 10.72 ids per person; ids change within
  tens of seconds): alerts are per worker within a cooldown, not per long-term identity, and worker
  ids are never shown.
- **Distances**: the forklift bias on W022 is 0.43 m, measured farther than the truth, which is the
  unsafe direction. Distances to other vehicle classes are not measured.
- **Repeats and false alerts in busy areas**: W010 racks 13 false of 56 alerts; W010 open_floor
  2.93 alerts per alerted event.
- **Strict recall is low on some rules** (forklift rule 41 % over the test scenes, W019 forklift
  0.273, W015 south_staging_floor 0.467): by design the cooldown holds back a second alert for the
  same person; "covered" counts those events, strict recall does not.
- **Outside the spec**: W020 (16 cameras, but also the most people and the training
  building) reaches MOTA 0.168; the cause is not established.
- One run per scene; no confidence intervals.

### Reproduce

```bash
# W010 with the demo config (configs/warehouse.yaml) -> site_eval.json
python scripts/fetch_smartspaces.py train/Warehouse_010
python scripts/make_site_streams.py train/Warehouse_010
python scripts/eval_site.py --all-zones

# any other scene -> site_eval_warehouse_0XX.json (train/ for 009 012 013 015 017 018 019, val/ for 020 022)
python scripts/fetch_smartspaces.py train/Warehouse_013
python scripts/make_site_streams.py train/Warehouse_013
python scripts/eval_site.py --config configs/eval/warehouse_013.yaml --all-zones

# spoken warehouse rules (needs Ollama with qwen2.5:1.5b) -> rule_parsing_site.json
python scripts/eval_rules.py --site
```

`eval_site.py` writes to the work dir's `results/` and to `docs/results/`. `--dets-cache FILE` stores
the detector output on the first run and replays it afterwards (tracker and rule changes without the
GPU); `--tag NAME` writes an experiment to the work dir only. `frozen/` is a copy of the
frozen-parameter run; it cannot be re-run from this tree, because the earlier alert logic is not
kept. `eval_rules.py --site --dev` runs only the dev set and writes to the work dir
only; a full `--site` run measures held-out v2 again, so it would no longer be a first measurement.

<a id="hailo-10h-compiled-and-checked-bit-exactly-in-hailos-emulator-hailo"></a>

## Hailo-10H: hard-hat detector compiled, both networks checked bit-exactly in Hailo's emulator (`hailo/`)

Without the device, Stage I takes both hard-hat-mode vision networks through the real Hailo
toolchain (Dataflow Compiler 5.4.0 + Hailo Model Zoo recipes, in WSL) and measures what INT8 costs
with the DFC emulator, whose `SDK_QUANTIZED` context reproduces the chip's integer arithmetic. Only
the hard-hat detector is compiled to a HEF, and it has never run on hardware; the CLIP image encoder
of the yes/no check was quantized and checked in the emulator only, with no HEF. The DFC's
TensorFlow does not support RTX 50-series GPUs, so everything ran on the CPU, without the GPU-only
fine-tuning levels: the detector at optimization level 1 (equalization + bias correction)
with 512 calibration images and activation clipping, the CLIP encoder with the Model Zoo's own
script (level 0, 16-bit layer norms and residual adds).

**Hard-hat detector** (`hailo_int8.json`, `pc_baseline.json`; `hailo/build_helmet.py`,
`hailo/eval_int8.py`, `hailo/eval_pc_baseline.py`). Same 2,001 test images, same COCO evaluator
(pycocotools), same NMS settings as the HEF carries (score ≥ 0.2, IoU 0.7):

| Model | mAP50 | mAP50-95 | Hard hat AP50 | Bare head AP50 |
|---|---|---|---|---|
| PC, PyTorch fp32 (`pc_baseline.json`) | 0.8927 | 0.5663 | 0.9035 | 0.8819 |
| Hailo graph, float (`SDK_FP_OPTIMIZED`) | 0.8927 | 0.5662 | 0.9035 | 0.8819 |
| **Hailo INT8 (`SDK_QUANTIZED`, what the chip computes)** | **0.889** | 0.56 | 0.9008 | 0.8772 |

The float Hailo graph matches PyTorch within 0.0001, so parsing and the model script are exact;
INT8 costs 0.0037 mAP50 and 0.0062 mAP50-95.

The calibration set matters a lot (table below). The calibration images come from the validation split, never
test, but all three variants were evaluated on the same test split and the shipped one was chosen
from these test results: the 64-image variant has the higher mAP50 (0.8911) and bare-head AP50
(0.8841), the shipped variant the higher mAP50-95 (0.56). The published INT8 figure is therefore a
pick made on the test set, not a single blind measurement.

| Calibration | mAP50 | mAP50-95 | Bare head AP50 | File |
|---|---|---|---|---|
| 64 images, random sample (seed 0) | 0.8911 | 0.5396 | 0.8841 | `helmet_c64_optimized_int8.json` |
| 512 images | 0.8513 | 0.5363 | 0.857 | `hailo_int8_c512.json` |
| **512 images + activation clipping (0.01–99.99 percentiles), shipped** | 0.889 | **0.56** | 0.8772 | `hailo_int8.json` |

Our reading (not separately measured): more images widen the activation ranges with rare outliers
and coarsen the INT8 steps; with the ranges clipped to the 0.01–99.99 percentiles, box placement
(mAP50-95) stays closest to float. (The 0.9393 in the detector table above is Ultralytics' own metric at score ≥ 0.001;
pycocotools at that threshold gives 0.9245 (`pc_baseline.json`). The 0.2 score floor of the
HEF's built-in NMS is what lowers both columns here. Compare 0.889 only with 0.8927, never with
0.9393: they are different evaluations.)

The HEF (`helmet_yolov8s.hef`) is compiled with `compiler_optimization_level=max`, as
`hailomz compile --performance` does; it has not been run on hardware yet. The 0.889 above was
measured on the quantized model in the emulator, not by running the HEF. The profiler cannot give
FPS for a multi-context network without the device; the reference for the same YOLOv8s on Hailo-10H
is 166 FPS at batch 1 and 252 FPS at batch 8 (Hailo Model Zoo, **reference**).

**CLIP image encoder for the yes/no check** (`clip_probe_int8.json`; `hailo/clip_probe_int8.py`).
Model Zoo `clip_vit_b_32_image_encoder` with its model script, 64 site crops from the train split
as calibration, the same 193 test crops; quantized and run in the emulator only (no HEF):

| Features → head | Accuracy | Wrong flag filtered | Real violation kept |
|---|---|---|---|
| PC PyTorch (fp16 on the GPU) → head trained on PC features (shipped today) | 95.3 % | 95.7 % | 94.9 % |
| Hailo INT8 → the same PC head | 94.3 % | 95.7 % | 92.9 % |
| **Hailo INT8 → head retrained on INT8 features** | **95.3 %** | 96.8 % | 93.9 % |

Float Hailo features match the PyTorch encoder (cosine 0.9999 to its fp16 GPU features); INT8
features keep cosine 0.9634. Reusing the PC head loses a point of accuracy (94.3 %) and keeps
fewer real violations (92.9 %). Retraining the 2×512 head on the chip's own features recovers
accuracy (95.3 %), with violations kept at 93.9 % against 94.9 % on the PC, so Stage II retrains
the head on device features.

**Warehouse mode**: the site detector (`site_yolov8s`, 6 classes, 960 px) has not been compiled to a
HEF yet (`detector.hef` in `configs/warehouse.yaml` is marked "not built yet"); its INT8 accuracy is
not measured.

## Not measured yet (Stage II)

- Anything on the UGen300 itself: HEF throughput and latency over USB, GenAI latency on
  hailo-ollama, device power, end-to-end alert latency on the device.
- Rule-parsing and report-action accuracy with the Hailo GenAI build of Qwen2.5-1.5B (a different
  quantization): every LLM figure above was measured with Ollama's qwen2.5:1.5b on the PC GPU.
- The warehouse site detector on Hailo: HEF build, INT8 accuracy, and its compute load at up to 12
  cameras (960 px; Stage II week 1).
- Open-vocabulary detection accuracy (objects named in rules) and the head-size distance estimate in
  hard-hat mode.
- How often face pixelation misses a face (it follows head detections in hard-hat mode and person
  boxes in warehouse mode).
- Warehouse mode on real footage from a real warehouse, with cameras we calibrate ourselves.
- Worker identity across cameras whose views do not overlap (no appearance re-id is used yet).
- Distances to vehicles other than forklifts (their pads are rough guesses).
- Speech-to-text accuracy on recorded spoken rules: the parsing sets above are typed sentences.
- Daily report actions judged by safety officers on a real site's alert log.
