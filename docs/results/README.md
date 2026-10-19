English | [繁體中文](README.zh-TW.md)

# Stage I results

The measurements behind the [README](../../README.md). Everything here was measured on the PC
prototype (NVIDIA GeForce RTX 5060 Ti, 18–19 October 2026) with the same networks that will run
on the Hailo-10H, unless it is marked **reference** (Hailo's published figures) or **estimate**
(derived from them). The raw JSON files sit next to this page, [FACTS.md](../FACTS.md) lists every
published number, and [SETUP.md](../SETUP.md) shows how to re-run them.

## At a glance

| | Result |
|---|---|
| [Hard-hat detector](#hard-hat-detector) | **0.939 mAP50** on 2,001 test images it never saw |
| [Ten cameras, one detector](#ten-cameras-one-detector) | about **105 frames/s** analysed; about **40 %** of one UGen300 (estimate) |
| [Spoken rules](#spoken-rules) | exactly right rule: **95 %** in hard-hat mode, **84 %** and **88 %** on two held-out warehouse sets, against 16–40 % for the LLM alone; mean 347–395 ms per rule on the PC |
| [Alert double-check](#alert-double-check) | **95.3 %** correct on 193 held-out crops in **15.2 ms**; a generative VLM: 91.7 % in 283 ms |
| [Daily report actions](#daily-report-actions) | actions aimed at the wrong problem: **20 % → 0 %** (free text vs menu, two blind AI judges) |
| [Warehouse mode: people](#test-scenes-people-and-positions) | six unseen buildings: head count off by **0.05–1.74** people on average (summing cameras: up to 60.09); mean position error 0.173–0.26 m |
| [Warehouse mode: alerts](#test-scenes-alerts-by-zone-kind) | door and dock zones: **53 of 53** entries covered, **0** false of 52 alerts; forklift 3 m rule: **92 %** covered, 1 false of 26; aisles 83 %, work areas 96 % |
| [Hailo-10H INT8](#hailo-10h-build) | detector **0.889** mAP50 vs 0.8927 in float; double-check **95.3 %** with its head retrained on INT8 features (Hailo emulator) |

## Hard-hat detector

YOLOv8s fine-tuned on Roboflow Hard Hats v2 (CC BY 4.0) with two classes, hard hat and bare head.
The test split is 2,001 images never used in training (4,479 hard hats, 1,038 bare heads),
evaluated with Ultralytics' standard validation at 640 px.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All | 0.9132 | 0.8986 | **0.9393** | 0.5945 |
| Hard hat | 0.9194 | 0.8964 | 0.9447 | 0.5464 |
| Bare head (violation) | 0.9069 | 0.9008 | 0.9338 | 0.6425 |

11.14 M parameters and 28.6 GFLOPs; 2.098 ms per image on the RTX 5060 Ti.
Source: `detector_metrics.json`.

## Ten cameras, one detector

Hard-hat mode on ten looping demo streams at a 10 fps target each. Every frame goes through one
batched YOLOv8s.

| | |
|---|---|
| Frames analysed | about **105 per second** |
| One batch of 10 frames | 31–40 ms on the PC |
| Load on one UGen300 | about **40 %** (estimate: 100 frames/s against Hailo's reference of 252 FPS for YOLOv8s at batch 8) |
| Warehouse mode, 11 cameras | one 11-frame batch at 960 px in about 63–95 ms on the PC |

Figures from the live run; capacity tables in [`benchmark.md`](benchmark.md).

## Spoken rules

The officer's sentence goes to Qwen2.5-1.5B, which drafts the rule as JSON. A checker (a
deterministic rule extractor) reads the same sentence and overrides every number, distance, time
and object it finds stated there. The officer then confirms. A rule counts as correct only if every
stated field is right: type, object, count or distance, time window, duration, and the cameras or
floor zones it applies to. Test sentences are typed, in Mandarin and English.

**Hard-hat mode** (rules over a set of cameras):

| | Dev set (32 sentences, used to build the checker) | **Held-out (20, never used for tuning)** |
|---|---|---|
| LLM alone | 25 % | 40 % |
| Checker alone | 100 % | 90 % |
| **LLM + checker (shipped)** | 96.9 % | **95 %** |

**Warehouse floor plan** (rules over floor zones and the site detector's objects, such as forklift
and pallet truck). The sentences were written and labelled by AI agents that never saw the parser
code.

| | Dev set (15) | Held-out v1, first run (25) | **Held-out v2, fresh (25)** |
|---|---|---|---|
| LLM alone | 20 % | 16 % | **24 %** |
| Checker alone | 100 % | 80 % | 68 % |
| **LLM + checker (shipped)** | 100 % | 84 % | **88 %** |

After the first run of v1 we fixed the parser for the phrasings it missed (zone names such as
"aisle seven" or 「七號走道」, linger verbs, "N or more people"), which brings v1 to 100 %. A new pair
of agents then wrote v2, which was measured once.

**Why both parts are needed.** On v2 the checker alone found no rule in several rephrasings ("goes
above 12", 「碼頭區的人數只要大於十」) that the checked LLM draft got right. In the other direction,
the checker corrects the LLM. Two cases from the live server:

- "Alert me if more than two people are at the site gate after 10 pm": the LLM proposed 22:00–00:00;
  the checker applied its convention that "after 10 pm" runs until 06:00.
- 「A區有人靠近挖土機三公尺內要立刻警告」: the LLM proposed the object "excavation pit"; the checker
  used "excavator".

Three v2 sentences are still wrong: a duration said as 「待了半分鐘」 ("stayed half a minute"),
"six or more people" in aisle seven, and 「晚上八點以後，倉庫裡只要看到人就通報我」 ("after 8 pm,
tell me if anyone is in the warehouse"). This is why every rule is shown to the officer before it
goes live.

**Speed.** Mean 347 ms per rule (hard-hat held-out) to 395 ms (warehouse v2), at most 638 ms, on
the PC GPU. On the UGen300 the rule JSON is estimated at about 11.3 s and Whisper at about 2.4 s
for a 5 s sentence (estimates from Hailo's reference speeds, `benchmark.md`).

Source: `rule_parsing.json`, `rule_parsing_site.json`, `rule_parsing_site_v1_first_run.json`.

## Alert double-check

"Is this person wearing a hard hat?" has two answers, so SayGuard asks for a decision, not a chat.
The design follows the "System One" idea of TypeSafe AI's Jev: return a typed answer instead of
generating text. Jev itself is only a hosted API, so we built a model in the same style from parts
Hailo supports on the Hailo-10H: a frozen CLIP image encoder and a 512→2 decision head. It replaced our first
double-check, the generative Qwen2-VL-2B, and is 18× faster per alert and more accurate.
Four methods on the same 193 held-out crops (94 wearing, 99 bare heads), cut the way an alert crop
is cut:

| Method | Accuracy | Wrong flag filtered | Real violation kept | PC time per crop |
|---|---|---|---|---|
| CLIP ViT-B/32 zero-shot (two text prompts) | 88.6 % | 86.2 % | 90.9 % | 3.49 ms batched |
| **CLIP ViT-B/32 + linear head (shipped)** | **95.3 %** | 95.7 % | 94.9 % | **15.2 ms** one crop (p95 24.7); 3.49 ms batched |
| ResNet18, fine-tuned | 95.3 % | 97.9 % | 92.9 % | 1.7 ms batched |
| Qwen2-VL-2B, generative | 91.7 % | 89.4 % | 93.9 % | 283 ms one crop (p95 531) |

"One crop" is how an alert is checked; "batched" is a 128-crop batch divided by its size, so the
two are not comparable.

We ship the CLIP head rather than ResNet18 (same accuracy) because it keeps more real violations,
and a new yes/no question only needs another 2×512 head on the same CLIP features, not a new
network. The head was trained on 2,950 crops. Since the check drops about 5 % of real violations,
filtered alerts stay in the feed, dimmed, instead of being deleted.

Source: `probe_train.json`, `yesno_check.json`, `vlm_check.json`.

## Daily report actions

Code computes every number, table and summary in the daily report from the alert log. The LLM only
chooses the three recommended actions. We compared three ways of letting it do that:

- **Menu (default):** code lists the (camera, problem) pairs that really happened; the LLM picks
  three and, for each, one of three fixes we vetted. Its reply can only contain list numbers, so it
  cannot name a camera or a problem that did not happen.
- **Structured:** the LLM picks three issues by number and writes the action text; code fills in the
  camera and problem, and replaces text that names anything else with a template.
- **Free text** (the earlier prompt, kept as a baseline): three bullets written from a few facts.

The test used the same 20 synthetic days for all three approaches (English and Chinese
alternating) and Qwen2.5-1.5B. Two AI judges scored each action without knowing which approach
wrote it, given each camera's real problems that day. They agreed on "wrong problem" for 176 of 177
actions.

| Approach | Actions | Wrong problem (both judges) | Wrong problem (either judge) | Sensible (both judges) | Names a camera |
|---|---|---|---|---|---|
| Free text | 60 | 20.0 % | 20.0 % | 45.0 % | 88.3 % |
| Structured | 57 | 3.5 % | 5.3 % | 52.6 % | 100.0 % |
| · text written by the LLM | 43 | 4.7 % | 7.0 % | 37.2 % | 100.0 % |
| · replaced by template text | 14 | 0.0 % | 0.0 % | 100.0 % | 100.0 % |
| **Menu (default)** | 60 | **0.0 %** | **0.0 %** | **100.0 %** | 100.0 % |

For the menu, "sensible" rates the choice of issue and fix, since the wording is our vetted text.

An automatic keyword check (does an action name a camera together with a problem that camera did
not have?) flagged 8 of 60 free-text actions (13.3 %) and none in the other two modes. The judges
still found 3.5–5.3 % wrong-problem actions in structured mode, so a keyword check is not enough
when the 1.5B model writes its own text. That is why the default lets it choose but not write.
Structured mode produced a valid plan on 19 of 20 days, the other two on all 20.

Source: `report_actions_judged.json`; the judges' prompt, blind batches and verdicts are in
[report_actions_judging/](report_actions_judging/README.md).

## Warehouse mode

Warehouse mode fuses every calibrated camera of a warehouse onto its floor plan
([how it works](../SITE_MODE.md)). To measure it, the exact live fusion and rule code runs frame by
frame over synchronized streams from NVIDIA PhysicalAI-SmartSpaces (synthetic warehouses,
CC BY 4.0) and is scored against the dataset's ground truth, which knows every person's floor
position and identity. Each scene is 3,000 frames at 10 fps, about five minutes (W022: 3,090
frames).

### Scenes and settings

The site detector (YOLOv8s, six classes: person, forklift, pallet truck, transporter, mobile robot,
humanoid robot) was fine-tuned at 960 px on 8,000 images from 8 training scenes in 5 buildings,
never on an evaluation scene.

| Role | Scenes | Use |
|---|---|---|
| **Test** | W010, W013, W015, W018, W019, W022 | six buildings the detector never trained on; settings were chosen elsewhere |
| Validation | W012, W020 | settings chosen on W020 |
| Development | W009, W017 | W009 shares building and cameras with training scene W008, W017 with test scene W018; development only |

The zones of every evaluation scene except W010 were drawn by AI agents that saw only a coverage
map, before any result. W010, the demo scene, keeps the demo's zones.

**Settings were chosen on W020 only:** the detection confidence thresholds, a 0.1 m pad on forklift
distances, and "in view" meaning at least 40 px tall at 1280×720 (person detection was 68 % at
20–40 px and 92 % at 40–80 px). Deployment spec: **one UGen300 per area of up to 12 cameras, with
every watched zone seen by at least 2 cameras at ≥ 40 px.**

### What is scored

- **People:** mean head count per frame, compared with the truth and with the sum of every
  camera's person tracks (what a per-camera system counts).
- **Positions:** workers matched to true people within 1 m on the floor: recall, precision, MOTA,
  mean position error, duplicates (share of person-frames with two ids on one person), IDF1, and
  how many ids each real person was split into.
- **Forklifts:** share of forklift frames with the forklift detected; error of the
  worker-to-forklift distance against the true gap (median, MAE, and bias; a positive bias means
  measured farther than the truth).
- **Alerts:** an event is a person within 3 m of a forklift for 2 s, or inside a zone for 3 s, with
  a 60 s cooldown. An alert that matches no event is **false**. **Covered** = the event got its own
  alert, or started within the cooldown of an alert on the same person (no repeat, by design).
  **Strict recall** counts only own alerts. **Alerts per alerted event** of 1.0 means no repeats.
- **Per-camera baseline:** the same detections and floor distances, but each camera alerts on its
  own. The only difference is one identity per worker across cameras.

### Test scenes: people and positions

| Scene | Cameras | True people (mean) | Warehouse mode (error) | Sum of cameras (error) |
|---|---|---|---|---|
| W010 | 11 | 24.97 | **26.56** (1.74) | 51.64 (26.67) |
| W013 | 10 | 15.83 | **15.49** (0.55) | 39.99 (24.16) |
| W015 | 10 | 22.0 | **22.91** (1.05) | 82.09 (60.09) |
| W018 | 3 | 1.0 | **1.05** (0.05) | 2.76 (1.76) |
| W019 | 3 | 2.07 | **1.78** (0.34) | 4.5 (2.45) |
| W022 | 4 | 9.8 | **8.48** (1.35) | 13.51 (3.86) |

Summing cameras counts anyone two cameras see twice: 82.09 for 22.0 people on W015.

| Scene | Recall | Precision | MOTA | Mean position error | Duplicates | IDF1 | Ids per person |
|---|---|---|---|---|---|---|---|
| W010 | 0.928 | 0.873 | 0.788 | 0.209 m | 2.35 % | 0.516 | 10.72 |
| W013 | 0.961 | 0.982 | 0.941 | 0.176 m | 1.55 % | 0.633 | 4.94 |
| W015 | 0.987 | 0.948 | 0.932 | 0.173 m | 3.35 % | 0.701 | 3.64 |
| W018 | 1.0 | 0.955 | 0.953 | 0.231 m | 0.0 % | 0.977 | 1.0 |
| W019 | 0.842 | 0.977 | 0.82 | 0.26 m | 1.45 % | 0.59 | 6.33 |
| W022 | 0.83 | 0.959 | 0.791 | 0.243 m | 1.2 % | 0.287 | 6.58 |

Positions are accurate to 0.173–0.26 m on average. Identities are short-lived, so the dashboard
counts and alerts without showing ids, and the alert logic decides for itself whether a new id is
the same worker.

### Test scenes: forklifts

| Scene | Forklift found (share of forklift frames) | Distance error, median | MAE | Bias |
|---|---|---|---|---|
| W010 | 0.986 | 0.2 m | 0.28 m | -0.12 m |
| W018 | 0.169 | 0.45 m | 0.65 m | -0.02 m |
| W019 | 0.115 | 0.38 m | 0.44 m | -0.08 m |
| W022 | 0.793 | 0.47 m | 0.6 m | 0.43 m |

W013 and W015 have no forklift in view.

### Test scenes: alerts by zone kind

| Kind | Events | Covered | Strict recall | False alerts | Frozen run: covered / false |
|---|---|---|---|---|---|
| Door and dock zones | 53 | **53 (100 %)** | 52 (98 %) | 0 of 52 | 50 (94 %) / 0 of 26 |
| Forklift 3 m rule | 61 | **56 (92 %)** | 25 (41 %) | 1 of 26 | 54 (89 %) / 1 of 24 |
| Narrow lanes and aisles | 118 | **98 (83 %)** | 71 (60 %) | 3 of 77 (4 %) | 76 (64 %) / 1 of 40 |
| Work areas | 338 | **323 (96 %)** | 237 (70 %) | 26 of 362 (7 %) | 281 (83 %) / 16 of 244 |
| Other zones | 48 | 43 (90 %) | 34 (71 %) | 6 of 44 (14 %) | 34 (71 %) / 4 of 30 |

The forklift rule's strict recall is 41 % because its 60 s cooldown deliberately does not re-alert
a worker who steps in and out of 3 m of a parked forklift.

**Frozen run and final run.** The frozen run used the settings as they stood before any test
result. The final run, which is the current code, adds two alert-logic fixes, developed on
validation and development scenes only:

1. **Same worker or someone else.** A new id near a worker alerted within the cooldown counts as
   that worker only if it stands right on them, or appears where they vanished within a few seconds.
   Anyone else, such as the next worker down the same aisle, gets their own alert.
2. **Re-entry.** Coming back into a zone after 10 s outside is a new entry and alerts again.

Coverage rose for every kind, and alerts and false alerts rose with it. The detector and the
confidence settings did not change.

### Test scenes: alerts per rule

With the same detections, the per-camera baseline covers far fewer zone events (W010 dock: 0.364
against 1.0; W013 south_open_floor: 0.028 against 0.972) and raises false alerts in zones nobody
entered (W019 dock_approach_floor: 6). Warehouse mode does not win everywhere: on the forklift rule
the baseline matches it on W018 and W019, and on W022 north_staging warehouse mode is lower (0.5
against 0.75).

<details>
<summary>All rules and zones on the six test scenes</summary>

Covered is shown as own alert + in cooldown = share of events.

| Scene | Rule / zone | Kind | Events | Covered | Strict recall | Alerts (false) | Alerts per alerted event | Per-camera: covered | Per-camera: alerts (false) |
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

</details>

### Validation and development scenes

These scenes are kept out of the headline numbers, but they show where the method strains. W020 is the hardest: 16
cameras (outside the deployment spec), the most people (49.02 on average) and the same building as
the training scenes. MOTA falls to 0.168 there, and 16 of 51 forklift alerts are false.

<details>
<summary>Full tables for W012, W020, W009 and W017</summary>

| Scene | Role | Cameras | True people | Warehouse mode (error) | Sum of cameras (error) | Recall | Precision | MOTA | Position error | Duplicates | IDF1 | Ids per person |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W012 | validation | 8 | 15.0 | 15.05 (0.31) | 30.78 (15.78) | 0.976 | 0.973 | 0.947 | 0.199 m | 2.43 % | 0.651 | 5.53 |
| W020 | validation | 16 | 49.02 | 42.1 (7.19) | 97.13 (48.11) | 0.519 | 0.605 | 0.168 | 0.348 m | 2.63 % | 0.158 | 22.32 |
| W009 | development | 12 | 13.0 | 13.99 (1.1) | 56.8 (43.8) | 0.977 | 0.909 | 0.874 | 0.174 m | 7.56 % | 0.485 | 10.0 |
| W017 | development | 3 | 1.75 | 1.17 (0.63) | 3.87 (2.12) | 0.617 | 0.924 | 0.554 | 0.237 m | 0.32 % | 0.7 | 8.0 |

Forklifts: W020 found in 0.802 of forklift frames (distance error median 0.32 m, MAE 0.52 m, bias
0.01 m); W017 0.331 (median 0.3 m, MAE 0.46 m, bias 0.06 m); W012 and W009 have no forklift in view.

| Scene | Rule / zone | Events | Covered | Strict recall | Alerts (false) | Alerts per alerted event | Per-camera: covered | Per-camera: alerts (false) |
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

</details>

Source: `site_eval*.json` (final run), `frozen/` (frozen run), `site_by_zone_kind.json`,
`w020_settings_analysis.json`.

## Hailo-10H build

With no device yet, Stage I took both hard-hat-mode vision networks through Hailo's real toolchain
(Dataflow Compiler 5.4.0 with Hailo Model Zoo recipes) and measured the cost of INT8 in the
compiler's emulator, which reproduces the chip's integer arithmetic bit for bit. The hard-hat
detector is compiled to a HEF (`helmet_yolov8s.hef`, highest optimization level); the CLIP image
encoder of the double-check is quantized and checked in the emulator. The compiler does not support
RTX 50-series GPUs, so everything ran on the CPU, without the GPU-only fine-tuning steps of
quantization.

**Hard-hat detector**, on the same 2,001 test images, scored with pycocotools using the NMS settings
built into the HEF (score ≥ 0.2, IoU 0.7):

| Model | mAP50 | mAP50-95 | Hard hat AP50 | Bare head AP50 |
|---|---|---|---|---|
| PC, PyTorch fp32 | 0.8927 | 0.5663 | 0.9035 | 0.8819 |
| Hailo graph, float | 0.8927 | 0.5662 | 0.9035 | 0.8819 |
| **Hailo INT8 (what the chip computes)** | **0.889** | 0.56 | 0.9008 | 0.8772 |

The float Hailo graph matches PyTorch, so the conversion is exact; INT8 costs 0.0037 mAP50. The
numbers here are lower than in the first table because the HEF's built-in NMS drops boxes scoring
below 0.2: compare 0.889 with 0.8927, not with 0.9393.

How the calibration images (from the validation split) are used matters:

| Calibration | mAP50 | mAP50-95 | Bare head AP50 |
|---|---|---|---|
| 64 images | 0.8911 | 0.5396 | 0.8841 |
| 512 images | 0.8513 | 0.5363 | 0.857 |
| **512 images, activations clipped to the 0.01–99.99 percentiles (shipped)** | 0.889 | **0.56** | 0.8772 |

With clipping, box placement (mAP50-95) stays closest to float. Hailo's reference speed for the same
YOLOv8s on Hailo-10H is 166 FPS at batch 1 and 252 FPS at batch 8 (Hailo Model Zoo, **reference**).

**CLIP image encoder for the double-check:** the Model Zoo's `clip_vit_b_32_image_encoder`,
calibrated on 64 site crops and tested on the same 193 crops as above.

| Features → head | Accuracy | Wrong flag filtered | Real violation kept |
|---|---|---|---|
| PC (PyTorch) → PC head (shipped today) | 95.3 % | 95.7 % | 94.9 % |
| Hailo INT8 → the same PC head | 94.3 % | 95.7 % | 92.9 % |
| **Hailo INT8 → head retrained on INT8 features** | **95.3 %** | 96.8 % | 93.9 % |

INT8 features keep a cosine similarity of 0.9634 to PyTorch's (float Hailo graph: 0.9999). Reusing
the PC head loses a point; retraining the small head on the chip's own features recovers it, so
Stage II retrains it on device features.

Source: `hailo_int8.json`, `pc_baseline.json`, `hailo_int8_c512.json`,
`helmet_c64_optimized_int8.json`, `clip_probe_int8.json`; the build flow is in
[HAILO_PORT.md](../HAILO_PORT.md).

## Limits and what is not measured

**Limits of these results**

- **PC only.** Nothing has run on a UGen300 yet. UGen300 load and latency figures are estimates
  from Hailo's reference speeds. All LLM accuracy was measured with Ollama's qwen2.5:1.5b on the PC
  GPU, not with the Hailo GenAI build, which is quantized differently.
- **Test inputs.** Rule sentences are typed, not spoken. The double-check's 94 hard-hat crops are
  labelled dataset crops standing in for the detector's real false flags. The report test uses 20
  synthetic days, scored by AI judges rather than safety officers.
- **Synthetic warehouses.** No real warehouse has been measured yet; real cameras need their own
  calibration.
- **Not every test scene is independent.** W010 is the demo scene: the tracker was developed on it
  and it keeps the demo's zones. W017, used for alert-logic work, has the same building and
  cameras as test scene W018. The two alert-logic fixes came after the frozen run, so both runs
  are reported. The INT8 calibration recipe was picked from three variants scored on the test
  split.
- **Forklift detection** was weak on W018 and W019 (found in 0.169 and 0.115 of forklift frames).
  The detector was not retrained on test scenes; a short fine-tune on a customer's own vehicles is
  planned.
- **Short-lived identities.** Up to 10.72 ids per person on the test scenes (on W010 about one
  every 28 s). Alerts are per worker within a cooldown, and ids are never shown.
- **Distances.** On W022 forklift distances came out 0.43 m too far on average, which is the unsafe
  direction. Distances to other vehicles are not measured.
- **Busy work areas** get repeat and false alerts (W010 racks: 13 false of 56; W010 open_floor: 2.93
  alerts per alerted event).
- **Outside the spec.** W020 (16 cameras) reaches MOTA 0.168.
- One run per scene, no confidence intervals.

**Not measured yet (Stage II)**

- On the UGen300: HEF throughput and latency over USB, hailo-ollama latency, power and end-to-end
  alert latency.
- The warehouse detector on Hailo: HEF build, INT8 accuracy and load at 12 cameras (the 12-camera
  limit is a coverage spec, not a measured compute limit).
- Speech-to-text accuracy on recorded speech.
- Open-vocabulary detection and the head-size distance estimate in hard-hat mode; how often face
  pixelation misses a face.
- Worker identity across cameras whose views do not overlap.
- Report actions judged by safety officers on a real site's alert log.
