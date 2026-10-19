English | [繁體中文](SITE_MODE.zh-TW.md)

# Warehouse mode (site mode)

Hard-hat mode treats every camera on its own. Warehouse mode (called *site mode* in the code)
treats all the cameras of one area as **one place**: each camera is calibrated, so every person
box becomes a position on the floor plan in metres. The same worker seen by four cameras is
counted once, not once per camera, and raises one alert per entry, not one per camera; zones are
drawn once on the plan instead of per camera; and "within 3 m of a forklift" is measured on the
floor, not guessed from head size.

The demo runs on 11 synchronized, calibrated cameras of NVIDIA PhysicalAI-SmartSpaces
`Warehouse_010` (synthetic, CC BY 4.0). It was measured on six test warehouses the site detector
was never trained on, against NVIDIA's ground truth. W010 is also the demo scene, so it is not a
blind test (see [Limits](#limits-what-is-not-measured)). Everything below runs on the PC prototype
(RTX 5060 Ti). The site detector has not been compiled for the UGen300 yet.

- Config: [`configs/warehouse.yaml`](../configs/warehouse.yaml)
- Code: [`sayguard/site.py`](../sayguard/site.py) (calibration, floor plan, zones),
  [`sayguard/sitetrack.py`](../sayguard/sitetrack.py) (one identity per worker, vehicles),
  [`sayguard/site_pipeline.py`](../sayguard/site_pipeline.py) (the live loop and the fusion shared with
  the evaluation), [`sayguard/site_view.py`](../sayguard/site_view.py) (camera tiles and live floor plan),
  `RuleEngine.evaluate_site` in [`sayguard/rules/engine.py`](../sayguard/rules/engine.py) (rules per worker)
- Evaluation: [`scripts/eval_site.py`](../scripts/eval_site.py), scene configs in
  [`configs/eval/`](../configs/eval/), results in [`docs/results/`](results/) (`site_eval*.json`; the
  frozen run in `docs/results/frozen/`); every number on this page is copied from
  [`docs/FACTS.md`](FACTS.md)

## Run it

```bash
python scripts/fetch_smartspaces.py train/Warehouse_010     # calibration, floor plan, ground truth, 11 videos
python scripts/make_site_streams.py train/Warehouse_010     # frame-locked 1280 px, 10 fps streams (needs ffmpeg + ffprobe)
python -m sayguard serve --config configs/warehouse.yaml    # http://127.0.0.1:8000
```

You also need the site detector weights at `@work/models/site_yolov8s.pt` (see
[The site detector](#the-site-detector)) and, for spoken rules, an Ollama-compatible server with
`qwen2.5:1.5b` and Whisper-Small, as in hard-hat mode. `@work` is the work directory outside the
repo (`SAYGUARD_WORK`). `start_demo_warehouse.cmd` is the author's Windows launcher for the same
command. Its paths are hard-coded.

## How it works

Each tick, frames from every camera go through one batched detector call. Each camera tracks its
own person boxes. Then:

### 1. Calibrated cameras and the floor plan

SmartSpaces ships a `calibration.json` with, for every camera, a 3×3 homography from the floor
(world z = 0, metres) to image pixels, a 3×4 camera matrix, and the floor-plan transform (plan pixel
= (world + translation) × scale, y axis flipped). `Site.load` reads it. Streams are downscaled from
the calibrated resolution, so every point is rescaled before it is projected.

### 2. From a box to a floor position: feet, head and the segment between

- **Feet point**: the bottom centre of the box, projected through the inverse homography.
- **Head point**: the top centre of the box, projected onto the plane at standing head height
  (`HEAD_TOP_M`) with the 3×4 camera matrix, then dropped to the floor.
- A box cut by a shelf, a cart or the frame edge does not end at the feet. Its bottom then
  projects far beyond the worker along the camera ray. So each view carries the **floor segment**
  from its head point to its feet point: the worker is somewhere on it. If a box touches the
  bottom edge of the frame, its feet are treated as hidden and the head point is used.
  Segments longer than `MAX_SEGMENT_M` are dropped (near the horizon the projection blows up).

### 3. Per-view uncertainty

Every view gets `sigma`, its floor uncertainty in metres: the floor distance that one pixel spans
at the point used, times the box's localisation error (a few pixels plus a share of the box height),
clipped to 0.15–5 m. A worker 40 px tall at the far end of a camera's view can be metres
uncertain. A well-seen worker near a camera sits at the bottom of that range. Match radii grow with
`sigma`, and positions are weighted by it.

### 4. Fusion: one worker across cameras

`SiteTracker` turns camera tracks into workers on the floor. No appearance model is involved: the
identity is geometric.

- **Position**: a weighted median of the views whose feet are visible. Then each view moves to the
  closest point of its own segment, and the weighted median is taken again. One view with hidden
  feet cannot drag a worker metres away.
- **Sticky links**: a camera track keeps its worker unless it keeps disagreeing with the other
  cameras (a wrong join, or two people who walked together and parted).
- **Join**: a new camera track joins the nearest worker within reach of its segment that no other
  track of the same camera already represents. A track-id switch, or a worker walking into the
  next camera's view, keeps the identity.
- **Spawn**: leftover tracks start new workers, pairing across cameras. A track that only one
  camera sees must persist a few ticks first, and longer when its position is uncertain. A
  flickering false detection does not become a worker.
- **Merge**: two workers that stay close with no camera in common are one person. If a camera
  recently saw them as two people standing apart, the merge waits much longer. The older id survives.
- **Hidden workers**: a worker nobody sees keeps its id for a few seconds. The pick-up radius grows
  at walking pace, so walking behind a rack does not make a new person. Ids are never reused.

The head count on the dashboard is the number of workers, next to "cameras count N": the sum of
every camera's own person tracks, which counts a person once per camera that sees them.

### 5. Confidence filter

The detector runs at confidence 0.25. A worker is **shown** (on the map, in the head count, to the
rules) once some camera sees it with confidence ≥ 0.7, and **kept** while one still sees it with
≥ 0.6 (`SHOW_CONF`, `KEEP_CONF`). Fainter tracks keep their id but stay hidden. Vehicles and robots
need ≥ 0.6 (`OBJECT_CONF`). All three values were chosen on the validation scene (W020), never on a
test scene (`docs/results/w020_settings_analysis.json`). On W020 the worker filter raised the
precision of worker positions from 0.459 (no filter) to 0.603, while recall fell from 0.551 to
0.521. Of the forklift boxes below 0.6, 2,078 of 6,284 were false, mostly pallet trucks and
background; at 0.6 or more, 119 of 11,103 were.

### 6. Vehicles from ground contacts

The rules need no vehicle identity, only "is one near this worker now". For every vehicle box that
does not touch the bottom edge, its three ground contacts (both bottom corners and the bottom
centre) are projected to the floor, then clustered across cameras by class. The distance to a
worker is the distance to the nearest ground contact any camera saw, minus a pad for the part of
the body that reaches past it. The forklift pad (0.1 m) was chosen on W020, where it brought the
measured distance bias from -0.38 m to 0.01 m. The pads for pallet trucks, transporters and mobile
robots are rough guesses, not measured. Robots are their own detector classes, so they are never
counted as workers.

### 7. Rules on floor zones

Zones are polygons on the floor plan (`plan_polygon` in `map.png` pixels, or `polygon` in metres)
under `zones:` in the config, more specific zones first. A worker is in every zone that contains
their floor position, plus the whole-site zone (`site.zone`, here `warehouse`). Rules are
evaluated per worker, not per camera (`RuleEngine.evaluate_site`):

| Rule type | In warehouse mode |
|---|---|
| `person_present` | per worker inside the rule's zones |
| `person_count_over` | workers counted once across cameras, inside the rule's zones |
| `object_near_person` | metres on the floor from each worker to the nearest vehicle of that class |
| `object_present` | an object of that class on the floor inside the rule's zones |
| `no_helmet` | refused: the site detector sees whole people, not hard hats |

A rule can name any class the site detector knows (forklift, pallet truck, transporter, mobile
robot, humanoid robot). Any other object a rule names falls back to the open-vocabulary detector
(YOLO-World), which has not been measured in this mode. Which floor zones each camera sees is
worked out once, from the calibration. It is used only to check a drafted rule's scope ("no camera
covers aisle 7") and for the dashboard's coverage preview. Evaluation uses worker positions.

The two default rules in `configs/warehouse.yaml` are "Keep 3 m from forklifts" (whole warehouse,
2 s) and "No walking in aisle 7 (forklift parked)" (3 s), each with a 60 s cooldown.

### 8. Alert logic: one alert per worker per entry

- The condition must hold for the rule's `min_duration_s`. Short detection gaps do not reset the
  timer.
- Each worker is one subject (`g<id>`), however many cameras see them, so a rule fires once per
  person, not once per camera. The rule's `cooldown_s` then spaces reminders for that worker.
- Within tens of seconds, one person can go through several fused ids (10.72 ids per person over
  the 5-minute W010 clip). So when a new id appears at an alerted worker's spot, the engine
  decides whether it is the same person. If the alerted worker's own id is **still tracked**, the
  new id is them only when it stands right on them (a second id on one person). If that id is
  **gone**, the new id is them when it appears near where they were alerted within 10 s
  (`SITE_REAPPEAR_S`), or near where they were last seen within 3 s (`SITE_HANDOVER_S`, radius
  growing at walking pace). Anyone else, such as the next worker down the same aisle, gets their
  own alert.
- A new id at the same spot also keeps the running duration of the id it replaces, so an id
  switch does not restart the timer.
- **Re-entry**: presence rules alert again when a worker comes back into the zone after at least
  10 s outside (`SITE_REENTRY_S`), cooldown or not. The cooldown spaces reminders while someone
  stays; it must not carry over to whoever an id is on next.
- Site-wide rules (count, object present) are one subject for the whole site. Their alert goes to
  the camera that sees most of what triggered them.

### 9. What the dashboard shows, and privacy

- The live floor plan (`/stream/map`) shows zones, camera positions, workers as dots (red while in
  an alert), vehicles, and a red line with the distance in metres for an active proximity alert.
  Camera tiles draw a worker in an alert in red in every camera that sees them.
- **Worker ids are not shown.** They are reliable at any one moment but change within tens of
  seconds (10.72 ids per person over the 5-minute W010 clip, IDF1 0.516), so a number on screen
  would mislead. Alert records keep the internal id, the cameras and the floor position for the log.
- **Faces are pixelated** in every detected person box, in every tile and snapshot. The site
  detector has no head boxes, so the face region is estimated from the top of each person box.
  Every person box the detector returned is blurred, including boxes the tracker left out. How
  often this estimate misses a face (for example, a worker bending down) is not measured.
- Everything runs locally on site (today the PC prototype; the UGen300 in Stage II). Frames and
  alerts are not sent to any cloud service.

### Parameters

Values as set in the code. The first four were chosen by measurement on W020. The alert-logic
values were developed on the development and validation scenes. The rest are design values.

| Parameter | Value | Where | How it was set |
|---|---|---|---|
| Worker shown / kept | confidence ≥ 0.7 / ≥ 0.6 | `SHOW_CONF`, `KEEP_CONF` (site_pipeline.py) | chosen on W020 |
| Vehicles and robots | confidence ≥ 0.6 | `OBJECT_CONF` | chosen on W020 |
| Forklift distance pad | 0.1 m | `VEHICLE_PAD_M` (sitetrack.py) | chosen on W020 |
| "In view" for scoring and for the deployment spec | ≥ 40 px tall at 1280×720 | `IN_VIEW_PX` (eval_site.py) | chosen on W020 |
| Head height for head points | 1.75 m | `HEAD_TOP_M` (site.py) | design value |
| Match radius / merge distance | 1.2 m (+ sigma) / 0.7 m | `MATCH_M`, `MERGE_M` | design values |
| Hidden worker kept | 6 s, pick-up radius +1 m/s up to 3 m | `LOST_S`, `REJOIN_SPEED`, `REJOIN_MAX` | design values |
| Same worker after id change | 1.5 m; re-appear 10 s; hand-over 3 s | `SITE_SAME_PLACE_M`, `SITE_REAPPEAR_S`, `SITE_HANDOVER_S` (engine.py) | developed on the development and validation scenes |
| Re-entry for presence rules | 10 s outside | `SITE_REENTRY_S` | developed on the development and validation scenes |
| Other vehicle pads | 0.3–0.4 m | `VEHICLE_PAD_M` | rough guesses, not measured |

## The site detector

Off-the-shelf detectors did not work on the synthetic warehouse. On Warehouse_010, YOLO-World
found 0 of 30 forklift boxes. COCO YOLOv8s matched 7 as "truck" and took 39 % of the humanoid
robots for people. So SayGuard trains its own detector: **YOLOv8s, 6 classes** (person,
forklift, pallet truck, transporter, mobile robot, humanoid robot). It was fine-tuned at 960 px on
8,000 images from 8 SmartSpaces training scenes in 5 synthetic buildings (W000, 002, 004, 006,
008, 011, 014, and W016 cameras 0-8; W000, 002, 004 and 006 are one building). It was never
trained on an evaluation scene. `scripts/prepare_site_dataset.py` refuses W010,
W018 and W020 outright. The other evaluation scenes are simply not in the training lists. Live,
one 11-frame batch at 960 px takes about 63-95 ms on the RTX 5060 Ti. Weights:
`@work/models/site_yolov8s.pt`. The Hailo HEF (`site_yolov8s.hef`) is not built yet.

## Spoken rules for the warehouse

Rules are spoken in Mandarin or English, as in hard-hat mode, and the same principle holds: **the
LLM proposes, code checks, the officer confirms**. Qwen2.5-1.5B drafts the rule as schema-bound
JSON. The deterministic rule extractor overrides the type, object, number, distance or time it
finds stated in the sentence (it can miss a phrasing). A zone named without a camera covers every
camera, because a zone is a region of the floor, not a camera. The officer confirms the draft
before it goes live. Fixes for the warehouse: zone names with numerals or number words, an
optional 區, distance phrasing, linger verbs, "N or more", and "AMR" next to Chinese text.

The sentences (`tests/site_rule_sentences.json`) were written and labelled by AI agents (writers
and reviewers) that never saw the parser code. They are typed sentences: speech recognition is not
part of these numbers, and its accuracy is not measured. A sentence counts only if every labelled
field is right. All runs used Ollama `qwen2.5:1.5b` on the PC GPU; the Hailo GenAI build for the
UGen300 is not measured.

| Set | Sentences | LLM alone | Rule extractor alone | LLM + check | Latency (mean / max) |
|---|---|---|---|---|---|
| dev (may be used to fix the parser) | 15 | 0.2 | 1.0 | **1.0** | 363 / 457 ms |
| held-out v1, first run (before the fixes it exposed) | 25 | 0.16 | 0.8 | **0.84** | – |
| held-out v1 after those fixes (seen since, no longer clean) | 25 | 0.16 | 0.96 | 1.0 | 392 / 515 ms |
| held-out v2 (fresh, written blind, never used for tuning) | 25 | 0.24 | 0.68 | **0.88** | 395 / 638 ms |

The honest figure is the fresh set: 88 % exactly right, against 24 % for the LLM alone. Offline
(no LLM), the rule extractor alone gets 68 % of the fresh set right.

## Evaluation protocol

### Scenes and roles

Some SmartSpaces scenes are **same-camera twins** (000/001, 002/003, 004/005, 006/007, 008/009,
017/018, 020/021): the same building and the same cameras, byte-identical `calibration.json` and
`map.png`. Buildings were told apart by their floor plan (`map.png`): one building hosts
W000–W007 and W020/021 (five camera sets), and every other twin pair or single scene has a plan
of its own. No test scene shares a floor plan with a training scene; W018 shares one only with
development scene W017. The scenes have these roles:

| Role | Scenes | Use |
|---|---|---|
| **Test** | W010, W013, W015, W018, W019, W022 | the detector was never trained on them and the measured settings were not chosen on them; floor plans distinct from the training scenes. Caveats: W010 is the demo scene (tracker developed watching it, demo zones); W018's twin W017 was used to develop the alert logic |
| Validation | W012, W020 | the measured settings were chosen on W020 only. W020 is the building of training scenes W000-W007 (other cameras), and its 16 cameras are outside the 12-camera spec |
| Development | W009 (twin of training scene W008), W017 (twin of test scene W018) | development only |
| Detector training | W000, W002, W004, W006, W008, W011, W014, W016 | – |

The detector could not be retrained for the small warehouses without burning a test scene. Every
remaining scene is in a building already in use: W009 is the twin of training scene W008, W017 the
twin of test scene W018, and W001, W003, W005, W007 and W021 are in the training building. None of
them is a new building.

### What is measured

`scripts/eval_site.py` runs the live fusion (`SiteFusion` and `RuleEngine.evaluate_site`, the same
code as the dashboard) frame by frame over the whole synchronized clip (3000 frames at 10 fps; W022
3090). It scores the result against NVIDIA's ground truth, which knows every person's floor
position and gives them the same id in every camera. The ground truth never enters the pipeline.

- **Head count**: shown workers per frame against the true number of people visible in at least
  one configured camera, and against the camera sum. Reported as means and mean absolute errors.
- **Positions and identity** (`motmetrics`, match radius 1 m): recall, precision, MOTA, mean
  position error in metres, IDF1. *Duplicates* is the share of person-frames with two worker ids on
  one person at once (many-to-one matching, which one-to-one MOT matching hides). *Ids per person*
  is how many site ids each real person was split into over the clip.
- **Events**: a ground-truth event is a person within 3 m of a forklift's 3D footprint for the
  rule's 2 s, or a person inside a zone for the rule's duration. Gaps under 1 s do not split an
  event. With `--all-zones`, every zone without a presence rule also gets one ("Zone check: <zone>",
  3 s, 60 s cooldown).
- **Alerts**: an alert belongs to an event of the same rule if its worker is within 2.5 m of that
  event's person at alert time. An alert that matches no event is **false**. **Strict recall**
  counts events with an alert of their own. **Covered** also counts events whose person had already
  been alerted for that rule within the cooldown before the event started: by design they get no
  new alert. *Per alerted event* is alerts on events divided by events alerted (1.0 = one alert
  each).
- **Per-camera baseline**: the same detections and the same floor distances, but each camera
  alerts on its own tracks with the duplicate guards SayGuard used before (same-place IoU, a 15 s
  gap per camera, one presence subject per camera).
- **Forklifts**: *found* is the share of forklift-frames (a forklift visible in a configured
  camera) in which a detected forklift's ground contacts lie within 2.5 m of its footprint. *Distance error* is
  measured worker-to-forklift distance minus the ground-truth gap to the 3D footprint, for people
  within 6 m of a forklift that was detected.
- The JSON files also hold `alerts_in_view` (only people at least 40 px tall in some camera). The
  tables below use `alerts`, which counts every event.

### Zones

The zones in `configs/eval/*.yaml` were drawn on coverage maps, where at least two cameras see a
standing person at least 40 px tall. They were drawn by reviewers (separate AI agents) who had not
seen any result. The agent that drew a scene's zones also gave each zone a kind, from the coverage
map only: door/dock, lane (narrow lanes and aisles), work area, or other. The kinds are recorded
in `docs/results/site_by_zone_kind.json`, not in the eval configs. W010 keeps the demo's own zones
from `configs/warehouse.yaml`, so its zones were not drawn blind; its kinds were assigned by us:
aisle 7 lane, dock door, racks and open floor work.

The agents' kinds are not fully consistent. In W015, `west_rack_aisle` is a work area but
`east_rack_aisle` is "other"; in W022, `picking_aisle_r2_r3` is a work area but `main_aisle_r1_r2`
is a lane. We kept the kinds as the agents labelled them. The kinds file was written after the
results and the eval configs do not record kinds, so the repository holds no record of the kinds
from before the results. The "other" row is `east_rack_aisle` alone (W013's `west_rack_line` had
no events), and it is shown in every table below, with the highest false-alert rate of any kind.

### Frozen run, then final run

First, a **frozen run**: the shipped detector and confidence settings on every scene, reported
as-is (`docs/results/frozen/`). Then, after the frozen results had been seen, two alert-logic
changes were developed on the development and validation scenes (W009, W012, W017, W020), never on
a test scene:

1. The same-worker guard became presence-aware. A new id counts as an alerted worker only while
   that worker's own id is gone (re-appear 10 s / hand-over 3 s), or when it stands right on them.
2. Presence rules alert again on a new entry after 10 s outside.

The test scenes were then re-run once (`docs/results/`). The detector and confidence settings are
the same in both runs. The headline numbers below are from this final run; the frozen run is shown
next to them.

## Results

### Test scenes, by zone kind (six test warehouses)

**Covered** = the event got an alert of its own, or happened inside the cooldown of an alert for
the same person. **Strict recall** counts only events with an alert of their own. Counts from
`docs/results/site_by_zone_kind.json`.

Final run:

| Kind | Events covered | Strict recall | False alerts |
|---|---|---|---|
| Door / dock zones | 53/53 (100 %) | 52/53 (98 %) | 0 of 52 |
| Forklift 3 m rule | 56/61 (92 %) | 25/61 (41 %) | 1 of 26 |
| Narrow lanes / aisles | 98/118 (83 %) | 71/118 (60 %) | 3 of 77 (4 %) |
| Work areas | 323/338 (96 %) | 237/338 (70 %) | 26 of 362 (7 %) |
| Other zones | 43/48 (90 %) | 34/48 (71 %) | 6 of 44 (14 %) |

Frozen run (before the two alert-logic changes):

| Kind | Events covered | Strict recall | False alerts |
|---|---|---|---|
| Door / dock zones | 50/53 (94 %) | 26/53 (49 %) | 0 of 26 |
| Forklift 3 m rule | 54/61 (89 %) | 23/61 (38 %) | 1 of 24 |
| Narrow lanes / aisles | 76/118 (64 %) | 39/118 (33 %) | 1 of 40 |
| Work areas | 281/338 (83 %) | 147/338 (43 %) | 16 of 244 |
| Other zones | 34/48 (71 %) | 21/48 (44 %) | 4 of 30 |

- The changes raised coverage and strict recall, and also the number of alerts and false alerts:
  work areas 16 of 244 → 26 of 362 false, lanes 1 of 40 → 3 of 77, other 4 of 30 → 6 of 44; door
  alerts doubled from 26 to 52, with none false.
- The forklift rule's strict recall is 41 % because its 60 s cooldown deliberately does not
  re-alert a worker who steps in and out of 3 m of a parked forklift. Those events count as covered,
  not as alerted.
- Two test scenes are not fully independent: W010 is the demo scene (see [Limits](#limits-what-is-not-measured)),
  and W017, used to develop the alert logic, is W018's twin.

### Head count

Site mode counts each worker once. The camera sum counts a worker once per camera that sees them.

| Scene | Role | Cameras | True people (mean) | Site mode (abs. error) | Camera sum (abs. error) |
|---|---|---|---|---|---|
| W010 | test | 11 | 24.97 | **26.56** (1.74) | 51.64 (26.67) |
| W013 | test | 10 | 15.83 | **15.49** (0.55) | 39.99 (24.16) |
| W015 | test | 10 | 22.0 | **22.91** (1.05) | 82.09 (60.09) |
| W018 | test | 3 | 1.0 | **1.05** (0.05) | 2.76 (1.76) |
| W019 | test | 3 | 2.07 | **1.78** (0.34) | 4.5 (2.45) |
| W022 | test | 4 | 9.8 | **8.48** (1.35) | 13.51 (3.86) |
| W012 | validation | 8 | 15.0 | 15.05 (0.31) | 30.78 (15.78) |
| W020 | validation, out of spec | 16 | 49.02 | 42.1 (7.19) | 97.13 (48.11) |
| W009 | development | 12 | 13.0 | 13.99 (1.1) | 56.8 (43.8) |
| W017 | development | 3 | 1.75 | 1.17 (0.63) | 3.87 (2.12) |

### Positions and identity

| Scene | Recall | Precision | MOTA | Mean position error | Duplicates | IDF1 | Ids per person |
|---|---|---|---|---|---|---|---|
| W010 | 0.928 | 0.873 | 0.788 | 0.209 m | 2.35 % | 0.516 | 10.72 |
| W013 | 0.961 | 0.982 | 0.941 | 0.176 m | 1.55 % | 0.633 | 4.94 |
| W015 | 0.987 | 0.948 | 0.932 | 0.173 m | 3.35 % | 0.701 | 3.64 |
| W018 | 1.0 | 0.955 | 0.953 | 0.231 m | 0.0 % | 0.977 | 1.0 |
| W019 | 0.842 | 0.977 | 0.82 | 0.26 m | 1.45 % | 0.59 | 6.33 |
| W022 | 0.83 | 0.959 | 0.791 | 0.243 m | 1.2 % | 0.287 | 6.58 |
| W012 | 0.976 | 0.973 | 0.947 | 0.199 m | 2.43 % | 0.651 | 5.53 |
| W020 | 0.519 | 0.605 | 0.168 | 0.348 m | 2.63 % | 0.158 | 22.32 |
| W009 | 0.977 | 0.909 | 0.874 | 0.174 m | 7.56 % | 0.485 | 10.0 |
| W017 | 0.617 | 0.924 | 0.554 | 0.237 m | 0.32 % | 0.7 | 8.0 |

Mean position errors on the test scenes are 0.173–0.26 m. Identities change within tens of seconds
(ids per person above 1 on every scene but W018; 10.72 over the 5-minute W010 clip), which is why
the dashboard does not show them.

### Forklifts

| Scene | Role | Forklift found (share of forklift-frames) | Distance error: median / MAE / bias |
|---|---|---|---|
| W010 | test | 0.986 | 0.2 / 0.28 / -0.12 m |
| W018 | test | 0.169 | 0.45 / 0.65 / -0.02 m |
| W019 | test | 0.115 | 0.38 / 0.44 / -0.08 m |
| W022 | test | 0.793 | 0.47 / 0.6 / 0.43 m |
| W020 | validation | 0.802 | 0.32 / 0.52 / 0.01 m |
| W017 | development | 0.331 | 0.3 / 0.46 / 0.06 m |

W013, W015, W012 and W009 have no forklift in view of their cameras. Across the test scenes with
forklifts, the share found ranges from 0.115 (W019) to 0.986 (W010); the causes of the misses are
not established.

### Alerts per zone, test scenes

Site mode on the left. The per-camera baseline uses the same detections on the right.

| Scene | Rule / zone | Kind | Events | Covered (own + in cooldown) | Strict recall | Alerts | False | Per alerted event | Per-camera: covered / alerts / false |
|---|---|---|---|---|---|---|---|---|---|
| W010 | Keep 3 m from forklifts | forklift | 42 | 18+22 = 0.952 | 0.429 | 18 | 0 | 1.0 | 0.905 / 23 / 0 |
| W010 | No walking in aisle 7 | lane | 9 | 8+1 = 1.0 | 0.889 | 8 | 0 | 1.0 | 0.889 / 10 / 0 |
| W010 | dock | door | 22 | 21+1 = 1.0 | 0.955 | 21 | 0 | 1.0 | 0.364 / 14 / 0 |
| W010 | racks | work | 43 | 38+5 = 1.0 | 0.884 | 56 | 13 | 1.13 | 0.186 / 31 / 0 |
| W010 | open_floor | work | 48 | 42+6 = 1.0 | 0.875 | 130 | 7 | 2.93 | 0.021 / 45 / 0 |
| W013 | west_cross_aisle | lane | 17 | 11+5 = 0.941 | 0.647 | 12 | 0 | 1.09 | 0.529 / 34 / 0 |
| W013 | north_staging_area | work | 11 | 9+1 = 0.909 | 0.818 | 11 | 2 | 1.0 | 0.455 / 25 / 5 |
| W013 | south_open_floor | work | 36 | 26+9 = 0.972 | 0.722 | 29 | 0 | 1.12 | 0.028 / 29 / 0 |
| W013 | west_rack_line | other | 0 | – | – | 0 | 0 | – | – / 0 / 0 |
| W015 | north_cross_aisle | lane | 40 | 27+9 = 0.9 | 0.675 | 27 | 0 | 1.0 | 0.25 / 22 / 0 |
| W015 | south_staging_floor | work | 120 | 56+54 = 0.917 | 0.467 | 59 | 3 | 1.0 | 0.075 / 40 / 0 |
| W015 | west_rack_aisle | work | 46 | 38+7 = 0.978 | 0.826 | 48 | 0 | 1.26 | 0.022 / 38 / 0 |
| W015 | east_rack_aisle | other | 48 | 34+9 = 0.896 | 0.708 | 44 | 6 | 1.12 | 0.125 / 40 / 0 |
| W018 | Keep 3 m from forklifts | forklift | 8 | 4+4 = 1.0 | 0.5 | 5 | 1 | 1.0 | 1.0 / 7 / 1 |
| W018 | staging_west_walkway | lane | 0 | – | – | 0 | 0 | – | – / 3 / 3 |
| W018 | loading_dock_staging | door | 10 | 10+0 = 1.0 | 1.0 | 10 | 0 | 1.0 | 0.5 / 15 / 0 |
| W018 | north_open_floor | work | 0 | – | – | 0 | 0 | – | – / 0 / 0 |
| W019 | Keep 3 m from forklifts | forklift | 11 | 3+5 = 0.727 | 0.273 | 3 | 0 | 1.0 | 0.727 / 4 / 0 |
| W019 | dock_side_aisle | lane | 1 | 1+0 = 1.0 | 1.0 | 2 | 1 | 1.0 | 1.0 / 7 / 4 |
| W019 | dock_staging_area | door | 21 | 21+0 = 1.0 | 1.0 | 21 | 0 | 1.0 | 0.333 / 15 / 0 |
| W019 | dock_approach_floor | work | 0 | – | – | 0 | 0 | – | – / 6 / 6 |
| W022 | Keep 3 m from forklifts | forklift | 0 | – | – | 0 | 0 | – | – / 0 / 0 |
| W022 | main_aisle_r1_r2 | lane | 51 | 24+12 = 0.706 | 0.471 | 28 | 2 | 1.08 | 0.157 / 9 / 1 |
| W022 | north_staging | work | 4 | 2+0 = 0.5 | 0.5 | 3 | 1 | 1.0 | 0.75 / 3 / 0 |
| W022 | picking_aisle_r2_r3 | work | 30 | 26+4 = 1.0 | 0.867 | 26 | 0 | 1.0 | 0.333 / 22 / 0 |

Reading it:

- In zone rules, site mode covers most events, usually with one alert each. The per-camera
  baseline has one presence subject per camera, so a second person entering a zone that the same
  camera alerted on within the cooldown gets nothing. Its coverage in work areas falls as low as
  0.021 (W010 open floor).
- In the forklift rule the two modes cover about the same events on the test scenes (W010
  0.952 vs 0.905, W018 1.0 vs 1.0, W019 0.727 vs 0.727). The gain there is fewer duplicate alerts,
  not more coverage.
- The open zones of W010 over-alert: 130 alerts for 48 events on the open floor (2.93 per alerted
  event), and 13 false alerts in the racks.

### Validation and development scenes

| Scene | Role | Rule / zone | Events | Covered (own + in cooldown) | Strict recall | Alerts | False | Per alerted event | Per-camera: covered / alerts / false |
|---|---|---|---|---|---|---|---|---|---|
| W012 | validation | west_aisle | 23 | 22+0 = 0.957 | 0.957 | 23 | 0 | 1.05 | 0.304 / 17 / 0 |
| W012 | validation | north_staging_area | 41 | 27+14 = 1.0 | 0.659 | 28 | 0 | 1.04 | 0.146 / 20 / 0 |
| W012 | validation | south_staging_area | 32 | 16+11 = 0.844 | 0.5 | 16 | 0 | 1.0 | 0.312 / 15 / 0 |
| W012 | validation | west_rack_face | 0 | – | – | 0 | 0 | – | – / 0 / 0 |
| W020 | validation | Keep 3 m from forklifts | 95 | 32+22 = 0.568 | 0.337 | 51 | 16 | 1.09 | 0.411 / 73 / 19 |
| W020 | validation | pallet_row_aisle | 18 | 11+4 = 0.833 | 0.611 | 12 | 1 | 1.0 | 0.222 / 22 / 2 |
| W020 | validation | pallet_staging_bays | 28 | 19+2 = 0.75 | 0.679 | 28 | 6 | 1.16 | 0.5 / 32 / 1 |
| W020 | validation | conveyor_work_cell | 13 | 9+1 = 0.769 | 0.692 | 24 | 6 | 2.0 | 0.692 / 18 / 0 |
| W020 | validation | conveyor_side_walkway | 3 | 3+0 = 1.0 | 1.0 | 3 | 0 | 1.0 | 1.0 / 8 / 0 |
| W009 | development | north_walkway | 53 | 32+12 = 0.83 | 0.604 | 36 | 2 | 1.06 | 0.132 / 60 / 0 |
| W009 | development | north_door | 5 | 3+1 = 0.8 | 0.6 | 6 | 2 | 1.33 | 0.6 / 15 / 5 |
| W009 | development | fenced_staging | 16 | 16+0 = 1.0 | 1.0 | 16 | 0 | 1.0 | 0.812 / 30 / 0 |
| W009 | development | pallet_block | 0 | – | – | 0 | 0 | – | – / 11 / 11 |
| W017 | development | Keep 3 m from forklifts | 32 | 6+17 = 0.719 | 0.188 | 6 | 0 | 1.0 | 0.906 / 17 / 0 |
| W017 | development | north_approach_aisle | 0 | – | – | 0 | 0 | – | – / 1 / 1 |
| W017 | development | dock_doors_apron | 0 | – | – | 0 | 0 | – | – / 5 / 5 |
| W017 | development | dock_staging_area | 30 | 10+0 = 0.333 | 0.333 | 10 | 0 | 1.0 | 0.2 / 15 / 0 |

W020 (16 cameras, outside the spec) has the worst head count, positions and identity of all scenes
(MOTA 0.168, IDF1 0.158) and the most false forklift alerts (16 of 51). Its weaker camera coverage
is the clearest difference from W010, but not a proven cause: W020 also has the most people (49.02
on average) and is the building of training scenes W000–W007. On W017 the per-camera baseline
covers more forklift events than site mode (0.906 vs 0.719).

### Settings chosen on validation (W020)

- A worker is shown when a camera sees it with confidence ≥ 0.7 and kept while ≥ 0.6. Vehicles
  need ≥ 0.6.
- Forklift distance pad 0.1 m (bias -0.38 m → 0.01 m on W020).
- "In view" = ≥ 40 px tall at 1280×720. On W020, person detection was 68 % at 20-40 px against 92 %
  at 40-80 px.

## Limits: what is not measured

- **Synthetic footage only.** Every number comes from rendered SmartSpaces warehouses. Real
  footage (crouching, carried loads, glare, real camera placement) has not been measured.
- **Calibration is required.** The cameras need a floor homography and a 3×4 camera matrix.
  SayGuard has no calibration tool yet, so a real site would need one.
- **Identity over time.** The fused ids change within tens of seconds (W010: 10.72 ids per person
  over the 5-minute clip, IDF1 0.516). No appearance re-identification exists. Hailo's person re-id
  networks are planned, not built.
- **W010 is not fully independent.** The detector was never trained on it, but it is the demo
  scene, the tracker was built while watching it (`sitetrack.py` cites its ground truth), and its
  zones are the demo config's, not drawn blind. Its numbers are less independent than those of the
  other five test scenes.
- **W018 and the alert logic.** W017, used to develop the alert logic, is the same building and
  cameras as test scene W018. The two alert-logic changes were made after the frozen run had been
  seen (on W009, W012, W017 and W020).
- **Small samples.** Several zones have no events at all. W018 averages 1.0 visible person.
  W022's north staging has 4 events. Treat single-zone percentages with care. The zone-kind
  aggregate is the more stable figure.
- **Forklift detection is weak on W018 and W019** (found in 0.169 and 0.115 of forklift-frames,
  against 0.793 on W022 and 0.986 on W010). The causes are not established. The rule still covered
  8 of 8 and 8 of 11 events there. Fine-tuning on a customer's own vehicles is the planned
  mitigation; how much it helps is not measured.
- **Strict recall is lower than coverage.** The forklift rule's 60 s cooldown deliberately does not
  re-alert a worker who steps in and out of 3 m of a parked forklift, so strict recall on the test
  scenes is 41 % against 92 % covered. W010's forklift rule covers 0.952 of events but alerts on
  0.429 of them directly.
- **Out of spec, it is worst.** W020 (16 cameras) has MOTA 0.168 and covers 0.568 of forklift
  events, with 16 false alerts of 51. W020 also has the most people and is the training building,
  so the camera count is not proven to be the cause.
- **Not measured at all:** the open-vocabulary fallback in this mode, the pads for vehicles other
  than forklifts, how often the estimated face region misses a face, hand-over between areas
  (not implemented: a worker crossing into another area gets a new id), speech recognition on
  spoken warehouse rules (the parsing sets are typed), the LLM on the UGen300 (Hailo GenAI build),
  and the site detector on Hailo-10H (HEF not built). The frozen run cannot be re-run from this
  tree: the earlier alert logic is not kept, so `docs/results/frozen/` is the record.

## Deployment spec

- **One UGen300 per area of ≤ 12 cameras.** Larger buildings are split into areas, each with its
  own floor plan and config.
- **Watched zones must be seen by ≥ 2 cameras at ≥ 40 px** (a standing person at least 40 px tall).

The 12-camera limit is a coverage spec: the test scenes (3 to 11 cameras) are inside it, and W020
(16 cameras), outside it, gives the worst tracking of all scenes, though camera count is not proven
to be the cause (see above). The 40 px rule comes from the validation scene: person detection falls
from 92 % at 40-80 px to 68 % at 20-40 px on W020. Neither is a compute figure. Whether one UGen300
sustains 12 cameras with this 960 px detector is not measured yet, because the HEF is not built;
that measurement is planned for Stage II, week 1.

## Reproduce every number

Install `requirements.txt` (it includes `motmetrics`, which `eval_site.py` imports) and have
`ffmpeg` and `ffprobe` on `PATH`. The site detector weights must be at
`@work/models/site_yolov8s.pt`.

### Data and streams

W020 and W022 are in the `val/` split. The other scenes are in `train/`.

```bash
python scripts/fetch_smartspaces.py train/Warehouse_010
python scripts/make_site_streams.py train/Warehouse_010
# the same two commands for train/Warehouse_009, 012, 013, 015, 017, 018, 019 and val/Warehouse_020, 022
```

### Site-mode evaluation

Each command writes its JSON to `@work/results/` and `docs/results/`.

```bash
python scripts/eval_site.py --all-zones                                          # W010 -> site_eval.json
python scripts/eval_site.py --config configs/eval/warehouse_013.yaml --all-zones  # -> site_eval_warehouse_013.json
python scripts/eval_site.py --config configs/eval/warehouse_015.yaml --all-zones
python scripts/eval_site.py --config configs/eval/warehouse_018.yaml --all-zones
python scripts/eval_site.py --config configs/eval/warehouse_019.yaml --all-zones
python scripts/eval_site.py --config configs/eval/warehouse_022.yaml --all-zones
python scripts/eval_site.py --config configs/eval/warehouse_012.yaml --all-zones
python scripts/eval_site.py --config configs/eval/warehouse_020.yaml --all-zones
python scripts/eval_site.py --config configs/eval/warehouse_009.yaml --all-zones
python scripts/eval_site.py --config configs/eval/warehouse_017.yaml --all-zones
```

Useful options: `--dets-cache dets_w013.pkl` saves the detector output on the first run and replays
it afterwards (tied to the weights it was made with), so tracker and rule changes can be measured
without the GPU. `--tag NAME` writes only to `@work/results/` (the usual file name with `_NAME`
appended, plus an alert dump) and leaves `docs/results/` alone. `--limit N` runs only the first N
frames.

### Aggregate by zone kind

No script in the repo computes it. From the repository root, this Python reproduces the test-scene
table (change `docs/results/` to `docs/results/frozen/` for the frozen run):

```python
import json, collections
KIND = {"Keep 3 m from forklifts": "forklift", "No walking in aisle 7 (forklift parked)": "lane",
        "dock": "door", "loading_dock_staging": "door", "dock_staging_area": "door",
        "west_cross_aisle": "lane", "north_cross_aisle": "lane", "staging_west_walkway": "lane",
        "dock_side_aisle": "lane", "main_aisle_r1_r2": "lane",
        "west_rack_line": "other", "east_rack_aisle": "other"}          # every other zone: work
tot = collections.defaultdict(lambda: [0, 0, 0, 0, 0])                 # events, own alert, covered, alerts, false
for f in ["site_eval"] + [f"site_eval_warehouse_{n}" for n in ("013", "015", "018", "019", "022")]:
    for rule, s in json.load(open(f"docs/results/{f}.json"))["alerts"]["site"].items():
        t = tot[KIND.get(rule.removeprefix("Zone check: "), "work")]
        for i, v in enumerate((s["events"], s["events_alerted"], s["events_alerted"] + s["events_in_cooldown"],
                               s["alerts"], s["false_alerts"])):
            t[i] += v
for k, (e, o, c, a, fa) in tot.items():
    print(f"{k:8s} covered {c}/{e}  strict {o}/{e}  false {fa} of {a}")
```

### Spoken rules

```bash
python scripts/eval_rules.py --site          # needs Ollama with qwen2.5:1.5b -> docs/results/rule_parsing_site.json
python scripts/eval_rules.py --site --dev    # dev set only, writes to @work/results/ only
```

`docs/results/rule_parsing_site_v1_first_run.json` is the kept record of held-out v1's first run.
The parser has changed since, so it cannot be reproduced.

### The site detector

The shipped weights come from two runs: a first model on four scenes, then a continuation on the
eight-scene set. GPU training is not bit-exact, so a rebuild gives similar weights, not identical
ones.

```bash
# fetch each scene first: python scripts/fetch_smartspaces.py train/Warehouse_000 (and 002, 004, 006, 008, 011, 014, 016)
python scripts/prepare_site_dataset.py --train train/Warehouse_000 train/Warehouse_004 train/Warehouse_008 train/Warehouse_014 --val train/Warehouse_016 --name site
python scripts/train_detector.py --data @work/datasets/site/data.yaml --name site_v8s --out site_v1_best.pt --imgsz 960 --epochs 30 --batch 16
python scripts/prepare_site_dataset.py --train train/Warehouse_000 train/Warehouse_002 train/Warehouse_004 train/Warehouse_006 train/Warehouse_008 train/Warehouse_011 train/Warehouse_014 "train/Warehouse_016:~9,10" --val "train/Warehouse_016:9,10" --name site_v3
python scripts/train_detector.py --model site_v1_best.pt --data @work/datasets/site_v3/data.yaml --name site_v8s_v3 --out site_yolov8s.pt --imgsz 960 --epochs 20 --batch 16
```

### Unit tests

```bash
python -m pytest                                                                   # 264 passing
python -m pytest tests/test_sitetrack.py tests/test_site_rules.py tests/test_sources_synced.py   # the site-mode tests
```
