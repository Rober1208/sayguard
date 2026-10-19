English | [繁體中文](DEMO_SCRIPT.zh-TW.md)

# 3-minute demo video: script and shot list (Stage I)

Target length 2:50 with English narration (competition rule §6.1); the hard limit is 3:00.
Record the dashboard at 1920×1080 (OBS, display capture, browser in full screen) and add the
voice-over afterwards, so the narration can follow what actually happened in each take.
You need two takes: **hard-hat mode** (`configs/site.yaml`) and **warehouse mode**
(`configs/warehouse.yaml`). The two modes cannot run at the same time.

Every measured number on screen comes from [FACTS.md](FACTS.md), which is generated from the raw
results in [results/](results/README.md). UGen300 throughput, load and latency figures are
estimates from Hailo's published Model Zoo numbers. Label them as estimates on screen. The
prototype in the video runs on a PC GPU: say "locally" or "on site" for what it does today, and
"on the UGen300 in Stage II" for the plan.

## What the video has to get across

1. **One box watches every camera.** In hard-hat mode it checks hard hats on ten cameras. In
   warehouse mode it fuses eleven cameras on one floor plan. Today the box is a PC GPU; in Stage II
   it is one UGen300.
2. **The local LLM does two jobs a detector cannot.** It turns a spoken or typed sentence into a
   rule, in Mandarin or English, and it plans the daily report.
3. **The LLM proposes, code checks, the officer confirms the rule.** For the report, the LLM can
   only choose listed issues and vetted fixes. Every number in the video is measured or labelled as
   an estimate.

## Shot list

| Time | Section | Screen | Narration |
|---|---|---|---|
| 0:00–0:13 | Problem | Title card for 2 s: "SayGuard · one UGen300 per site area · every camera watched · rules you can say". Then the hard-hat dashboard with ten tiles of looping demo streams. | "Sites are full of cameras that few people watch live. SayGuard is one box next to the recorder: a PC GPU today, an ASUS UGen300 in Stage II. It watches every camera, and the safety officer adds rules by saying them." |
| 0:13–0:37 | Hard-hat mode | Camera wall. Point at the KPI strip ("Frames analysed / s", "Est. UGen300 load"). A tile with a bare head (House Frame in our footage) shows the red "NO HARD HAT" box, then the alert card appears and its badge changes from "Checking…" to "Confirmed by double-check". Zoom in on a pixelated face. Caption 1. | "In hard-hat mode, ten cameras with a ten-frames-a-second target share one batched YOLOv8s detector: about 105 frames a second, measured. At a nominal 100 frames a second, that is an estimated 40 % of one UGen300. A bare head must persist for two seconds. Then a CLIP yes-or-no check looks at the crop and confirms the alert or marks it as a false alarm. Faces are pixelated locally." |
| 0:37–1:12 | Spoken rule (the LLM's first job) | (1) Hold the mic button and say "Alert me if more than two people are at the site gate after 10 pm." The transcript appears, then the "Proposed rule" card with the badge "LLM + checker". "When" reads 22:00–06:00. Open "Checker notes", which shows the LLM's time window and the one the checker used. Click "Cancel". (2) Type "A區有人靠近挖土機三公尺內要立刻警告" and click "Check". The card reads "Person within 3 m of an excavator", Zone A, severity High. Click "Activate rule". (3) On the Excavation Pit tile, an orange "excavator" box, a red distance line in metres and the alert appear. Caption 2. | "This is the LLM's first special job. A new rule about known objects, zones, distances, counts or times is one sentence. Whisper transcribes it, Qwen2.5-1.5B drafts structured JSON, and deterministic code checks the numbers, times and objects it finds in what was said. Here the model ended 'after 10 pm' at midnight; the checker applied the stated convention, 22:00 to 06:00, and shows the correction. Mandarin works the same way, and the confirmed excavator rule is live within seconds on this PC. An open-vocabulary detector finds the excavator from the word alone." |
| 1:12–1:44 | Warehouse mode | Cut to the warehouse take. The floor-plan tile is labelled "Floor plan · each worker counted once" and shows workers, zones and vehicles. The KPI "Workers on site" shows the unique count, with camera detections beneath it. An alert "Worker within 3 m of a forklift": the worker turns red on the plan and in every camera tile that sees them, and the alert card lists the cameras ("seen by …") and the distance in metres. Then hold the mic and say "Alert me if anyone enters the dock." On the card, "Where" reads Dock. Click "Activate rule". Caption 3. | "Warehouse mode is the multi-camera version, shown here on a synthetic warehouse. Eleven calibrated cameras are fused on one floor plan, so each worker is counted once, not once per camera. Adding up the cameras would roughly double the head count. Zones are drawn once on the plan, distances to forklifts are measured in metres, and each worker raises one alert per entry, not one per camera, lit in every camera that sees them. Rules still come by voice, with zone names as people say them." |
| 1:44–2:10 | The daily report (the LLM's second job) | Back to the hard-hat take, after at least 15 minutes of history. Click "Daily safety report". The modal shows "Writing the report with the on-device LLM…". The report then shows the summary, the per-camera table and three "Recommended actions" in the form **camera** · problem: fix. Zoom in on the footer: "numbers and tables computed by SayGuard, actions drafted by on-device LLM (qwen2.5:1.5b)". Caption 4. | "The LLM's second job is the daily report, but code computes every number and table. The model only plans: from the problems that really happened, it picks the three that matter and a vetted fix for each, as numbered choices. So it cannot blame the wrong camera or invent a figure. Two blind AI judges found the wrong problem in 20 % of freely written actions, and in none of these." |
| 2:10–2:31 | Numbers | Full-screen results card (caption 5). | "Every number is measured or labelled as an estimate, and the scripts are in the repo. On typed held-out sentences in hard-hat mode, the checker lifts the small model from 40 to 95 % of rules exactly right. On six test buildings the detector never trained on, after two alert-logic fixes, warehouse mode covered all 53 door and dock events, 52 with an alert of their own." |
| 2:31–2:50 | Ask | Model map (job · model · Hailo-10H reference), then the closing card "SayGuard · github.com/Rober1208/sayguard". Caption 6. | "Every job maps to a published Hailo-10H model, and our hard-hat detector is already compiled for it: in Hailo's emulator, INT8 scores 0.889 mAP50 against 0.893 in float. In Stage II we move every model onto one 2.5-watt UGen300 per site area and measure it there. We are asking for that chance, on a real site." |

If your take of shot 0:37 shows no checker note (the LLM got the window right), replace the
sentence about midnight with: "When the model gets a detail wrong, the checker corrects it and
shows the correction." The midnight case is the one recorded on the live server
(FACTS.md, "Rule-parsing examples"). The LLM runs at temperature 0, so the same sentence
usually gives the same draft.

## On-screen captions (copy exactly)

1. `10 cameras at a 10 fps target · ≈ 105 frames/s measured through one batched YOLOv8s · ≈ 40 % of one UGen300 at the nominal 100 frames/s (estimate from Hailo's YOLOv8s reference, 252 FPS at batch 8)`
2. `Whisper-Small → Qwen2.5-1.5B (JSON schema) → deterministic checker → officer confirms · rule parse 0.35–0.40 s mean (max 0.64 s) on the PC · UGen300 estimate ≈ 11.3 s per rule (not measured)`
3. `Warehouse_010 (NVIDIA PhysicalAI-SmartSpaces, synthetic; our demo scene) · 11 calibrated cameras · people present on average 24.97 · fused on one floor plan 26.56 · summed over cameras 51.64`
4. `20 synthetic days · 2 blind AI judges per action · wrong problem: LLM writing freely 20.0 % → LLM choosing a vetted fix 0.0 %`
5. Results card:
   ```
   Rule parsing, typed held-out sentences (rules exactly right; speech recognition not measured)
     Hard-hat mode, 20 sentences:     LLM alone 40 %     →  LLM + checker 95 %     (rule extractor alone 90 %)
     Warehouse mode, 2 × 25:          LLM alone 16–24 %  →  LLM + checker 84–88 %  (rule extractor alone 68–80 %)
     LLM measured with Ollama qwen2.5:1.5b on the PC; the Hailo build is not measured yet
   Daily report actions, 2 blind AI judges: wrong problem 20.0 % → 0.0 %
   Yes/no check (CLIP, probe_train.json):   95.7 % of hard-hat crops passed (proxy for false flags) · 94.9 % of violations kept
   Warehouse mode, 6 test buildings the detector never trained on
     final run, after two alert-logic fixes developed on non-test scenes once the frozen run had been seen
     covered = an alert of its own, or inside the cooldown of an alert for the same person
                        covered             own alert (strict)   false alerts
     door / dock        53 of 53 (100 %)    52 of 53 (98 %)      0 of 52
     forklift 3 m       56 of 61 (92 %)     25 of 61 (41 %)*     1 of 26
     lanes / aisles     98 of 118 (83 %)    71 of 118 (60 %)     3 of 77
     work areas         323 of 338 (96 %)   237 of 338 (70 %)    26 of 362
     other zones        43 of 48 (90 %)     34 of 48 (71 %)      6 of 44
     frozen run, before the fixes: covered 94 / 89 / 64 / 83 / 71 % · false alerts 0 of 26 · 1 of 24 · 1 of 40 · 16 of 244 · 4 of 30
     * the 60 s cooldown deliberately does not re-alert a worker stepping in and out of 3 m of a parked forklift
     W010 is the demo scene (zones from the demo config); W017 (development) is W018's twin
   Hard-hat detector:  mAP50 0.9393 (Ultralytics evaluator) · Hailo's emulator, a different evaluation: INT8 0.889 vs float 0.8927 (compare only these two)
   ```
6. `Hailo-10H references: YOLOv8s 252 FPS (batch 8) · Qwen2.5-1.5B-Instruct 7.35 tokens/s · Whisper-Small 10.6 tokens/s · CLIP ViT-B/32 81.5 FPS · one UGen300 per area of ≤ 12 cameras (coverage spec; warehouse-mode load measured in Stage II)`

## Before recording

### Take 1: hard-hat mode

- Start Ollama (`ollama serve`, with `qwen2.5:1.5b` pulled), then `python -m sayguard serve`, and
  open `http://127.0.0.1:8000/?lang=en`. On Windows, `start_demo.cmd` does all three; the page
  reconnects by itself while the models load.
- In "Active rules", delete the test rules and keep only "Hard hat required on site". Let it run
  for about 15 minutes so the report has history on several cameras. The LLM plans the actions
  only when at least three (camera, problem) pairs happened. With fewer, the actions come from the
  template, and the report footer says "template report".
- Warm up before you press record:
  - Speak one throwaway rule and cancel it. Whisper loads on first use.
  - Wait for the first hard-hat alert. This loads the CLIP yes/no check.
  - Activate and then delete any object rule. YOLO-World loads the first time a rule names an
    object and stays loaded afterwards.
- The KPI "On-device LLM" must read Online. If it does not, the rule card shows the badge
  "Rule extractor" instead of "LLM + checker", and the LLM shots cannot be recorded.

### Take 2: warehouse mode

- Once: `python scripts/fetch_smartspaces.py train/Warehouse_010` and
  `python scripts/make_site_streams.py train/Warehouse_010`. The site detector weights must be at
  `@work/models/site_yolov8s.pt` (training is described in [SITE_MODE.md](SITE_MODE.md)).
- Stop take 1 first. Both modes listen on port 8000 by default and need the same GPU. Then start
  Ollama and run `python -m sayguard serve --config configs/warehouse.yaml`. On Windows,
  `start_demo_warehouse.cmd` does both and opens the page.
- The two default rules ("Keep 3 m from forklifts", "No walking in aisle 7 (forklift parked)")
  produce the alerts. The footage loops; at each loop the floor-plan state starts over, so record
  the floor-plan shot within one loop.
- Speak English while the dashboard is in English. The dashboard sends its UI language to the
  speech model, so a spoken Mandarin rule needs the 中文 toggle first. Typed Mandarin works in
  either language.

### Both takes

- Stop other GPU-heavy programs, such as training jobs, so that every camera stays at 10 fps.
- Honesty checks: call the warehouse footage synthetic, and call the UGen300 load an estimate.
  The dashboard's "on-device LLM" labels name the design target; in the narration say "locally",
  because the demo runs on the PC. Worker ids are never drawn in warehouse mode, because they
  change within tens of seconds (10.7 ids per person in the 5-minute Warehouse_010 clip); do not
  describe them as stable identities. The excavator is found by open-vocabulary detection, and the
  distance line in hard-hat mode is estimated from head size; neither accuracy is measured, so do
  not quote either as a measurement.
