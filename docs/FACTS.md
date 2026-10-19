# SayGuard facts sheet (generated from docs/results/*.json — use ONLY these numbers)

## Hard-hat detector (detector_metrics.json)
```
{"generated": "2026-10-18T13:13:22", "model": {"weights": "D:\\ugen-work\\models\\helmet_yolov8s.pt", "architecture": "YOLOv8s (fine-tuned, 2 classes)", "size_mb": 22.52, "params": 11136374, "params_m": 11.14, "gflops": 28.6}, "dataset": {"source": "keremberke/hard-hat-detection (Roboflow Hard Hats v2, CC BY 4.0), classes helmet/head", "data_yaml": "D:\\ugen-work\\datasets\\helmet\\data.yaml", "split": "test", "images": 2001, "instances": {"helmet": 4479, "head": 1038}, "image_dir": "D:\\ugen-work\\datasets\\helmet\\images\\test"}, "settings": {"imgsz": 640, "batch": 16, "workers": 2, "precision": "fp16", "conf": 0.001, "iou": 0.7, "note": "Ultralytics val defaults; P/R are at the max-F1 confidence"}, "hardware": {"device": "NVIDIA GeForce RTX 5060 Ti", "torch": "2.11.0+cu128", "ultralytics": "8.4.182"}, "metrics": {"mAP50": 0.9393, "mAP50-95": 0.5945, "mAP75": 0.6572, "precision": 0.9132, "recall": 0.8986, "per_class": {"helmet": {"precision": 0.9194, "recall": 0.8964, "mAP50": 0.9447, "mAP50-95": 0.5464, "images": 1626, "instances": 4479}, "head": {"precision": 0.9069, "recall": 0.9008, "mAP50": 0.9338, "mAP50-95": 0.6425, "images": 413, "instances": 1038}}, "images_evaluated": 2001, "instances": 5517, "speed_ms_per_image": {"preprocess": 0.68, "inference": 2.098, "loss": 0.0, "postprocess": 0.974}, "confusion_matrix": {"labels": ["helmet", "head", "background"], "rows": "predicted", "cols": "true", "matrix": [[4095, 225, 14808], [334, 795, 4220], [50, 18, 0]]}, "save_dir":
```

## Spoken rules, camera sites (rule_parsing.json)
- dev (used to build the extractor): 32 sentences; accuracy {'llm_only': 0.25, 'rules': 1.0, 'llm+check': 0.969}; llm+check latency {'mean': 352, 'max': 502}
- held-out (never used for tuning): 20 sentences; accuracy {'llm_only': 0.4, 'rules': 0.9, 'llm+check': 0.95}; llm+check latency {'mean': 347, 'max': 469}

## Spoken rules, warehouse floor plan (rule_parsing_site.json; v1 first run in rule_parsing_site_v1_first_run.json)
- site dev (may be used to fix the parser): 15 sentences; accuracy {'llm_only': 0.2, 'rules': 1.0, 'llm+check': 1.0}; llm+check latency {'mean': 363, 'max': 457}
- site held-out v1 (first run 84 % before fixes; seen since): 25 sentences; accuracy {'llm_only': 0.16, 'rules': 0.96, 'llm+check': 1.0}; llm+check latency {'mean': 392, 'max': 515}
- site held-out v2 (fresh, never used for tuning): 25 sentences; accuracy {'llm_only': 0.24, 'rules': 0.68, 'llm+check': 0.88}; llm+check latency {'mean': 395, 'max': 638}
- FIRST RUN of held-out v1 (before the parser fixes it exposed): 25 sentences; accuracy {'llm_only': 0.16, 'rules': 0.8, 'llm+check': 0.84}

## Daily report actions (report_actions.json; 20 synthetic days, Qwen2.5-1.5B)
- keyword check: {"free_text": {"days": 20, "accepted": 20, "actions": 60, "name_a_camera": 45, "misattributed": 8, "misattributed_pct": 13.3, "accepted_pct": 100.0}, "structured": {"days": 20, "accepted": 19, "actions": 57, "name_a_camera": 57, "misattributed": 0, "misattributed_pct": 0.0, "accepted_pct": 95.0}, "menu": {"days": 20, "accepted": 20, "actions": 60, "name_a_camera": 60, "misattributed": 0, "misattributed_pct": 0.0, "accepted_pct": 100.0}}
- independent blind judges: {"method": "2 independent judge agents per action, blind to which approach wrote it, given each camera's real problems that day (prompt, batches and verdicts: docs/results/report_actions_judging/)", "agreement_on_wrong_problem": "176/177", "by_approach": {"free_text": {"actions": 60, "wrong_problem_both_judges_pct": 20.0, "wrong_problem_either_judge_pct": 20.0, "sensible_both_judges_pct": 45.0, "names_a_camera_pct": 88.3}, "structured": {"actions": 57, "wrong_problem_both_judges_pct": 3.5, "wrong_problem_either_judge_pct": 5.3, "sensible_both_judges_pct": 52.6, "names_a_camera_pct": 100.0}, "structured/llm": {"actions": 43, "wrong_problem_both_judges_pct": 4.7, "wrong_problem_either_judge_pct": 7.0, "sensible_both_judges_pct": 37.2, "names_a_camera_pct": 100.0}, "structured/template": {"actions": 14, "wrong_problem_both_judges_pct": 0.0, "wrong_problem_either_judge_pct": 0.0, "sensible_both_judges_pct": 100.0, "names_a_camera_pct": 100.0}, "menu": {"actions": 60, "wrong_problem_both_judges_pct": 0.0, "wrong_problem_either_judge_pct": 0.0, "sensible_both_judges_pct": 100.0, "names_a_camera_pct": 100.0}}}

## Alert double-check (yesno_check.json, probe_train.json)
```
{"encoder": "CLIP ViT-B/32 (OpenAI), frozen", "head": "linear 512->2 on 10x unit features", "train_crops": 2950, "val_accuracy": 0.95, "weight_decay": 0.0, "test_crops": {"helmet": 94, "head": 99}, "test_accuracy": 0.953, "false_alarm_filter_rate": 0.957, "violation_confirm_rate": 0.949, "latency_ms_one_crop": {"mean": 15.2, "p95": 24.7, "device": "cuda", "gpu": "NVIDIA GeForce RTX 5060 Ti"}}
```

## Hailo-10H (hailo_int8.json, clip_probe_int8.json)
```
{"har": "helmet_c512clip_optimized.har", "images": 2001, "instances": 5517, "fp_optimized": {"mAP50-95": 0.5662, "mAP50": 0.8927, "helmet_mAP50": 0.9035, "head_mAP50": 0.8819, "seconds": 167.8}, "quantized": {"mAP50-95": 0.56, "mAP50": 0.889, "helmet_mAP50": 0.9008, "head_mAP50": 0.8772, "seconds": 1301.7}, "mAP50_drop": 0.0037, "mAP50-95_drop": 0.0062}
```
```
{"crops": {"train": 2950, "val": 578, "test": 193}, "cosine_to_pytorch": {"fp_optimized": 0.9999, "quantized": 0.9634}, "pc_probe_on_pc_features": {"accuracy": 0.953, "false_alarm_filter_rate": 0.957, "violation_confirm_rate": 0.949}, "pc_probe_on_int8_features": {"accuracy": 0.943, "false_alarm_filter_rate": 0.957, "violation_confirm_rate": 0.929}, "int8_probe_on_int8_features": {"accuracy": 0.953, "false_alarm_filter_rate": 0.968, "violation_confirm_rate": 0.939}, "note": "Model Zoo clip_vit_b_32_image_encoder + its model script; 64 site crops as calibration; bit-exact INT8 emulation (SDK_QUANTIZED). Same crops and probe recipe as scripts/eval_yesno.py."}
```

## Site mode (site_eval*.json; frozen-parameter run in docs/results/frozen/)
Roles: test = W010, W013, W015, W018, W019, W022 (never trained or tuned on, distinct buildings from training); validation = W012, W020 (W020 = same building as training scenes W000-W007, 16 cameras: out of the 12-camera spec); W009 (twin of training scene W008) and W017 (twin of test W018) = development only.

### W010 (11 cameras, 3000 frames at 10.0 fps)
- people: true 24.97, site mode 26.56 (abs err 1.74), camera sum 51.64 (abs err 26.67)
- positions: recall 0.928, precision 0.873, MOTA 0.788, mean position error 0.209 m, duplicates 2.35 % of person-frames, IDF1 0.516, ids per person 10.72
- forklift found in 0.986 of forklift-frames; distance error median 0.2 m, MAE 0.28 m, bias -0.12 m
- Keep 3 m from forklifts: 42 events; site mode covered 18+22 (own alert + in cooldown) = 0.952, strict recall 0.429, alerts 18, false 0, per alerted event 1.0; per-camera mode covered 0.905, alerts 23, false 0
- No walking in aisle 7 (forklift parked): 9 events; site mode covered 8+1 (own alert + in cooldown) = 1.0, strict recall 0.889, alerts 8, false 0, per alerted event 1.0; per-camera mode covered 0.889, alerts 10, false 0
- Zone check: dock: 22 events; site mode covered 21+1 (own alert + in cooldown) = 1.0, strict recall 0.955, alerts 21, false 0, per alerted event 1.0; per-camera mode covered 0.364, alerts 14, false 0
- Zone check: racks: 43 events; site mode covered 38+5 (own alert + in cooldown) = 1.0, strict recall 0.884, alerts 56, false 13, per alerted event 1.13; per-camera mode covered 0.186, alerts 31, false 0
- Zone check: open_floor: 48 events; site mode covered 42+6 (own alert + in cooldown) = 1.0, strict recall 0.875, alerts 130, false 7, per alerted event 2.93; per-camera mode covered 0.021, alerts 45, false 0

### W013 (10 cameras, 3000 frames at 10.0 fps)
- people: true 15.83, site mode 15.49 (abs err 0.55), camera sum 39.99 (abs err 24.16)
- positions: recall 0.961, precision 0.982, MOTA 0.941, mean position error 0.176 m, duplicates 1.55 % of person-frames, IDF1 0.633, ids per person 4.94
- forklift found in None of forklift-frames; distance error median None m, MAE None m, bias None m
- Zone check: west_cross_aisle: 17 events; site mode covered 11+5 (own alert + in cooldown) = 0.941, strict recall 0.647, alerts 12, false 0, per alerted event 1.09; per-camera mode covered 0.529, alerts 34, false 0
- Zone check: north_staging_area: 11 events; site mode covered 9+1 (own alert + in cooldown) = 0.909, strict recall 0.818, alerts 11, false 2, per alerted event 1.0; per-camera mode covered 0.455, alerts 25, false 5
- Zone check: south_open_floor: 36 events; site mode covered 26+9 (own alert + in cooldown) = 0.972, strict recall 0.722, alerts 29, false 0, per alerted event 1.12; per-camera mode covered 0.028, alerts 29, false 0
- Zone check: west_rack_line: 0 events; site mode covered 0+0 (own alert + in cooldown) = None, strict recall None, alerts 0, false 0, per alerted event None; per-camera mode covered None, alerts 0, false 0

### W015 (10 cameras, 3000 frames at 10.0 fps)
- people: true 22.0, site mode 22.91 (abs err 1.05), camera sum 82.09 (abs err 60.09)
- positions: recall 0.987, precision 0.948, MOTA 0.932, mean position error 0.173 m, duplicates 3.35 % of person-frames, IDF1 0.701, ids per person 3.64
- forklift found in None of forklift-frames; distance error median None m, MAE None m, bias None m
- Zone check: north_cross_aisle: 40 events; site mode covered 27+9 (own alert + in cooldown) = 0.9, strict recall 0.675, alerts 27, false 0, per alerted event 1.0; per-camera mode covered 0.25, alerts 22, false 0
- Zone check: south_staging_floor: 120 events; site mode covered 56+54 (own alert + in cooldown) = 0.917, strict recall 0.467, alerts 59, false 3, per alerted event 1.0; per-camera mode covered 0.075, alerts 40, false 0
- Zone check: west_rack_aisle: 46 events; site mode covered 38+7 (own alert + in cooldown) = 0.978, strict recall 0.826, alerts 48, false 0, per alerted event 1.26; per-camera mode covered 0.022, alerts 38, false 0
- Zone check: east_rack_aisle: 48 events; site mode covered 34+9 (own alert + in cooldown) = 0.896, strict recall 0.708, alerts 44, false 6, per alerted event 1.12; per-camera mode covered 0.125, alerts 40, false 0

### W018 (3 cameras, 3000 frames at 10.0 fps)
- people: true 1.0, site mode 1.05 (abs err 0.05), camera sum 2.76 (abs err 1.76)
- positions: recall 1.0, precision 0.955, MOTA 0.953, mean position error 0.231 m, duplicates 0.0 % of person-frames, IDF1 0.977, ids per person 1.0
- forklift found in 0.169 of forklift-frames; distance error median 0.45 m, MAE 0.65 m, bias -0.02 m
- Keep 3 m from forklifts: 8 events; site mode covered 4+4 (own alert + in cooldown) = 1.0, strict recall 0.5, alerts 5, false 1, per alerted event 1.0; per-camera mode covered 1.0, alerts 7, false 1
- Zone check: staging_west_walkway: 0 events; site mode covered 0+0 (own alert + in cooldown) = None, strict recall None, alerts 0, false 0, per alerted event None; per-camera mode covered None, alerts 3, false 3
- Zone check: loading_dock_staging: 10 events; site mode covered 10+0 (own alert + in cooldown) = 1.0, strict recall 1.0, alerts 10, false 0, per alerted event 1.0; per-camera mode covered 0.5, alerts 15, false 0
- Zone check: north_open_floor: 0 events; site mode covered 0+0 (own alert + in cooldown) = None, strict recall None, alerts 0, false 0, per alerted event None; per-camera mode covered None, alerts 0, false 0

### W019 (3 cameras, 3000 frames at 10.0 fps)
- people: true 2.07, site mode 1.78 (abs err 0.34), camera sum 4.5 (abs err 2.45)
- positions: recall 0.842, precision 0.977, MOTA 0.82, mean position error 0.26 m, duplicates 1.45 % of person-frames, IDF1 0.59, ids per person 6.33
- forklift found in 0.115 of forklift-frames; distance error median 0.38 m, MAE 0.44 m, bias -0.08 m
- Keep 3 m from forklifts: 11 events; site mode covered 3+5 (own alert + in cooldown) = 0.727, strict recall 0.273, alerts 3, false 0, per alerted event 1.0; per-camera mode covered 0.727, alerts 4, false 0
- Zone check: dock_side_aisle: 1 events; site mode covered 1+0 (own alert + in cooldown) = 1.0, strict recall 1.0, alerts 2, false 1, per alerted event 1.0; per-camera mode covered 1.0, alerts 7, false 4
- Zone check: dock_staging_area: 21 events; site mode covered 21+0 (own alert + in cooldown) = 1.0, strict recall 1.0, alerts 21, false 0, per alerted event 1.0; per-camera mode covered 0.333, alerts 15, false 0
- Zone check: dock_approach_floor: 0 events; site mode covered 0+0 (own alert + in cooldown) = None, strict recall None, alerts 0, false 0, per alerted event None; per-camera mode covered None, alerts 6, false 6

### W022 (4 cameras, 3090 frames at 10.0 fps)
- people: true 9.8, site mode 8.48 (abs err 1.35), camera sum 13.51 (abs err 3.86)
- positions: recall 0.83, precision 0.959, MOTA 0.791, mean position error 0.243 m, duplicates 1.2 % of person-frames, IDF1 0.287, ids per person 6.58
- forklift found in 0.793 of forklift-frames; distance error median 0.47 m, MAE 0.6 m, bias 0.43 m
- Keep 3 m from forklifts: 0 events; site mode covered 0+0 (own alert + in cooldown) = None, strict recall None, alerts 0, false 0, per alerted event None; per-camera mode covered None, alerts 0, false 0
- Zone check: main_aisle_r1_r2: 51 events; site mode covered 24+12 (own alert + in cooldown) = 0.706, strict recall 0.471, alerts 28, false 2, per alerted event 1.08; per-camera mode covered 0.157, alerts 9, false 1
- Zone check: north_staging: 4 events; site mode covered 2+0 (own alert + in cooldown) = 0.5, strict recall 0.5, alerts 3, false 1, per alerted event 1.0; per-camera mode covered 0.75, alerts 3, false 0
- Zone check: picking_aisle_r2_r3: 30 events; site mode covered 26+4 (own alert + in cooldown) = 1.0, strict recall 0.867, alerts 26, false 0, per alerted event 1.0; per-camera mode covered 0.333, alerts 22, false 0

### W009 (12 cameras, 3000 frames at 10.0 fps)
- people: true 13.0, site mode 13.99 (abs err 1.1), camera sum 56.8 (abs err 43.8)
- positions: recall 0.977, precision 0.909, MOTA 0.874, mean position error 0.174 m, duplicates 7.56 % of person-frames, IDF1 0.485, ids per person 10.0
- forklift found in None of forklift-frames; distance error median None m, MAE None m, bias None m
- Zone check: north_walkway: 53 events; site mode covered 32+12 (own alert + in cooldown) = 0.83, strict recall 0.604, alerts 36, false 2, per alerted event 1.06; per-camera mode covered 0.132, alerts 60, false 0
- Zone check: north_door: 5 events; site mode covered 3+1 (own alert + in cooldown) = 0.8, strict recall 0.6, alerts 6, false 2, per alerted event 1.33; per-camera mode covered 0.6, alerts 15, false 5
- Zone check: fenced_staging: 16 events; site mode covered 16+0 (own alert + in cooldown) = 1.0, strict recall 1.0, alerts 16, false 0, per alerted event 1.0; per-camera mode covered 0.812, alerts 30, false 0
- Zone check: pallet_block: 0 events; site mode covered 0+0 (own alert + in cooldown) = None, strict recall None, alerts 0, false 0, per alerted event None; per-camera mode covered None, alerts 11, false 11

### W017 (3 cameras, 3000 frames at 10.0 fps)
- people: true 1.75, site mode 1.17 (abs err 0.63), camera sum 3.87 (abs err 2.12)
- positions: recall 0.617, precision 0.924, MOTA 0.554, mean position error 0.237 m, duplicates 0.32 % of person-frames, IDF1 0.7, ids per person 8.0
- forklift found in 0.331 of forklift-frames; distance error median 0.3 m, MAE 0.46 m, bias 0.06 m
- Keep 3 m from forklifts: 32 events; site mode covered 6+17 (own alert + in cooldown) = 0.719, strict recall 0.188, alerts 6, false 0, per alerted event 1.0; per-camera mode covered 0.906, alerts 17, false 0
- Zone check: north_approach_aisle: 0 events; site mode covered 0+0 (own alert + in cooldown) = None, strict recall None, alerts 0, false 0, per alerted event None; per-camera mode covered None, alerts 1, false 1
- Zone check: dock_doors_apron: 0 events; site mode covered 0+0 (own alert + in cooldown) = None, strict recall None, alerts 0, false 0, per alerted event None; per-camera mode covered None, alerts 5, false 5
- Zone check: dock_staging_area: 30 events; site mode covered 10+0 (own alert + in cooldown) = 0.333, strict recall 0.333, alerts 10, false 0, per alerted event 1.0; per-camera mode covered 0.2, alerts 15, false 0

### W012 (8 cameras, 3000 frames at 10.0 fps)
- people: true 15.0, site mode 15.05 (abs err 0.31), camera sum 30.78 (abs err 15.78)
- positions: recall 0.976, precision 0.973, MOTA 0.947, mean position error 0.199 m, duplicates 2.43 % of person-frames, IDF1 0.651, ids per person 5.53
- forklift found in None of forklift-frames; distance error median None m, MAE None m, bias None m
- Zone check: west_aisle: 23 events; site mode covered 22+0 (own alert + in cooldown) = 0.957, strict recall 0.957, alerts 23, false 0, per alerted event 1.05; per-camera mode covered 0.304, alerts 17, false 0
- Zone check: north_staging_area: 41 events; site mode covered 27+14 (own alert + in cooldown) = 1.0, strict recall 0.659, alerts 28, false 0, per alerted event 1.04; per-camera mode covered 0.146, alerts 20, false 0
- Zone check: south_staging_area: 32 events; site mode covered 16+11 (own alert + in cooldown) = 0.844, strict recall 0.5, alerts 16, false 0, per alerted event 1.0; per-camera mode covered 0.312, alerts 15, false 0
- Zone check: west_rack_face: 0 events; site mode covered 0+0 (own alert + in cooldown) = None, strict recall None, alerts 0, false 0, per alerted event None; per-camera mode covered None, alerts 0, false 0

### W020 (16 cameras, 3000 frames at 10.0 fps)
- people: true 49.02, site mode 42.1 (abs err 7.19), camera sum 97.13 (abs err 48.11)
- positions: recall 0.519, precision 0.605, MOTA 0.168, mean position error 0.348 m, duplicates 2.63 % of person-frames, IDF1 0.158, ids per person 22.32
- forklift found in 0.802 of forklift-frames; distance error median 0.32 m, MAE 0.52 m, bias 0.01 m
- Keep 3 m from forklifts: 95 events; site mode covered 32+22 (own alert + in cooldown) = 0.568, strict recall 0.337, alerts 51, false 16, per alerted event 1.09; per-camera mode covered 0.411, alerts 73, false 19
- Zone check: pallet_row_aisle: 18 events; site mode covered 11+4 (own alert + in cooldown) = 0.833, strict recall 0.611, alerts 12, false 1, per alerted event 1.0; per-camera mode covered 0.222, alerts 22, false 2
- Zone check: pallet_staging_bays: 28 events; site mode covered 19+2 (own alert + in cooldown) = 0.75, strict recall 0.679, alerts 28, false 6, per alerted event 1.16; per-camera mode covered 0.5, alerts 32, false 1
- Zone check: conveyor_work_cell: 13 events; site mode covered 9+1 (own alert + in cooldown) = 0.769, strict recall 0.692, alerts 24, false 6, per alerted event 2.0; per-camera mode covered 0.692, alerts 18, false 0
- Zone check: conveyor_side_walkway: 3 events; site mode covered 3+0 (own alert + in cooldown) = 1.0, strict recall 1.0, alerts 3, false 0, per alerted event 1.0; per-camera mode covered 1.0, alerts 8, false 0

### Aggregate over the 6 test scenes (by zone kind; W013-W022 zones drawn blind by AI agents on coverage maps, W010 keeps the demo config's zones; kinds in docs/results/site_by_zone_kind.json)
- door/dock zones: 53/53 events covered (100 %), 0 of 52 alerts false
- forklift 3 m rule: 56/61 covered (92 %), 1 of 26 alerts false
- narrow lanes/aisles: 98/118 covered (83 %), 3 of 77 false (4 %)
- work areas: 323/338 covered (96 %), 26 of 362 false (7 %)
- 'other' zones: 43/48 (90 %), 6 of 44 false (14 %)
- frozen run (before the two alert-logic fixes, same detector and confidence settings): door 50/53 (94 %), 0 of 26 false; forklift 54/61 (89 %), 1 of 24; lanes 76/118 (64 %), 1 of 40; work 281/338 (83 %), 16 of 244; other 34/48 (71 %), 4 of 30
- 'covered' = an alert of its own OR inside the cooldown of an alert for the same person. Strict recall (an alert of its own), final run: door 52/53 (98 %), forklift 25/61 (41 %), lanes 71/118 (60 %), work 237/338 (70 %), other 34/48 (71 %). Frozen strict: door 49 %, forklift 38 %, lanes 33 %, work 43 %, other 44 %. The forklift rule's 60 s cooldown deliberately does not re-alert a worker who steps in and out of 3 m of a parked forklift.
- Test-scene caveats: W010 is the demo scene and the tracker was developed while watching it; W017 (alert-logic development) is the same room and cameras as test scene W018; the two alert-logic fixes were made after the frozen run had been seen (developed on W009/W012/W017/W020).

## Settings chosen on validation (W020) only
- worker shown when a camera sees it with confidence >= 0.7, kept while >= 0.6; vehicles need >= 0.6; forklift distance pad 0.1 m (bias -0.38 m -> 0.01 m on W020)
- 'in view' = >= 40 px tall at 1280x720 (person detection 68 % at 20-40 px vs 92 % at 40-80 px on W020)
- deployment spec: one UGen300 per area of <= 12 cameras; watched zones seen by >= 2 cameras at >= 40 px


## Other facts used in the deck (sources noted)
- Off-the-shelf detectors on the synthetic warehouse (Warehouse_010): YOLO-World found 0 of 30 forklift boxes; COCO YOLOv8s matched 7 as "truck" and took 39 % of the humanoid robots for people (scripts/prepare_site_dataset.py docstring). Hence the trained 6-class site detector.
- Site detector: YOLOv8s, 6 classes (person, forklift, pallet truck, transporter, mobile robot, humanoid robot), fine-tuned on 8,000 images from 8 SmartSpaces training scenes (W000, 002, 004, 006, 008, 011, 014, 016 cams 0-8), 960 px; never on any evaluation scene. Live: about 63-95 ms per 11-frame batch at 960 px on the RTX 5060 Ti.
- Hard-hat mode live: 10 cameras at a 10 fps target each; every frame analysed, 100 frames/s (every camera at 10.0 fps) through one batched detector (31-40 ms per 10-frame batch). The loop takes at most one frame per camera per 10 Hz tick, so 100 frames/s is the ceiling; the "about 105" shown earlier came from a counting error in the dashboard throughput metric (N batches over N-1 intervals), fixed 2026-10-19; estimated about 40 % of one UGen300 at the nominal 100 frames/s (Hailo reference YOLOv8s 252 FPS at batch 8; benchmark.md 39.7 %).
- UGen300 latency estimates (benchmark.md, not measured): spoken rule -> rule JSON about 11.3 s; Whisper transcription of a 5 s rule about 2.4 s; a full LLM-written daily report about 48 s.
- All LLM accuracy above was measured with Ollama qwen2.5:1.5b on the PC GPU; the Hailo GenAI build (different quantization) is not measured.
- Stage I runs everything on the PC (RTX 5060 Ti); nothing has run on a UGen300 yet. Only the hard-hat detector is compiled to a HEF (not run on hardware); CLIP was quantized and checked in Hailo's emulator only.
- CLIP INT8: accuracy 95.3 % only with the head retrained on INT8 features (violations kept 93.9 %); the PC head on INT8 features: 94.3 % (violations kept 92.9 %).
- The 0.939 mAP50 (Ultralytics val, score >= 0.001) and the Hailo 0.889/0.893 (score >= 0.2 NMS, pycocotools) are different evaluations: compare 0.889 only with 0.893.
- Yes/no check rates 95.7 % / 94.9 % / 95.3 % are probe_train.json (the shipped probe); yesno_check.json is the earlier comparison table (its probe row: 96.8 % / 93.9 %). The 94 'hard-hat' test crops are labelled dataset crops, a proxy for real false flags.
- Worker ids change within tens of seconds: 10.72 ids per person over the 5-minute W010 clip (about one per 28 s), 22.32 on W020.
- Training data of the site detector: 8 scenes from 5 buildings (W000/002/004/006 are one building).
- Settings analysis on W020: docs/results/w020_settings_analysis.json (worker-filter grid, forklift box confidence, distance pad, in-view px).
- The 12-camera limit is a coverage spec from the test scenes; warehouse-mode compute load on the UGen300 (960 px detector) is not measured.
- The rule-parsing sets are typed sentences; speech-to-text accuracy on recorded speech is not measured. Open-vocabulary detection and head-size distance in hard-hat mode are not measured.
- Hailo-10H references (Hailo Model Zoo / GenAI Model Zoo, from the existing deck): YOLOv8s 166 FPS batch 1, 252 FPS batch 8; YOLO-World v2-S 46.6 FPS; Whisper-Small 10.6 tokens/s; Qwen2.5-1.5B-Instruct 7.35 tokens/s, 0.37 s to first token; CLIP ViT-B/32 image encoder 81.5 FPS (batch 1).
- UGen300: Hailo-10H, 8 GB LPDDR4, USB-C 10 Gbps or M.2, 40 TOPS INT4 / 20 TOPS INT8, 2.5 W typical; Taiwan retail NT$6,380 (Oct 2026).
- Unit tests: 301 passing (pytest).
- Rule-parsing examples from the live server: "Alert me if more than two people are at the site gate after 10 pm" -> LLM said time window 22:00-00:00, the checker used the stated convention 22:00-06:00; "A區有人靠近挖土機三公尺內要立刻警告" -> LLM said object "excavation pit", the checker used "excavator". Rule parse 0.2-0.5 s.
- NOT in any result file (older notes only, do not quote as measured): YOLO-World 12 ms per frame, Whisper-Small 0.4-0.8 s per rule. Tables show '-' for these.
- CLIP yes/no check latency: 15.2 ms mean per crop (p95 24.7 ms), probe_train.json — use this, not the older 13.6/14 ms.
- The zones of the evaluation scenes (configs/eval/*.yaml) were drawn by AI agents that saw only each scene's coverage map, before any result.
- Jev-style alert check vs a generative VLM: 15.2 ms (probe_train.json) vs 283 ms (vlm_check.json) per crop, one crop at a time on the same PC GPU = 18.6×, written as "18×". Accuracy 95.3 % vs 91.7 % on the same 193 crops.
- Jev: TypeSafe AI's "System One" decision model (early access 2026-09-15); returns typed answers (yes/no "noul", choice, score) instead of text; hosted API only, no on-prem or weights. Design inspiration only; SayGuard does not call Jev.
- Hailo GenAI Model Zoo, Hailo-10H: Qwen2-VL-2B-Instruct 1.02 s to first token, 7.04 tokens/s, vision encoder (336×336) 3.00 FPS, so about 1 s per crop; CLIP ViT-B/32 image encoder 81.5 FPS.
