# 3-minute demo video — script and shot list (Stage I)

Target length 2:50, English narration (competition rule §6.1). Record the dashboard at
1920×1080 (OBS, display capture) and lay the voice-over on top. All numbers below were
measured on the Stage I PC prototype (`docs/results/`); UGen300 figures are estimates
from the Hailo Model Zoo and are labelled as such on screen.

| Time | Screen | Narration |
|---|---|---|
| 0:00–0:12 | Title card → dashboard, ten live tiles | "Construction sites have cameras everywhere and nobody watching them. SayGuard turns one ASUS UGen300 into a safety officer for every camera — and you set it up by talking to it." |
| 0:12–0:35 | Camera wall; point at the KPI strip | "Ten cameras, ten frames a second each, one detector. All streams are batched into a single YOLOv8s pass — 105 frames per second in total. On the UGen300 that is about 40 % of its published YOLOv8s capacity, so there is headroom for more cameras." |
| 0:35–0:55 | House Frame tile: red box "NO HARD HAT 2.1 s"; alert card appears, badge "Confirmed by VLM" | "A worker without a hard hat is only reported after two seconds across several frames. Then a 2-billion-parameter vision-language model looks at the crop as a second opinion before anyone is paged — in about 0.3 seconds." |
| 0:55–1:05 | Site Office tile with a bare head and no alert; zoom on a pixelated face | "The office is exempt, so no alarm there. And faces are pixelated on the device — video never leaves the site." |
| 1:05–1:40 | Hold the mic, say "A區有人靠近挖土機三公尺內要立刻警告"; transcript; proposed-rule card with "LLM + checker" badge; click Activate | "New hazard today? Just say the rule — in Mandarin or English. Whisper transcribes it, a 1.5-billion-parameter LLM turns it into a structured rule, and a deterministic checker verifies every number, distance and time against what was actually said. On unseen sentences that combination gets 95 % of rules exactly right, where the LLM alone gets 40 %." |
| 1:40–2:00 | Excavation Pit tile: orange "excavator" box, red distance line "0.5 m", alert | "No retraining: the excavator is found by an open-vocabulary detector from the word itself, and distance is estimated from the size of the worker's head. Seconds after speaking, the rule is live." |
| 2:00–2:15 | Type "Alert me if more than two people are at the site gate after 10pm"; card shows 22:00–06:00 and the note "LLM said 22:00–00:00 → using 22:00–06:00" | "When the model gets a detail wrong, the checker shows you what it corrected." |
| 2:15–2:35 | Click "Daily safety report" → report renders | "At the end of the shift SayGuard writes the safety report. Every number comes straight from the alert log; the on-device LLM drafts tomorrow's actions — like a toolbox talk at the worst camera before the peak hour." |
| 2:35–2:50 | Slide: PC prototype → UGen300 model map | "Every model here — detector, open-vocabulary detector, Whisper, Qwen2.5, Qwen2-VL — is in the Hailo Model Zoo for Hailo-10H. Stage II moves the whole stack onto one 2.5-watt UGen300 next to the site's existing recorder." |

## Before recording

- Start Ollama (`D:\ugen-work\tools\ollama\ollama.exe serve`) and `python -m sayguard serve`; open `http://127.0.0.1:8000/?lang=en` (or `?lang=zh`).
- Warm up: run one spoken rule (loads Whisper) and wait for the first hard-hat alert (loads the VLM).
- Delete test rules except "Hard hat required on site"; let it run ~15 minutes so the report has history.
- Stop other GPU-heavy programs (training jobs and the like) so the frame rate stays at 10 fps per camera.
