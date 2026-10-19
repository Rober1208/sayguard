English | [繁體中文](PITCH_OUTLINE.zh-TW.md)

# Stage I proposal deck: outline and talking points

The deck has 20 slides in English (rules §6.1 allow at most 20 pages of body). The slide sources
are in `docs/deck/project/slides/<id>.html`, and the order is in `docs/deck/project/deck.json`.
This outline follows the same order. For each slide it gives what is on the slide and one paragraph
of talking points. Every measured number is copied from [FACTS.md](FACTS.md) or the result files it
names. Other figures cite their source (the configs, the competition rules, or a derivation noted
in the text). The slide footers name the result file or reference behind the numbers.

Stage I runs everything on a PC GPU (RTX 5060 Ti); nothing has run on a UGen300 yet. In this
outline "on site" and "local" mean what runs today, and "on the UGen300 (Stage II)" means the plan.

Required sections (rules §6.1) are context and problem, solution (tools, technology, method),
hardware/software architecture, expected benefits and outputs, and references with the GitHub
link. The scoring weights in the competition rules decide the emphasis: business potential and
feasibility 35 %, innovation 30 %, practical value 25 %, storytelling 10 %.

| Required section | Slides |
|---|---|
| Context and problem | 1 cover, 2 problem |
| Solution: tools, technology, method | 3 idea, 4 demo, 5 llm, 6 voice, 8 report, 9 falsealarm, 10 site, 13 detector, 14 privacy |
| Hardware/software architecture | 15 hardware, 16 models |
| Expected benefits and outputs (with evidence) | 7 accuracy, 8 report, 9 falsealarm, 11 siteevidence, 12 sitelimits, 17 business, 18 stage2, 19 risks |
| References and GitHub link | 20 close |

## The thread through the deck: what the local LLM is for

Judges will see many detectors. What sets SayGuard apart is how it uses a small local LLM
(Qwen2.5-1.5B-Instruct, called through the Ollama API). In Stage I it runs with Ollama on the PC
GPU. In Stage II it moves to the UGen300, where hailo-ollama serves the same API. The LLM does two
jobs a detector cannot do. The principle is the title of slide 5: **the LLM proposes, code checks,
the officer confirms.** For rules this holds word for word. Nobody confirms the report plan, so
there the LLM may only choose: issues from the problems that actually happened, and fixes from a
vetted list.

1. **It turns a sentence into a rule, in the site's own words.** The model fills a JSON schema
   generated from SayGuard's rule model. A deterministic checker then overrides the numbers,
   distances, times, durations and objects that it finds stated in the sentence. It can miss a
   phrasing, which is one reason the officer reads the rule in plain words before it goes live. On
   typed held-out sentences (rules exactly right; speech recognition not included), hard-hat mode
   goes from 40 % with the LLM alone to 95 % with the checker. Warehouse mode goes from 16 % to
   84 % on the first blind set (its first run) and from 24 % to 88 % on a fresh one.
   The same job covers language. Rules can be spoken in Mandarin or English. Whisper is primed with
   the site's zone and camera names, and the LLM gets the zone names in both languages. The checker
   matches a zone name whether its number is written in digits, Chinese numerals or English words
   (七號走道, 7號走道, aisle seven), with or without a trailing 區. Simplified Chinese output is
   converted to Taiwan Traditional Chinese. Language is measured only inside the rule tests: every
   rule test set mixes Mandarin and English sentences. A rule takes 0.35–0.40 s on average to
   parse on the PC (max 0.64 s on the held-out sets). When the LLM is offline, the rule extractor
   (the deterministic part, which the checker also uses) parses rules alone: 90 % on the
   hard-hat-mode held-out set and 68 % on the fresh warehouse set. On that set it produced no rule
   for several rephrasings that the checked LLM draft got right, which is why the LLM stays in the
   loop.
2. **It plans the daily report.** Code computes every number, table and the summary. The model
   sees only the (camera, problem) pairs that actually happened, and for each pair three vetted
   fixes. It returns issue and fix numbers in a JSON schema, so it cannot name a camera that had
   no problem or invent a figure. Two blind AI judges per action found the wrong problem in 20.0 %
   of freely written actions, in 3.5 % when the model picked the issue and wrote the action itself
   (keyword-checked), and in 0.0 % with the shipped design.

All LLM accuracy figures were measured with Ollama qwen2.5:1.5b on the PC GPU. The Hailo GenAI
build of the same model uses a different quantization and is not measured yet. The LLM runs only
on demand, for a rule or a report, so it does not compete with the detectors on every frame. The
Hailo GenAI Model Zoo lists Qwen2.5-1.5B-Instruct at 7.35 tokens/s with 0.37 s to the first token
on Hailo-10H. From those rates, `docs/results/benchmark.md` estimates about 11.3 s from spoken rule
to rule JSON on the UGen300, and about 2.4 s for Whisper to transcribe a 5 s rule (estimates, not
measured). In the deck this thread covers slides 5–8. The demo video shows both jobs.

## Slide by slide

### 1 · cover: SayGuard

*On the slide:* kicker "ASUS UGEN AI LEAGUE · STAGE I PROPOSAL"; "One UGen300 per site area.
Every camera watched. Rules you can say."; chips "Battlefield Lightning · UGen300", "Workplace AI",
"[Team name] · [Onsite / Global Online track]".

SayGuard is a safety assistant for every camera in a site area, and you set it up by talking to it.
It has two modes. Hard-hat mode checks hard hats on every camera. Warehouse mode fuses all cameras
on one floor plan. Everything we show was measured on our working Stage I prototype, which runs on
a PC GPU. In Stage II it moves to one ASUS UGen300 per site area. UGen300 figures are estimates
from Hailo's published numbers until Stage II measures them on the device.

### 2 · problem: Sites have cameras. Few have anyone watching them live.

*On the slide:* three cards: "Found too late" (a missing hard hat may only be noticed in recorded
footage, after an incident); "Rules are fixed" (many AI cameras ship with a set list of rules, and a
new zone, distance, head-count or time rule means reconfiguring by hand); "Alarms get ignored"
(frequent false alerts wear down trust, and streaming site video to the cloud raises privacy and
bandwidth concerns).

There are three problems. First, few sites have anyone watching their cameras live, so a missing
hard hat may only be noticed in recorded footage after an incident. Second, many AI cameras ship
with a set list of rules and are set up camera by camera, so a new zone, distance, head-count or
time rule means reconfiguring by hand. Third, alarms that are often wrong wear down trust and get
ignored, and streaming site video to the cloud instead raises privacy and bandwidth concerns.

### 3 · idea: One box next to the recorder. Every camera covered.

*On the slide:* flow from the cameras the site already has (RTSP; an extra camera only where a
watched zone is seen by fewer than two; no cloud) to host + ASUS UGen300 (batched detectors, a CLIP
yes/no check, Whisper and Qwen2.5, all on site: PC GPU today, UGen300 in Stage II) to the safety
officer (checked alerts, a live floor plan, a daily report, new rules by voice). Two mode cards:
"Hard-hat mode: construction sites, hard hats on every camera" and "Warehouse mode: all cameras
fused on one floor plan, zones and forklift distances in metres".

This is the whole product in one picture. The site keeps the cameras it already has, plus a camera
wherever a watched zone is seen by fewer than two. We add a small host and one UGen300 next to the
recorder. Every model runs on site: on a PC GPU in the Stage I prototype, and on the UGen300 in
Stage II. The officer gets checked alerts, a live floor plan, a daily report and new rules by
voice. Both modes share this pipeline and differ only in what they watch: hard hats per camera in
hard-hat mode, or workers and vehicles on one shared floor plan in warehouse mode.

### 4 · demo: The working prototype

*On the slide:* the hard-hat dashboard screenshot; caption "10 looping demo streams, analysed live
· alerts double-checked locally · spoken rules · PC GPU today, UGen300 in Stage II".

This is the real dashboard at 1920 by 1080 (the screenshot's size), with ten looping demo streams,
each analysed at a target of ten frames per second. Red boxes are people without hard hats, and
faces are pixelated locally. Alerts carry the yes/no check's verdict. On the right, the officer
says or types a rule and confirms it. The same software runs on a PC GPU today and moves to the
UGen300 in Stage II.

### 5 · llm: A small LLM proposes. Code checks. The officer confirms each rule.

*On the slide:* kicker "INNOVATION · THE LOCAL LLM · TWO JOBS". Job 1, "Turns a sentence into a
rule": hard-hat mode 40 % → 95 %, warehouse mode 16–24 % → 84–88 % (typed held-out sentences, LLM
alone → with checker; the checker's rule extractor alone gets 90 % / 68–80 %). Job 2, "Plans the
daily report": wrong problem 20 % → 0 % (free text → vetted fix, 2 blind AI judges). A third card,
"Speaks like the site", covers language in both jobs (it is not a third job): Mandarin or English,
Taiwan wording, zone names as people say them ("七號走道", "aisle seven"); offline, the rule
extractor alone still parses 68–90 % of typed held-out rules. Bottom strip: "…after 10 pm" → LLM
22:00–00:00 → checker 22:00–06:00. Footer: measured with Qwen2.5-1.5B-Instruct via Ollama on the PC
GPU; the UGen300 build (hailo-ollama) is not measured yet.

This is our main innovation. A 1.5-billion-parameter model is not reliable enough to trust on its
own, so we never do. Hailo publishes a Hailo-10H build of it (7.35 tokens/s); running it alongside
the detectors is measured in Stage II. The model has two jobs. First, it turns a sentence into a
rule, in Mandarin or English, with the site's own zone names. Deterministic code overrides the
numbers, times and objects it finds stated in the sentence, and the officer confirms. On typed
held-out sentences this lifts accuracy from 40 % to 95 % in hard-hat mode, and from 16–24 % to
84–88 % in warehouse mode. The rule extractor alone handles the familiar phrasings (90 % /
68–80 %); the model adds the rephrasings it misses. Second, it plans the daily report. Code
computes every number; the model picks the three issues that matter and a vetted fix for each.
Nobody confirms that plan, so the model is only allowed to choose from those lists. Two blind AI
judges found the wrong problem in none of its actions, against 20 % when it wrote freely. The
example at the bottom is real: for "after 10 pm" the model said 22:00 to 00:00, and the checker
used the stated convention, 22:00 to 06:00. These accuracy figures were measured with Ollama on
the PC; the Hailo build is measured in Stage II.

### 6 · voice: Speak a rule, confirm it, it is live

*On the slide:* four steps, speak (Whisper-Small, site words as a decoder prompt) → draft
(Qwen2.5-1.5B, JSON under a fixed schema) → check (deterministic checker: overrides the numbers,
times and objects it finds stated) → confirm (the officer reads the rule in plain words and
switches it on), the note "LLM offline? The checker's rule extractor alone still parses 68–90 % of
typed held-out sentences", and the "Say a rule" panel screenshot.

The pipeline has four steps. Whisper-Small transcribes Mandarin or English, primed with the site's
zone and camera names; its accuracy on recorded speech is not measured yet. Qwen2.5-1.5B fills a
JSON schema generated from the rule model, so it answers in the rule's own fields. The checker
extracts what it finds stated explicitly in the sentence and wins every disagreement. It writes a
note whenever it overrides the LLM, so the officer sees what was corrected, and a phrasing the
checker misses is caught at the officer's confirmation. A rule that covers no camera cannot be
activated. When the LLM is offline, the rule extractor parses the rule alone (90 % on the
hard-hat-mode held-out set, 68 % on the fresh warehouse set). Parsing takes 0.35–0.40 s per rule on
average on the PC (max 0.64 s). On the UGen300 the estimate is about 11.3 s per rule, plus about
2.4 s to transcribe a 5 s rule (estimates from Hailo's published rates, not measured).

### 7 · accuracy: With the checker, a 1.5B model gets 84–95 % of held-out rules exactly right

*On the slide:* two groups of bars (LLM alone / rule extractor alone / LLM + checker). Hard-hat
mode, 20 typed held-out sentences: 40 %, 90 %, 95 %. Warehouse mode, 2 × 25 typed held-out
sentences: 16–24 %, 68–80 %, 84–88 %. A note: 84 % is the first set's first run, before the fixes
it exposed; 88 % is a fresh second set after the fixes. Footer: Ollama qwen2.5:1.5b on the PC GPU,
`scripts/eval_rules.py [--site]`, `docs/results/rule_parsing*.json`.

A rule counts as correct only if every labelled field matches: rule type, object, count or
distance, time window, duration when stated, and the cameras or zones it covers. The test
sentences are typed, in Mandarin and English; speech recognition is not part of this score. The
hard-hat-mode sentences were held out while the extractor was built. There the rule extractor
alone already gets 90 %, so the LLM's share is the rephrasings the extractor misses (95 %
together). For the warehouse, a writer and a reviewer, both AI agents that never saw the parser
code, wrote and labelled the sentences. We report their first held-out set's first, blind run:
84 % with the checker. That run exposed phrasing gaps, which we then fixed, so the set is no longer
clean. A new pair of blind AI agents, a writer and a reviewer, produced a fresh set that was never
used for tuning: 88 % with the fixed parser. On the warehouse sets the LLM alone stays at 16–24 %
and the rule extractor alone at 68–80 %. We ship the combination, and the officer confirms every
rule before it goes live.

### 8 · report: The daily report: code counts, the LLM plans

*On the slide:* three steps (numbers, tables and summary computed from the alert log; the LLM sees
only the (camera, problem) pairs that happened, with three vetted fixes for each kind; it returns
issue and fix numbers in a JSON schema). Table, "20 synthetic days · 2 blind AI judges per action",
with a note: the judges, AI agents, did not know which approach wrote an action and agreed on
"wrong problem" for 176 of 177 actions; 14 of the middle row's 57 actions fell back to template
text. The judges are AI agents, not people; say so when you present this slide.

| How the actions are written | Wrong problem (both judges) | Sensible (both judges) |
|---|---|---|
| LLM writes the actions freely | 20.0 % | 45.0 % |
| LLM picks the issue, writes the action (keyword-checked) | 3.5 % | 52.6 % |
| LLM picks the issue and a vetted fix (shipped) | 0.0 % | 100.0 % |

For the shipped design the wording of each fix is ours (vetted), so its "sensible" rates only the
choice of issue and fix (marked * on the slide). Footer: Qwen2.5-1.5B on the PC GPU,
`docs/results/report_actions_judged.json`, `report_actions_judging/` (script:
`scripts/eval_report.py`).

In testing, a 1.5B model writing the report itself misattributed counts and invented terms. So the
shipped design splits the work. Code computes every number, table and the summary from the alert
log. The model sees only the camera-and-problem pairs that actually happened, each with three
vetted fixes for that kind of problem, and it answers with numbers. It cannot name a camera that
had no problem, invent a figure, or prescribe hard hats for a crowding problem. Nobody confirms
this plan, and that is why the model may only choose. We tested this on 20 synthetic days. Two
independent AI judge agents rated each action, without knowing which approach wrote it, and they
agreed on whether it named the wrong problem for 176 of 177 actions. The 100 % "sensible" for the
shipped design rates the choice only, because we wrote the fixes. If the LLM is offline or its
answer is not valid, a template writes the actions, so the report always arrives.

### 9 · falsealarm: A second opinion before anyone is paged

*On the slide:* three gates: persistence, de-duplication, yes/no check; 95.7 % of hard-hat crops
correctly cleared (our proxy for wrong "no hard hat" flags), 94.9 % of violations confirmed,
15.2 ms per check on the PC; filtered alerts are never deleted. Footer: `scripts/train_probe.py`,
`docs/results/probe_train.json`.

There are three gates before a hard-hat alert is confirmed. First, a bare head must persist for
two seconds, and a short label history per track means one misclassified frame does not count.
Second, repeated alerts for the same person on the same camera are counted once. Third, a yes/no
check looks only at the crop around the flagged head. "Is this person wearing a hard hat?" is a
decision, not a conversation, so it does not need a generative model. A frozen CLIP ViT-B/32 image
encoder with a 2×512 decision head passes 95.7 % of 94 held-out crops of people wearing a hard hat
(labelled dataset crops, our proxy for wrong "no hard hat" flags) and confirms 94.9 % of real
violations (99 held-out bare-head crops), 95.3 % test accuracy in all (`probe_train.json`). It
takes 15.2 ms per crop on the PC. Because about 5 % of real violations (1 − 0.949) would still be
filtered, filtered alerts are never deleted: they stay in the feed, dimmed, for review. A
vision-language model remains available as an option (`verifier.backend: vlm`).

### 10 · site: Every camera on one floor plan

*On the slide:* the warehouse dashboard screenshot and short points: each worker counted once, not
once per camera; zones on the plan and distances to forklifts in metres; one alert per worker per
entry, not one per camera, lit in every camera that sees them; rules spoken with zone names.
Footer: `docs/results/site_eval.json`, synthetic warehouses, NVIDIA PhysicalAI-SmartSpaces
(CC BY 4.0).

Warehouse mode is where multi-camera fusion matters. Each camera's calibration maps the floor into
its image, so every detection becomes a position on one shared floor plan. Our scene is
Warehouse_010, the demo scene: eleven synchronized cameras in one of NVIDIA's synthetic warehouses.
On average 24.97 people are present. SayGuard counts 26.56, while adding up the cameras gives
51.64. Zones are drawn once on the plan instead of per camera. The distance from a worker to a
forklift is measured in metres, with a median error of 0.2 m on this scene. Each worker raises one
alert per entry, not one per camera, shown in every camera that sees them, and rules name zones the
way people say them. Worker ids are not shown: they change within tens of seconds (10.7 ids per
person over the 5-minute Warehouse_010 clip), so a number on screen would mislead.

### 11 · siteevidence: Six buildings the detector never trained on

*On the slide:* protocol line with the caveats (detector and confidence settings chosen on a
validation scene; zones drawn blind by AI agents except on the demo scene W010; W017 is W018's
twin); table "rule kind · covered · own alert · false alerts · frozen: covered · false"; the head
count error, fused against summed; a note with the definition of covered, the forklift cooldown
and the two alert-logic fixes. Footer: `scripts/eval_site.py`, `docs/results/site_eval*.json`,
`site_by_zone_kind.json`, `docs/results/frozen/`.

Final run, after two alert-logic fixes. "Covered" = an alert of its own, or inside the cooldown of
an alert for the same person. "Own alert" = strict recall.

| Rule kind (6 test scenes) | Events covered | Own alert (strict recall) | False alerts |
|---|---|---|---|
| Door / dock zones | 53 of 53 (100 %) | 52 of 53 (98 %) | 0 of 52 |
| Forklift 3 m rule | 56 of 61 (92 %) | 25 of 61 (41 %) | 1 of 26 |
| Narrow lanes / aisles | 98 of 118 (83 %) | 71 of 118 (60 %) | 3 of 77 (4 %) |
| Work areas | 323 of 338 (96 %) | 237 of 338 (70 %) | 26 of 362 (7 %) |
| Other zones | 43 of 48 (90 %) | 34 of 48 (71 %) | 6 of 44 (14 %) |

Frozen run, before the two fixes (same detector and confidence settings; the cleaner held-out
result):

| Rule kind | Events covered | Own alert (strict recall) | False alerts |
|---|---|---|---|
| Door / dock zones | 50 of 53 (94 %) | 49 % | 0 of 26 |
| Forklift 3 m rule | 54 of 61 (89 %) | 38 % | 1 of 24 |
| Narrow lanes / aisles | 76 of 118 (64 %) | 33 % | 1 of 40 |
| Work areas | 281 of 338 (83 %) | 43 % | 16 of 244 |
| Other zones | 34 of 48 (71 %) | 44 % | 4 of 30 |

The protocol is the point of this slide, and so are its caveats. The six test scenes (W010, W013,
W015, W018, W019, W022) are different buildings from the training scenes, and the detector never
trained on them. Two of them are not fully independent. W010 is our demo scene: the tracker was
developed while watching it, and its zones come from the demo config rather than being drawn
blind. W017, a development scene, is the same room and cameras as test scene W018. The confidence
and distance settings were chosen on the validation scene W020. The zones of the other five test
scenes were drawn by AI agents that saw only each scene's coverage map, before any result. An
event counts as covered when SayGuard raised an alert of its own for it, or the worker was inside
the cooldown of an alert already raised for them; strict recall counts only alerts of their own.
The forklift rule's 41 % strict recall is by design: its 60 s cooldown deliberately does not
re-alert a worker who steps in and out of 3 m of a parked forklift. Zones that fit none of the four
kinds ("other") have the highest false-alert rate, 6 of 44 (14 %).

The headline numbers are the final run, after two alert-logic fixes. We developed those fixes on
development and validation scenes (W009, W012, W017, W020) after we had seen the frozen run on the
test scenes. The frozen run, with the same detector and confidence settings, is the cleaner
held-out result: it covered 94 % of door events, 89 % of forklift events, 64 % of aisle events,
83 % of work-area events and 71 % of other-zone events. The fixes also raised the false alerts: in
work areas from 16 of 244 to 26 of 362, and in aisles from 1 of 40 to 3 of 77. Head-count error
across the six scenes is 0.05 to 1.74 people with fusion, against 1.76 to 60.09 when the cameras
are summed. Both runs are in the repository.

### 12 · sitelimits: Where it works, and the spec that follows

*On the slide:* coverage maps (orange = one camera, green = two or more cameras see a person
≥ 40 px tall). Warehouse_010: 11 cameras, floor seen twice, forklift rule 95 % covered, 43 % with
an alert of their own, no false alerts. Warehouse_020: 16 cameras with gaps, forklift rule 57 %
covered, 34 % with an alert of their own, 16 of 51 alerts false. The line "Coverage is the clearest
difference, not a proven cause". Three limits (forklifts missed in some buildings, no worker ids,
more than 12 cameras) and the spec line "Coverage spec: one UGen300 per area of ≤ 12 cameras ·
watched zones seen by ≥ 2 cameras at ≥ 40 px".

Coverage is the clearest difference we see between these two scenes, though we have not shown that
it is the cause. On Warehouse_010 almost the whole floor is seen by two or more cameras, and there
the forklift rule covered 95 % of events (43 % with an alert of their own, the rest inside the
cooldown) with no false alerts. Warehouse_020, our validation scene, is a large hall with 16 cameras
and big gaps. There the forklift rule covered only 57 % of events (34 % with an alert of their own),
and 16 of 51 alerts were false. W020 also differs in other ways: it has about twice the people
(49.02 against 24.97 on average), more cameras than the spec allows, and it is the same building as
training scenes W000–W007. Person detection on that scene was 68 % for people 20–40 px tall against
92 % at 40–80 px, which is where the 40-pixel requirement comes from. There are two further limits.
Forklifts were found in 98.6 % of forklift-frames on W010, but in only 16.9 % on W018 and 11.5 % on
W019 (79.3 % on W022); we have not established why. And person ids change within tens of seconds
(10.7 ids per person in 5 minutes on W010), so none is shown. The deployment spec follows from
coverage: one UGen300 per area of at most 12 cameras, with every watched zone seen by at least two
cameras at 40 px or more. The 12-camera figure is a coverage spec from the test scenes; the
warehouse-mode compute load on the UGen300 is not measured yet (Stage II week 1).

### 13 · detector: Detectors trained where it matters

*On the slide:* three cards, each with one figure: hard hats (0.939 mAP50; Hailo's emulator, a
separate evaluation: 0.889 INT8 vs 0.893 float); warehouse objects (12–99 % of forklift frames
found: 99 % W010, 79 % W022, 17 % W018, 12 % W019); anything else a rule names (YOLO-World found 0
of 30 labelled forklift boxes on Warehouse_010; the excavator alert is a demo, accuracy not
measured), and the small excavator snapshot. Footer: `docs/results/detector_metrics.json`,
`hailo_int8.json`, `site_eval*.json`.

SayGuard uses three detectors, each where it is needed. **Hard hats:** YOLOv8s fine-tuned on
Roboflow Hard Hats v2 (CC BY 4.0). It reaches mAP50 0.9393 on 2,001 test images it never saw,
with 0.9008 recall on bare heads, the violation class. It is compiled to a Hailo-10H HEF, which has
not run on hardware yet. In Hailo's bit-exact emulator the INT8 model scores mAP50 0.889 against
0.8927 for the float graph, both measured the same way. The 0.9393 comes from Ultralytics' own
evaluator, a different evaluation, so compare 0.889 only with 0.8927.
**Warehouse objects:** YOLOv8s with six classes (person, forklift, pallet truck, transporter,
mobile robot, humanoid robot). It was fine-tuned on 8,000 images from 8 synthetic training scenes
in 5 buildings, at 960 px, and never on an evaluation scene. It found forklifts in 98.6 % of
forklift-frames on Warehouse_010, but in only 16.9 % on W018, 11.5 % on W019 and 79.3 % on W022;
we have not established why. Robots have classes of their own, so they are not lumped in with
workers. **Anything else a rule names:** YOLO-World, from the word itself. That is how the
excavator demo works; open-vocabulary accuracy is not measured. On the synthetic warehouse
YOLO-World found 0 of 30 forklift boxes, and COCO YOLOv8s matched only 7 as "truck" and took 39 %
of the humanoid robots for people. So the classes a site depends on get a trained detector, and
open vocabulary covers everything else.

### 14 · privacy: Video never leaves the site.

*On the slide:* statement slide on the accent colour.

Privacy is designed in, not added. Every model runs on site: on the PC GPU today, on the UGen300 in
Stage II. That covers detection, speech, language and the yes/no check. Faces are pixelated in
every detected person box, in every tile and snapshot, using boxes the detector already finds, so
no extra model is needed. A person the detector misses is not blurred, and that miss rate is not
measured. Only event snapshots are stored, and they are deleted after 14 days by default
(`retention_days` in the configs). In warehouse mode no worker id is shown.

### 15 · hardware: One 2.5 W accelerator per site area

*On the slide:* flow from IP cameras / NVR to host to ASUS UGen300 to officer. Two boxes: Stage I
prototype (PC GPU, hard-hat mode: 10 cameras at a 10 fps target, about 105 frames/s measured
through one batched detector; estimate at the nominal 100 frames/s: 40 % of one UGen300) and
Stage II (detectors and GenAI move to the UGen300; one box per area of ≤ 12 cameras, a coverage
spec; warehouse-mode load measured in week 1). Footer: UGen300 figures from the ASUS product
specification; load estimate from `docs/results/benchmark.md`.

The hardware is deliberately simple: the cameras the site already has (plus one wherever a watched
zone is seen by fewer than two), a small host, and one UGen300 (Hailo-10H, 8 GB LPDDR4, USB-C
10 Gbps or M.2, 40 TOPS INT4 / 20 TOPS INT8, 2.5 W typical). In the Stage I prototype, ten cameras
at a target of ten frames a second gave about 105 frames a second, measured, through one batched
detector. At the nominal 100 frames a second, against Hailo's YOLOv8s reference of 252 FPS at batch
8, that is an estimated 40 % of one UGen300. In Stage II the detectors and the GenAI models move
onto the UGen300. The host keeps decoding, tracking, rules and the dashboard. One box per area of
at most 12 cameras is a coverage spec from the warehouse test scenes; the warehouse-mode compute
load on the UGen300 is not measured yet and is measured in Stage II week 1. An area with more than
12 cameras gets a second box.

### 16 · models: Every job maps to a Hailo-10H model

*On the slide:* table of job · model · PC prototype (measured) · Hailo-10H (Hailo reference),
then one line on what has been through Hailo's emulator and one line of UGen300 estimates. "–"
means not in any result file.

| Job | Model | PC prototype (measured) | Hailo-10H (Hailo reference) |
|---|---|---|---|
| Hard hats, all cameras | YOLOv8s, fine-tuned | 31–40 ms per 10-frame batch | 252 FPS at batch 8 |
| Warehouse objects | YOLOv8s, 6 classes, 960 px | 63–95 ms per 11-frame batch | YOLOv8s family (HEF in Stage II) |
| Objects named in rules | YOLO-World v2-S | – | 46.6 FPS |
| Speech to text | Whisper-Small | – | 10.6 tokens/s |
| Rules and report plan | Qwen2.5-1.5B-Instruct | 0.35–0.40 s mean per rule, max 0.64 s | 7.35 tokens/s, 0.37 s to first token |
| Yes/no check on alerts | CLIP ViT-B/32 + 2×512 head | 15.2 ms per crop (one per alert) | 81.5 FPS (batch 1) |

UGen300 estimates on the slide (from Hailo's published rates, not measured): spoken rule → rule
JSON about 11.3 s; a 5 s Whisper transcription about 2.4 s. Hard-hat mode at 10 cameras uses about
40 % of the UGen300 at the nominal 100 frames/s (slide 15); the warehouse-mode load is measured in
Stage II week 1. Hailo also lists YOLOv8s at 166 FPS at batch 1. Footer: PC: RTX 5060 Ti,
`docs/results/`; estimates: `docs/results/benchmark.md`; Hailo-10H: Hailo Model Zoo, GenAI Model
Zoo.

This table is the feasibility argument: every job maps to a model that Hailo already publishes for
Hailo-10H, with published throughput. Only one network is compiled so far. The hard-hat detector
is a HEF, not yet run on hardware. In Hailo's bit-exact emulator its INT8 model scores mAP50 0.889
against 0.8927 for the float graph (compare these two only; the 0.9393 PC score is a different
evaluation). The CLIP image encoder was quantized and checked in the same emulator. With the 2×512
head retrained on INT8 features, the yes/no check keeps 95.3 % accuracy (violations kept 93.9 %);
the PC head on INT8 features gets 94.3 %. The LLM is called through the Ollama API, which
hailo-ollama serves, so moving it onto the UGen300 should only change `llm.base_url`. Stage II
checks that on the device and measures the Hailo build, whose accuracy may differ from the Ollama
build we measured on the PC. The UGen300 figures are estimates from Hailo's published rates, not
measurements. The language models run only on demand; the detector runs on every frame, and the
yes/no check once per hard-hat alert.

### 17 · business: A box per site area, a subscription per site

*On the slide:* who buys, what we sell (a box per area of up to 12 cameras, a coverage spec), why
it pays (10 hard-hat cameras use about 40 % of one NT$6,380 accelerator, an estimate). Footer:
Taiwan retail list price of the UGen300, Oct 2026; load estimate from `docs/results/benchmark.md`;
[ ] = to be decided.

The customer is a construction contractor, factory or warehouse that already owns cameras and must
enforce safety rules. The buyer is the safety officer or EHS manager. We sell a SayGuard box (a
UGen300 plus a small host, installed next to the recorder) and a per-site software subscription
[NT$__ / month], through CCTV installers and in bundles with ASUS IoT hardware. The economics: the
UGen300 retails at NT$6,380 in Taiwan, one per site area. In hard-hat mode ten cameras use an
estimated 40 % of it, which works out to about NT$638 of accelerator per camera at ten cameras
(derived; the host is extra). It draws 2.5 W typical. There are no per-camera cloud fees, and video
stays on site. A new rule about known objects, zones, distances, counts or times is one sentence,
not a project; a new object class may need training.

### 18 · stage2: Four weeks on the UGen300

*On the slide:* four week cards and the line "Already done in Stage I". Footer: Stage II dates.

Stage II is a port, not a rebuild. Week 1: run the hard-hat HEF through the HailoRT backend for the
first time, measure batch-8 throughput over USB, compile the warehouse detector HEF and measure the
warehouse-mode load. Week 2: GenAI on the device, meaning the yes/no check with its head retrained
on device features, Qwen2.5 on hailo-ollama with the rule tests re-run, Whisper, and the YOLO-World
HEF. Week 3: the ASUS HQ test space. We calibrate a multi-camera floor plan, tune thresholds, and
measure alert latency, power and device load. Week 4: the finals package. Already done in Stage I:
both detectors trained, the hard-hat HEF compiled and checked in INT8 in Hailo's emulator, the full
pipeline and dashboard on the PC, warehouse mode tested on six test buildings the detector never
trained on, 264 unit tests, and every evaluation script in the repository.

### 19 · risks: What could go wrong, and the plan

*On the slide:* a table of six risks and their mitigations. Footer: test rates from
`docs/results/probe_train.json`, `rule_parsing*.json`, `site_eval_warehouse_018/019.json`.

| Risk | Mitigation |
|---|---|
| USB throughput below Hailo's PCIe reference; warehouse-mode load not measured yet | Measure both in week 1; batch 8; 5 fps per camera if needed |
| The small LLM misreads a rule, or is offline | The checker (the rule extractor) overrides the numbers, times and objects it finds; the officer confirms; the extractor alone gets 90 % (hard-hat mode) / 68 % (warehouse) of held-out rules |
| The yes/no check filters a real violation (5 % in the PC test: 1 − 0.949, `probe_train.json`) | Filtered alerts are kept and shown dimmed |
| Forklifts missed in some buildings (found in 17 % and 12 % of forklift-frames on W018 and W019; cause not established) | Fine-tune the site detector on the customer's own vehicles (amount of data to be measured in Stage II) |
| Large halls, distant cameras | Coverage spec: one box per area of at most 12 cameras; every watched zone seen by two or more cameras |
| Detectors and GenAI share one accelerator | GenAI runs only on demand; HailoRT scheduler load measured in week 3 |

Each open risk has a measurement and a fallback. The device-side risks are the ones we cannot close
on a PC: USB throughput, the warehouse-mode load, and one accelerator shared between detectors and
GenAI are all measured in Stage II, with batch 8, 5 fps per camera and on-demand GenAI as the
fallbacks. The model-side risks are handled by the design. The checker and the officer catch a
misread rule, filtered alerts stay visible, and the deployment spec keeps large halls within the
coverage we measured. The forklift misses are the one detector limit we could not fix in Stage I,
because retraining on a test building would have used up a test scene. We have not established their
cause, and how much customer data a fine-tune needs is not measured.

### 20 · close: SayGuard. Every camera watched. Most new rules one sentence away.

*On the slide:* links (code: github.com/Rober1208/sayguard; demo video: [YouTube link];
team: [names] · [email]) and references: Hailo Model Zoo and Hailo GenAI Model Zoo (Hailo-10H),
ASUS UGen300 specification, Roboflow Hard Hats v2 (CC BY 4.0), NVIDIA PhysicalAI-SmartSpaces
(CC BY 4.0), Ultralytics YOLOv8 and YOLO-World (AGPL-3.0), OpenAI Whisper and CLIP (MIT),
Qwen2.5 (Apache-2.0), Mixkit stock footage (Mixkit Stock Video Free License).

Thank you. A new rule about known objects, zones, distances, counts or times is one sentence away.
The code, the evaluation scripts and the results behind every measured number in this deck are in
the repository, and [FACTS.md](FACTS.md) lists each measured number with its source file.
Estimates are labelled as estimates.

## Questions to prepare for

- **Is the warehouse footage real?** No. It is synthetic: NVIDIA PhysicalAI-SmartSpaces (CC BY
  4.0). We chose it because it has synchronized, calibrated cameras and ground truth for every
  person's floor position, which is what makes the measurements possible. In Stage II we calibrate
  a real multi-camera floor plan at the test space. The hard-hat footage is free Mixkit stock clips.
- **Why not a larger model, or a VLM for everything?** Hailo publishes a Hailo-10H build of the 1.5B
  model (7.35 tokens/s); running it alongside the detectors is measured in Stage II, and we estimate
  about 11.3 s per rule on the UGen300. The checker plus the officer's confirmation make it usable
  (40 % → 95 % on typed held-out sentences in hard-hat mode, 16–24 % → 84–88 % in warehouse mode,
  measured with Ollama on the PC). The yes/no question needs no generative model: the CLIP head
  reaches 95.3 % test accuracy at 15.2 ms per crop (`probe_train.json`).
- **What happens when the LLM is wrong, or offline?** The checker overrides the numbers, times and
  objects it finds stated in the sentence and shows a note, and the officer confirms before a rule
  goes live. Offline, the rule extractor parses rules alone (90 % on the hard-hat-mode held-out
  set, 68 % on the fresh warehouse set), and the report uses its template.
- **Did you tune on the test scenes?** Not the detector or the confidence settings: the confidence
  and distance settings were chosen on the validation scene W020. The two alert-logic fixes were
  developed on development and validation scenes (W009, W012, W017, W020) after we had seen the
  frozen run on the test scenes, and W017 is the twin of test scene W018. W010 is the demo scene:
  the tracker was developed watching it, and it uses the demo's zones. The zones of the other five
  test scenes were drawn by AI agents from coverage maps before any result. Retraining the detector
  was not possible without using up a test scene, so we did not retrain it. We report both the
  frozen run, which is the cleaner held-out result, and the final run.
- **What is not measured yet?** Throughput, latency and power on the UGen300 itself, including the
  warehouse-mode load (Stage II). Rule-parsing and report accuracy with the Hailo build of
  Qwen2.5-1.5B. Speech recognition accuracy on recorded spoken rules (the rule tests are typed).
  Open-vocabulary detection accuracy, and the head-size distance in hard-hat mode. How often the
  face pixelation misses a person. Real, non-synthetic warehouse footage. The warehouse detector
  HEF, which is not built yet. Identity over time: ids change within tens of seconds, so none is
  shown.
- **How many cameras per box?** One UGen300 per area of at most 12 cameras, a coverage spec from
  the warehouse test scenes. Hard-hat mode at ten cameras with a 10 fps target is an estimated 40 %
  of one UGen300 (at the nominal 100 frames/s). The warehouse-mode load is measured in Stage II
  week 1.
