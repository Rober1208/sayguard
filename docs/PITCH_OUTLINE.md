English | [繁體中文](PITCH_OUTLINE.zh-TW.md)

# Stage I proposal deck: outline and talking points

The Stage I deck has 19 body slides and 2 appendix slides, all in English (the rules allow up to
20 pages of body; the appendix does not count). The slide sources are in
`docs/deck/project/slides/<id>.html`, and their order is in `docs/deck/project/deck.json`. This
outline follows the same order: for each slide, what is on it and one paragraph of talking points.
Every measured number comes from the result files in [docs/results/](results/), summarised in
[FACTS.md](FACTS.md). Stage I runs on a PC GPU with the same models planned for the UGen300, so
UGen300 figures in the deck are labelled as Hailo references or our estimates; the limits are on
slide 18.

## Required sections and scoring

The rules (§6.1) require context and problem, solution (tools, technology, method),
hardware/software architecture, expected benefits and outputs, and references with the GitHub
link. The scoring weights set the emphasis: business potential and feasibility 35 %, innovation
30 %, practical value 25 %, storytelling 10 %.

| Required section | Slides |
|---|---|
| Context and problem | 1 cover, 2 problem |
| Solution: tools, technology, method | 3 idea, 4 demo, 5 llm, 6 falsealarm, 7 voice, 8 report, 9 site, 11 detector, 12 privacy |
| Hardware/software architecture | 13 hardware, 14 models, appendix A1 |
| Expected benefits and outputs (with evidence) | 6 falsealarm, 7 voice, 8 report, 10 siteevidence, 15 business, 16 outcomes, 17 stage2, 18 risks |
| References and GitHub link | 19 close, appendix A2 |

## The thread through the deck: decide, don't chat

Jev is the "System One" model that TypeSafe AI opened for early access on 15 September 2026. It
starts from a simple observation: much of what software asks an AI is a decision, not a
conversation. Jev answers typed questions, such as a yes/no probability, a choice or a score, and
never writes free text; it is offered only as a hosted cloud API, with no weights and no
on-premises option.

SayGuard does not use Jev. We borrowed the idea and rebuilt it from small models that Hailo
already supports on the Hailo-10H, so every decision the AI makes in the product is a typed
answer. The idea shows up in three places:

1. **Yes/no: the alert check (slide 6).** "Is this person really without a hard hat?" Our first
   double-check was a generative vision-language model, Qwen2-VL-2B. We replaced it with a frozen
   CLIP ViT-B/32 image encoder and a 512→2 decision head: 15.2 ms vs 283 ms per crop on the same
   PC GPU (18× faster), and 95.3 % vs 91.7 % correct on the same 193 held-out crops.
2. **A typed form: spoken rules (slide 7).** The LLM (Qwen2.5-1.5B) does not write an answer. It
   fills a fixed rule form, and code checks every number, time and object against the sentence.
   Rules exactly right on held-out typed sentences, LLM alone → with the code check: 40 % → 95 % in
   hard-hat mode, 16–24 % → 84–88 % in warehouse mode.
3. **A choice: the daily report (slide 8).** Code counts everything from the alert log. The LLM
   only chooses which real issues matter most and which vetted fix goes with each. Actions aimed
   at the wrong problem: 20 % with free text, 0 % with choices.

The second and third are the LLM's special jobs, and the heart of the pitch: turning one spoken
sentence, in Mandarin or English, into a live rule, and planning tomorrow's safety work from
today's alerts. No detector can do either. Both work because the LLM answers inside a fixed shape
and code checks the result; for rules, the officer also confirms before anything goes live.

Typed answers suit the edge. They are short, so they are fast. They are bounded, so a model cannot
invent a camera, a problem or a number. And code can check them. That is why small models are
enough: a 1.5-billion-parameter LLM and a 512→2 decision head on a 2.5 W accelerator, instead of a
large model in the cloud. The LLM also runs only on demand, when someone speaks a rule or the daily
report is planned. Slide 5 states the idea, slides 6–8 show it at work, and slide 18 comes
back to it: a misread rule or a filtered alert stays visible instead of failing silently.

## Slide by slide

### 1 · cover: SayGuard

*On the slide:* "ASUS UGen AI League · Stage I proposal", the name, and the line "One UGen300 per
site area. Every camera watched. Rules you can say." Chips for Battlefield Lightning · UGen300,
Workplace AI, and the team name 逼寶隊.

SayGuard is designed to put one ASUS UGen300 next to the video recorder of each site area and
watch every camera in it. Safety officers set the rules by saying them, in Mandarin or English.
Like TypeSafe AI's Jev, every decision an AI model makes in SayGuard is a typed answer instead of
text, which keeps it fast enough for a 2.5-watt chip and easy for code to check. Today the Stage I prototype
runs the same software on a local PC GPU. Every figure in this deck comes from that working
prototype or from our own INT8 checks in Hailo's emulator, unless it is labelled as a Hailo or ASUS
reference or as an estimate.

### 2 · problem: Sites have cameras. Few have anyone watching them live.

*On the slide:* Three figures: 105 construction workers died at work in Taiwan in 2025, the most of
any industry, 2 in 3 from falls; 30 % of Taiwan's forklift accidents are a person struck by the
truck, the most common kind (2011–2024); and after 20 minutes of watching screens, a person's
attention falls "well below acceptable levels". Below them, three cards. "Found too late": a missing
hard hat may only be noticed in recorded footage, after an incident. "Rules are fixed": many AI
cameras ship with a set list of rules, and a new zone, distance or time rule means a vendor project.
"Alarms get ignored": frequent false alerts wear down trust, and sending site video to the cloud
raises privacy concerns. The footer cites the Taiwan Ministry of Labor (Jan 2026; forklift study Apr
2025) and a Sandia / NIJ security guide (1999), with links in appendix A2.

Construction sites and warehouses already have cameras, but few have anyone watching them live. The
stakes are real: in 2025, 105 construction workers died at work in Taiwan, more than in any other
industry, two in three of them from falls, according to the Ministry of Labor. Its institute's study
of forklift accidents found that a person struck by the truck is the most common kind, about 30 %.
And a person watching screens cannot keep up: a Sandia guide for the US Justice Department notes
that after only 20 minutes, attention falls well below acceptable levels. So a missing hard hat is
often noticed only in recorded footage, after an incident. Many AI cameras ship with a set list of
rules, so a new zone, distance or time rule means a vendor project. And alarms that are often wrong
tend to be ignored, while sending site video to the cloud raises privacy concerns. SayGuard answers
all three: live coverage of every camera, rules anyone can add by speaking, and alerts worth
reading.

### 3 · idea: One box watches every camera on site

*On the slide:* Five feature cards: every camera live (one detector for all streams; in a warehouse,
one live floor plan), rules you can say (live in seconds, no retraining for objects it already
knows), alerts people trust (a fast yes/no double-check; one worker, one alert), a daily safety
report (counts from the alert log, the LLM picks the top problems and vetted fixes), and video that
stays on site. A sixth card names the two modes today: hard-hat mode and warehouse mode.

This is the product. One small box sits next to the site's video recorder and watches every camera
at once. The safety officer adds rules simply by saying them, every hard-hat alert is double-checked
before anyone is paged, the daily safety report writes itself, and no video leaves the site. It runs in
two modes today. Hard-hat mode covers construction sites. Warehouse mode fuses all cameras into
one live floor plan, with zones and forklift distances in metres. The rest of the deck takes these
features one at a time.

### 4 · demo: The working prototype

*On the slide:* A full-HD screenshot of the dashboard in hard-hat mode, captioned "10 camera
streams analysed live · every alert double-checked · rules by voice · PC GPU today, UGen300 in
Stage II". The demo footage is Mixkit stock video.

This is the real dashboard in hard-hat mode. Ten looping site videos are analysed live. Red boxes
mark people without a hard hat, and faces are pixelated locally. Each alert shows whether the
double-check confirmed it, and the "Say a rule" panel is where the officer speaks or types a new
rule and switches it on. Everything you see runs on site, with no cloud service involved. Today it
runs on a PC GPU; in Stage II the same software runs on the UGen300.

### 5 · llm: Decide, don’t chat: every answer is typed

*On the slide:* A banner explains the Jev idea and why we rebuilt it with small models for a 2.5 W
UGen300. Three cards: the yes/no alert check, 18× faster (15 ms vs 283 ms per alert) and more
accurate (95.3 % vs 91.7 %); the typed rule form, 40 % → 95 % of rules exactly right (warehouse
16–24 % → 84–88 %); and the choice-based daily report, wrong-problem actions 20 % → 0 %.

This idea runs through all of SayGuard. TypeSafe AI's Jev showed that much of what software asks
an AI is not a conversation but a decision: a yes or no, a choice from a list, a typed form. Jev
is a cloud service and a safety box has to decide on site, so we rebuilt the idea from small
models Hailo already publishes for the Hailo-10H. For the alert check we replaced our generative vision-language model
with an image encoder and a two-way decision head: about eighteen times faster, and more accurate.
The language model fills a fixed rule form that code checks, and for the daily report it only
chooses among real issues and vetted fixes. Because every answer is typed, code can verify it, and
that is why a 1.5-billion-parameter model is enough.

### 6 · falsealarm: A Jev-style check, 18× faster than a VLM

*On the slide:* Bars compare our check (CLIP encoder + 512→2 head) with Qwen2-VL-2B on the same
193 held-out crops: 15.2 ms vs 283 ms per alert, 95.3 % vs 91.7 % correct, 95.7 % vs 89.4 % of
wrong "no hard hat" flags cleared, 94.9 % vs 93.9 % of real violations confirmed. Beside them sit
the three gates before anyone is paged (it must persist, one worker one alert, a yes/no check) and
the Hailo-10H outlook: CLIP's encoder at 81.5 FPS vs about 1 s per crop for Qwen2-VL-2B, with INT8
keeping 95.3 % in Hailo's emulator once the head is retrained on INT8 features.

This is where we applied the Jev idea first. Before anyone is paged, a violation has to persist,
repeats count once, and then a double-check looks only at the crop and answers one question: hard
hat, yes or no. Our first double-check was Qwen2-VL-2B, a two-billion-parameter generative model.
Jev cannot run on site, so we built a model in its style: a frozen CLIP image encoder and a small
trained head that only answers yes or no. On the PC it is about eighteen times faster, and more
accurate on the same crops. On the Hailo-10H the gap grows: by Hailo's figures the encoder runs at
81.5 frames per second, against about a second per crop for Qwen2-VL-2B. The encoder is already in
the Hailo Model Zoo. Quantized to INT8, it keeps the same accuracy in Hailo's emulator once the
head is retrained on INT8 features; compiling it for the device is Stage II work. Filtered alerts
are never deleted; they stay in the feed, dimmed.

### 7 · voice: Say a rule, confirm it, it is live

*On the slide:* Four steps: speak (Whisper, Mandarin or English), fill the form (Qwen2.5-1.5B
fills a fixed rule form, not free text), check (code corrects every number, time and object it
finds), and confirm (the officer reads the rule in plain words and switches it on). Rules exactly
right on typed sentences never seen in development: 95 % in hard-hat mode (LLM alone 40 %) and
84–88 % in warehouse mode (LLM alone 16–24 %), and 90 % / 74–86 % when the same sentences are
spoken by synthetic voices over site noise, next to a screenshot of the Say a rule panel.

The officer holds the button and says the rule. Whisper turns it into text, primed with the site's
own zone and object names. The small language model does not write an answer; it fills a fixed
rule form, and because the form is typed, code can check it: every number, time and object stated
in the sentence is corrected if the model got it wrong. The officer then reads the rule in plain
words and switches it on. On typed sentences never used in development, this gets 95 % of
construction-site rules and 84–88 % of warehouse rules exactly right, against 40 % and 16–24 % for
the LLM alone. A new rule about known objects, zones, distances, counts or times needs no data and
no retraining. We also spoke the same held-out sentences with synthetic Taiwan Mandarin and US
English voices over synthetic site noise: 90 % of construction-site rules and 74–86 % of warehouse
rules still came out exactly right. Most misses are words Whisper hears differently, such as a zone
letter B written as a similar-sounding character; those fixes are Stage II work on new sentences.

### 8 · report: The daily safety report writes itself

*On the slide:* Three steps: code counts every number from the alert log, the LLM chooses the top
real problems and one vetted fix for each, and the officer reads one page. Beside them, free text
vs choice: 20 % vs 0 % of actions aimed at the wrong problem, over 20 synthetic days and 60
actions per approach, rated by 2 AI judges who did not know which approach wrote each action.

The second special job of the small language model is the daily safety report, and again it makes
a choice instead of writing prose. Code computes every number and table from the alert log. The
model sees only the problems that really happened that day, with a short list of vetted fixes for
each kind, and answers with numbers: which problems matter most and which fix goes with each. We
compared this with letting the model write actions freely. Across twenty synthetic days, judged
blind by two AI judges, free text aimed one action in five at the wrong problem; the choice-based
plan never did. The model cannot invent a camera, a problem or a number, so the report can go out
without anyone rewriting it.

### 9 · site: Every camera on one floor plan

*On the slide:* The warehouse-mode dashboard on the 11-camera demo warehouse. Each worker is
counted once: on average 25 people present, 26.6 counted, against 51.6 when cameras are summed. Zones are
drawn on the plan, forklift distances are in metres with positions within about 0.2 m, each worker
gets one alert per entry, and rules use zone names ("aisle 7" or "七號走道").

This is warehouse mode on the synthetic eleven-camera warehouse we use as our demo scene. Instead
of judging each camera on its own, SayGuard places every detection on one floor plan, so a worker
seen by several cameras is one dot and one count, and each entry into a zone raises one alert, not
one per camera. Summing cameras counts many workers twice or more; the floor plan stays close to
the truth. Zones are drawn once on the plan, forklift distances are measured in metres, and the
officer can speak a rule using the zone names the site already uses, in Mandarin or English.

### 10 · siteevidence: Tested on six synthetic warehouses the detector never trained on

*On the slide:* Three headline figures: head count 0.05–1.7 people off on average (summing
cameras: up to 60 off), 100 % of door and dock entries alerted with no false alerts, and 92 % of
forklift close calls alerted with 1 false alert in 26. A table by zone kind adds work areas (323
of 338 alerted, 7 % false) and narrow aisles (98 of 118 alerted, 4 % false). A note below it
defines alerted as an alert or inside its cooldown, and says the figures follow two alert-logic
fixes; the frozen run gave doors 94 % and forklift 89 %.

We tested warehouse mode on six synthetic warehouses the detector never trained on, with every
setting chosen on a separate validation scene. One of them, Warehouse 010, is our demo scene, which
the tracker was developed on, so it is not a blind test. Fusing all cameras on one floor plan keeps
the head count within about two people of the truth, where summing the cameras can be off by sixty.
Every entry through a door or dock zone was alerted with no false alerts, and the forklift rule
caught 92 % of close calls. An event counts as alerted when the worker got an alert or was still
inside the cooldown of one; the strict numbers, and a run with the settings frozen before any fix,
are in the repository's results folder.

### 11 · detector: Trained detectors, plus words for the rest

*On the slide:* Hard hats: 0.939 mAP50 on 2,001 test images it never saw, from a YOLOv8s already
compiled for Hailo-10H (INT8 0.889 vs float 0.893 in Hailo's emulator). Warehouse objects: a
6-class YOLOv8s for people, vehicles and robots, trained on 8,000 frames from five synthetic
buildings and also compiled for Hailo-10H (emulator INT8 0.947 mAP50 = float). For anything else
a rule names, such as an excavator, YOLO-World finds objects from the word itself.

Each class a site depends on every day gets a trained detector. The hard-hat model scores 0.939
mAP50 on test images it never saw, and it is already compiled for Hailo-10H, where INT8 barely
changes its accuracy in Hailo's emulator. The warehouse model knows six classes of people,
vehicles and robots, and it is compiled for the chip too, with INT8 matching float in the emulator.
For anything else a spoken rule names, like the excavator in this snapshot, YOLO-World works from
the word alone. That part is a demo and its accuracy is not measured yet; in
our warehouse tests it was no substitute for the trained forklift detector, which is why everyday
classes get a trained model.

### 12 · privacy: Video never leaves the site.

*On the slide:* Every model runs on site (detection, speech, language and the yes/no check), on a
local PC today and the UGen300 in Stage II. Faces are pixelated from boxes the detector already
finds, with no extra model. Only event snapshots are kept, deleted after 14 days by default.

Every model runs on site: detection, speech recognition, the language model and the yes/no check.
That means a local PC in the prototype and the UGen300 in Stage II, so no video goes to the cloud.
Faces are pixelated using the boxes the detector already produces, which costs no extra model, and
the pixelation applies on screen and in saved snapshots alike. Only event snapshots are stored,
and they are deleted after fourteen days unless the site sets a different period.

### 13 · hardware: Plugs into the cameras a site already has

*On the slide:* A flow from IP cameras / NVR (the RTSP streams the site already records) to a mini
PC (decoding, tracking, floor plan, rules, dashboard) to the ASUS UGen300 (every neural network,
USB-C, 2.5 W) to the safety officer. Two figures: about 40 % of one UGen300 for ten hard-hat
cameras, all 100 frames/s through one batched detector (estimate), and zero cameras
replaced, with one box per area of up to 12 cameras.

Integration is deliberately simple. SayGuard reads the RTSP streams a site already records, so no
camera or recorder is replaced. A mini PC decodes the video, tracks workers, places them on the
floor plan, applies the rules and serves the dashboard, and a UGen300 on USB runs every neural
network. In today's prototype the same software runs on a PC GPU: ten hard-hat cameras go through
one batched detector, all 100 frames per second, which we estimate at about 40 % of one
UGen300. A site adds one box per area of up to twelve cameras, with no server room and no cloud.

### 14 · models: Every job runs on a Hailo-supported model

*On the slide:* A table of job, model, PC measurement and Hailo-10H reference: hard hats
(YOLOv8s; 31–40 ms per 10-frame batch; 252 FPS at batch 8), warehouse objects (6-class YOLOv8s at
960 px; 63–95 ms per 11-frame batch; HEF compiled, INT8 0.947 mAP50 = float in the emulator),
objects named in rules (YOLO-World v2-S; 46.6 FPS), speech (Whisper-Small; 10.6 tokens/s), rules
and report plan (Qwen2.5-1.5B; 0.35–0.40 s mean per rule, max 0.64 s; 7.35 tokens/s, 0.37 s first
token) and the Jev-style check (CLIP ViT-B/32 + 512→2 head; 15.2 ms per crop; 81.5 FPS). Below
it: Hailo's INT8 emulator, with one evaluator scoring float and INT8, gives 0.889 vs 0.893 mAP50
for the hard-hat detector, 0.947 vs 0.947 for the warehouse detector and 0.951 at the cameras'
16:9 shape (544×960, 43 % fewer pixels), and 95.3 % for the check with its head retrained on INT8
features. A second line estimates the LLM step of a spoken rule at about 11 s on the UGen300.

This table is the feasibility argument. Every job maps to a model Hailo already publishes for
Hailo-10H; the warehouse detector uses the same YOLOv8s architecture and is already compiled to a
HEF. The language model runs only on demand, when someone speaks a rule or
the daily report is planned, and our code calls it through the Ollama API, which hailo-ollama
serves on the UGen300. On the PC a rule takes well under a second; from Hailo's reference speeds
we estimate about eleven seconds on the UGen300, which is fine for a rule the officer confirms. Both
detectors are compiled to HEFs; they and the CLIP encoder were quantized to INT8 and checked in
Hailo's emulator, scored the same way as the float models, but not yet on hardware.

### 15 · business: One box per site area, via CCTV installers

*On the slide:* Who buys: construction contractors, factories and warehouses, through the site
safety officer or EHS manager. What we sell: the SayGuard box (a UGen300 and a small host next to
the recorder), one per area of up to 12 cameras, installed by CCTV installers alongside ASUS IoT
hardware. Why it pays: one NT$6,380 accelerator per area, about NT$640 per camera at 10 cameras; the
accelerator draws 2.5 W typical, plus a small host, with no server room and no cloud fees; and a new
rule is one spoken sentence, not a vendor project. A "Where it fits" strip below compares the
alternatives: a person at the screens (about NT$35,000 a month, one shift, attention fades), cloud
AI safety services (list prices such as US$45,000 per site plus US$2,500 per camera a year; video
clips leave the site) and AI cameras (new hardware, fixed rule list). The footer cites the UGen300
Taiwan retail price (Oct 2026), the MOL 2025 wage survey (guards and building managers) and the
Protex AI listing on AWS Marketplace (Oct 2026).

Sites already own cameras and answer for hard hats and forklift safety; what they rarely have is
someone watching every screen. We sell one SayGuard box per site area, a UGen300 with a small host
next to the recorder, and the CCTV installers who already serve these sites put it in alongside ASUS
IoT hardware; a bigger site simply adds boxes. The accelerator costs about NT$640 per camera at ten
cameras, draws very little power and sends nothing to the cloud. Compare that with the alternatives:
a person watching the screens costs about NT$35,000 a month for a single shift, by the Ministry of
Labor's wage survey for guards and building managers, and attention fades within minutes; cloud AI
safety services are typically sold per camera per year, for example US$45,000 per site plus US$2,500
per camera a year on one public listing, and their event clips leave the site; and AI cameras mean
new hardware with a fixed list of rules. When the site needs a new rule about objects, zones,
distances, counts or times the system already knows, the officer simply says it instead of opening a
project with a vendor.

### 16 · outcomes: What a site gets, and what Stage II will prove

*On the slide:* Three columns. For the site: every camera watched, every frame; one alert per
worker, double-checked; new rules by voice, confirmed by the officer; a daily report nobody
rewrites; no video leaving the site. Measured in Stage I: the alert check 95.3 % right at 15 ms
each; unseen rules 95 % / 84–88 % exactly right typed and 90 % / 74–86 % spoken; report actions on
the wrong problem 0 % (free text 20 %); head count within 1.7 people in six synthetic warehouses;
INT8 close to float in Hailo's emulator (0.889 vs 0.893 and 0.947 vs 0.947 mAP50). Stage II exit
criteria: 10 cameras × 10 fps on one UGen300, every frame; on-chip accuracy matching the emulator;
the rule tests re-run on the Hailo LLM build; warehouse cameras per box measured; alert latency
and power measured end to end. The footer notes that exit criteria are targets, not results, and
that the spoken tests use synthetic voices with site noise.

This is what SayGuard promises a site, and how we will know it holds. A site gets every camera
watched, one double-checked alert per worker, rules added by voice and confirmed by the officer, a
daily report nobody has to rewrite, and no video leaving the site. Stage I already measured the
middle column: the alert check is right 95 % of the time in 15 ms; unseen rules come out exactly
right 95 % and 84–88 % of the time when typed, and 90 % and 74–86 % when spoken by synthetic
voices over site noise; the report never aims an action at the wrong problem; the warehouse head
count stays within two people; and in Hailo's emulator the INT8 detectors nearly match their float
versions. Stage II succeeds when one UGen300 analyses every frame of ten cameras, the chip's
accuracy matches the emulator, the rule tests pass on the Hailo language-model build, we know how
many warehouse cameras one box takes, and alert latency and power are measured end to end.

### 17 · stage2: Four weeks on the UGen300

*On the slide:* Four week cards for Stage II (Nov 6 – Dec 4, 2026): week 1, detectors on device
(run both detector HEFs through HailoRT, measure FPS and latency over USB at batch 1 and 8, and
warehouse load at 960 and 544×960); week 2, GenAI on device (the Jev-style yes/no check compiled
for the chip, its head retrained on device features; Qwen2.5 on hailo-ollama with the rule tests
re-run; Whisper; the YOLO-World HEF); week 3, a real site test at the ASUS HQ test space
(calibrate a multi-camera floor plan, tune thresholds, measure alert latency, power and load);
week 4, the finals package (GitHub with README and benchmarks, demo video, final deck). A closing
line lists what Stage I already delivered: both detectors trained on the PC GPU and compiled to
Hailo-10H HEFs on a CPU and a cloud T4 (INT8 checked in Hailo's emulator, not yet run on
hardware), the full pipeline and dashboard, warehouse mode tested on six buildings the detector
never trained on, 353 unit tests and every evaluation script in the repo.

Stage II moves a system that works on a PC today onto the UGen300. Both detectors are already
compiled for Hailo-10H and their INT8 accuracy checked in Hailo's emulator, but nothing has run on
the device yet. Week one runs the detectors on the UGen300, measures batch-8 throughput over USB,
and measures the warehouse detector's load. Week two brings the yes/no check,
Qwen2.5 on hailo-ollama and Whisper onto the device, with the rule tests re-run on the device
build. Week three is a real test space with a calibrated floor plan, and week four delivers the
finals package.

### 18 · risks: What it cannot do yet, and the plan

*On the slide:* A table of seven limits, each with a plan: missed forklifts in two of the six test
warehouses (fine-tune on the customer's own vehicles, then test once on new footage); nothing run
on a UGen300 yet, so USB throughput and warehouse-mode load are unknown (measure both in week 1;
544×960 input, same accuracy, or 5 fps per camera if needed); two generative models cannot run at
once on one Hailo-10H (Whisper, then Qwen, behind a queue; the detectors keep running alongside);
Whisper mishearing some words, so spoken warehouse rules parse 74–86 % (fix on a new dev set, test
on fresh sentences; the officer confirms every rule); a misread or offline LLM (code checks every
rule; code alone still parses 68–90 % of rules); large halls with distant cameras (up to 12
cameras per box, every watched zone seen by two or more); and the double-check filtering a real
violation, 5 % in our test (filtered alerts stay in the feed, dimmed).

These are the limits we know of today, each with a plan. The biggest is forklifts in unfamiliar
buildings: in two of the six test warehouses the detector missed most of them, so we plan a short
fine-tune on each customer's own vehicles. Worker IDs change too often to show, so the dashboard
counts and alerts without them. Large halls need the coverage spec. Nothing has run on the
UGen300 yet, so throughput over USB and the load of warehouse mode are the first measurements of
Stage II. And because every AI answer is typed and checked, a misread rule or a filtered alert
stays visible to the officer instead of failing silently.

### 19 · close: SayGuard

*On the slide:* "Every camera watched. Rules about known objects and zones, one sentence away."
Links: code at github.com/Rober1208/sayguard, the demo video at youtu.be/SH1x9OGXXZQ and the
team, 逼寶隊. References: the Hailo Model Zoo and GenAI Model Zoo, the ASUS UGen300 specification, Roboflow
Hard Hats v2, NVIDIA PhysicalAI-SmartSpaces, Ultralytics YOLOv8 and YOLO-World, OpenAI Whisper
and CLIP, Qwen2.5, TypeSafe AI's Jev as the design idea, and Mixkit stock footage, with the full
links and the safety and price sources in appendix A2.

Thank you. To sum up: SayGuard is designed so that one UGen300 per site area watches every camera,
the safety officer adds rules by saying them, and every AI decision is a typed answer that code
can check. That is why small models are enough for a 2.5-watt chip. The code, the evaluation scripts and the results
behind every number we measured in this deck are in the repository at github.com/Rober1208/sayguard,
together with the Stage II plan for moving it all onto the UGen300.

## Appendix (not counted in the 20 pages)

### A1 · appendix_hailo: How the models share one Hailo-10H

*On the slide:* A table of when each model runs and its budget on the chip: the YOLOv8s detector
always, on every camera at batch 8 (about 40 % for 10 cameras × 10 fps at 640 px, estimated from
Hailo's 252 FPS); the CLIP encoder and 512→2 head per alert (81.5 FPS on the chip by Hailo's
figures, about 12 ms per crop, with the head on the host); YOLO-World v2-S per object rule, only on
the cameras that need it, every third frame (46.6 FPS); and Whisper-Small, then Qwen2.5-1.5B, on
demand, one after the other (3.4 s / 5.2 s to load when not resident). Below it: one shared
HailoRT device group, where vision and GenAI models can share the chip but two GenAI models cannot
run at once (Hailo's hailo-apps guide), with hailo-ollama joining through
`HAILO_OLLAMA_VDEVICE_GROUP_ID`; video over USB at about 1 Gb/s of the 10 Gb/s link (estimate);
the warehouse detector compiled at 960×960 (3 contexts) and at the cameras' 16:9 shape, 544×960
(4 contexts), where in Hailo's emulator the 16:9 build keeps its accuracy (INT8 0.951 mAP50) with
43 % fewer pixels, about 1.28× the work of a 640×640 frame; and what stays unknown until the device
arrives: warehouse fps at both shapes and with 12 cameras, the cost of switching contexts and
models, and detector speed while the LLM drafts a rule, all measured in weeks 1 and 2.

The detector runs on every frame of every camera; for ten hard-hat cameras that is about 40 % of
the chip by Hailo's reference speed. The CLIP encoder runs only when an alert needs checking,
YOLO-World only on cameras whose rules name an object it alone knows, and the two language models
only when someone speaks a rule or the daily report is due, one after the other, because Hailo's
guide says vision and generative models can share the device but two generative models cannot run
at once. Video over USB takes about a tenth of the link. The warehouse detector is compiled for the
chip at both shapes, in 3 and 4 contexts because the network is too large for one; fed frames in
the cameras' 16:9 shape, it keeps its accuracy in Hailo's emulator with 43 % fewer pixels. What we
cannot know before the device arrives is how fast the warehouse detector runs at either shape, the
cost of switching between contexts and between models, how much the detector slows while the
language model works, and how many warehouse cameras one box takes; those are the first
measurements of Stage II.

### A2 · appendix_refs: Sources behind the figures we did not measure

*On the slide:* Full links in four groups. Hardware and models: the ASUS UGen300 USB
specification, the Hailo Model Zoo (Hailo-10H tables), the Hailo GenAI Model Zoo (Whisper-Small,
Qwen2.5-1.5B), Hailo's hailo-apps guide to running vision and GenAI apps on one device, and
TypeSafe AI's Jev as the design idea. Data: Roboflow Hard Hats v2 and NVIDIA
PhysicalAI-SmartSpaces (both CC BY 4.0) and Mixkit stock video. Problem and market: the Taiwan
Ministry of Labor's 2025 occupational fatalities, the MOL Institute of Labor's review of forklift
accidents from 2011 to 2024, a Sandia / US National Institute of Justice guide to security
technologies, the MOL 2025 wage survey and the Protex AI list price on AWS Marketplace. Software:
Ultralytics YOLOv8 and YOLO-World, OpenAI Whisper and CLIP, Qwen2.5 and Ollama with their
licences, and the repository with every measured result.

These are the sources behind every figure in the deck that we did not measure ourselves: the
hardware and model references from ASUS and Hailo, the datasets, the safety statistics and the
price comparisons. Every number we measured is in the repository, next to the script that
produced it.
