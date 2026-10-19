English | [繁體中文](PITCH_OUTLINE.zh-TW.md)

# Stage I proposal deck: outline and talking points

The Stage I deck has 18 slides in English (the rules allow up to 20 pages of body). The slide
sources are in `docs/deck/project/slides/<id>.html`, and their order is in
`docs/deck/project/deck.json`. This outline follows the same order: for each slide, what is on it
and one paragraph of talking points. Every measured number comes from the result files in
[docs/results/](results/), summarised in [FACTS.md](FACTS.md). Stage I runs on a PC GPU with the
same models planned for the UGen300, so UGen300 figures in the deck are labelled as Hailo
references or our estimates; the limits are on slide 17.

## Required sections and scoring

The rules (§6.1) require context and problem, solution (tools, technology, method),
hardware/software architecture, expected benefits and outputs, and references with the GitHub
link. The scoring weights set the emphasis: business potential and feasibility 35 %, innovation
30 %, practical value 25 %, storytelling 10 %.

| Required section | Slides |
|---|---|
| Context and problem | 1 cover, 2 problem |
| Solution: tools, technology, method | 3 idea, 4 demo, 5 llm, 6 falsealarm, 7 voice, 8 report, 9 site, 11 detector, 12 privacy |
| Hardware/software architecture | 13 hardware, 14 models |
| Expected benefits and outputs (with evidence) | 6 falsealarm, 7 voice, 8 report, 10 siteevidence, 15 business, 16 stage2, 17 risks |
| References and GitHub link | 18 close |

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
report is planned. Slide 5 states the idea, slides 6–8 show it at work, and slide 17 comes
back to it: a misread rule or a filtered alert stays visible instead of failing silently.

## Slide by slide

### 1 · cover: SayGuard

*On the slide:* "ASUS UGen AI League · Stage I proposal", the name, and the line "One UGen300 per
site area. Every camera watched. Rules you can say." Chips for Battlefield Lightning · UGen300,
Workplace AI, and the team and track placeholders.

SayGuard is designed to put one ASUS UGen300 next to the video recorder of each site area and
watch every camera in it. Safety officers set the rules by saying them, in Mandarin or English.
Like TypeSafe AI's Jev, every decision an AI model makes in SayGuard is a typed answer instead of
text, which keeps it fast enough for a 2.5-watt chip and easy for code to check. Today the Stage I prototype
runs the same software on a local PC GPU. Every figure in this deck comes from that working
prototype unless it is labelled as a Hailo or ASUS reference or as an estimate.

### 2 · problem: Sites have cameras. Few have anyone watching them live.

*On the slide:* Three cards. "Found too late": a missing hard hat may only be noticed in recorded
footage, after an incident. "Rules are fixed": a new zone, distance, head-count or time rule means
reconfiguring by hand. "Alarms get ignored": frequent false alerts wear down trust, and streaming
video to the cloud raises privacy and bandwidth concerns.

Construction sites and warehouses already have cameras, but few have anyone watching them live, so
a missing hard hat may only be noticed in recorded footage after an incident. Many AI cameras ship
with a set list of rules, so adding a zone, a distance, a head count or a time window means
reconfiguring by hand. And alarms that are often wrong get ignored, while streaming site video to
the cloud raises privacy and bandwidth concerns. SayGuard answers all three: live coverage of
every camera, rules anyone can add by speaking, and alerts worth reading.

### 3 · idea: One box watches every camera on site

*On the slide:* Five feature cards: every camera live (one detector for all streams; in a
warehouse, one live floor plan), rules you can say (live in seconds, no data, no retraining),
alerts people trust (a fast yes/no double-check; one worker, one alert), a daily safety report
(counts from the alert log, the LLM picks the top problems and vetted fixes), and video that stays
on site. A sixth card names the two modes today: hard-hat mode and warehouse mode.

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
the Hailo Model Zoo, and its INT8 build keeps the same accuracy in Hailo's emulator once the head
is retrained on INT8 features. Filtered alerts are never deleted; they stay in the feed, dimmed.

### 7 · voice: Say a rule, confirm it, it is live

*On the slide:* Four steps: speak (Whisper, Mandarin or English), fill the form (Qwen2.5-1.5B
fills a fixed rule form, not free text), check (code corrects every number, time and object it
finds), and confirm (the officer reads the rule in plain words and switches it on). Rules exactly
right on typed sentences never seen in development: 95 % in hard-hat mode (LLM alone 40 %) and
84–88 % in warehouse mode (LLM alone 16–24 %), next to a screenshot of the Say a rule panel.

The officer holds the button and says the rule. Whisper turns it into text, primed with the site's
own zone and object names. The small language model does not write an answer; it fills a fixed
rule form, and because the form is typed, code can check it: every number, time and object stated
in the sentence is corrected if the model got it wrong. The officer then reads the rule in plain
words and switches it on. On typed sentences never used in development, this gets 95 % of
construction-site rules and 84–88 % of warehouse rules exactly right, against 40 % and 16–24 % for
the LLM alone. A new rule about known objects, zones, distances, counts or times needs no data and
no retraining.

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

### 10 · siteevidence: Tested on six warehouses it has never seen

*On the slide:* Three headline figures: head count 0.05–1.7 people off on average (summing
cameras: up to 60 off), 100 % of door and dock entries alerted with no false alerts, and 92 % of
forklift close calls alerted with 1 false alert in 26. A table by zone kind adds work areas (323
of 338 alerted, 7 % false) and narrow aisles (98 of 118 alerted, 4 % false).

We tested warehouse mode on six synthetic warehouses the detector never trained on, with every
setting chosen on a separate validation scene. Fusing all cameras on one floor plan keeps the head
count within about two people of the truth, where summing the cameras can be off by sixty. Every
entry through a door or dock zone was alerted with no false alerts, and the forklift rule caught
92 % of close calls. An event counts as alerted when the worker got an alert or was still inside
the cooldown of one; the strict numbers, and a run with the settings frozen before any fix, are in
the repository's results folder.

### 11 · detector: Trained detectors, plus words for the rest

*On the slide:* Hard hats: 0.939 mAP50 on 2,001 test images it never saw, from a YOLOv8s already
compiled for Hailo-10H (INT8 0.889 vs float 0.893 in Hailo's emulator). Warehouse objects: a
6-class YOLOv8s for people, vehicles and robots, trained on 8,000 frames from five synthetic
buildings. For anything else a rule names, such as an excavator, YOLO-World finds objects from the
word itself.

Each class a site depends on every day gets a trained detector. The hard-hat model scores 0.939
mAP50 on test images it never saw, and it is already compiled for Hailo-10H, where INT8 barely
changes its accuracy in Hailo's emulator. The warehouse model knows six classes of people,
vehicles and robots. For anything else a spoken rule names, like the excavator in this snapshot,
YOLO-World works from the word alone. That part is a demo and its accuracy is not measured yet; in
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
(YOLOv8s; 31–40 ms per 10-frame batch; 252 FPS at batch 8), warehouse objects (6-class YOLOv8s;
63–95 ms per 11-frame batch), objects named in rules (YOLO-World v2-S; 46.6 FPS), speech
(Whisper-Small; 10.6 tokens/s), rules and report plan (Qwen2.5-1.5B; 0.35–0.40 s mean per rule,
max 0.64 s; 7.35 tokens/s, 0.37 s first token) and the Jev-style check (CLIP ViT-B/32 + 512→2
head; 15.2 ms per crop; 81.5 FPS). Below it: INT8 in Hailo's emulator keeps 0.889 vs 0.893 mAP50
for the hard-hat detector and 95.3 % for the check with its head retrained on INT8 features, and
the LLM step of a spoken rule takes an estimated 11 s on the UGen300 (Whisper adds about 2.4 s).

This table is the feasibility argument. Every job maps to a model Hailo already publishes for
Hailo-10H; the warehouse detector uses the same YOLOv8s architecture, and we compile its HEF in
the first week of Stage II. The language model runs only on demand, when someone speaks a rule or
the daily report is planned, and our code calls it through the Ollama API, which hailo-ollama
serves on the UGen300. On the PC a rule takes well under a second; from Hailo's reference speeds
we estimate about eleven seconds on the UGen300, which is fine for a rule the officer confirms. So
far only the hard-hat detector is compiled to a HEF, and it and the CLIP encoder have been checked
in Hailo's emulator, not yet on hardware.

### 15 · business: One box per site area, via CCTV installers

*On the slide:* Who buys: construction contractors, factories and warehouses, through the site
safety officer or EHS manager. What we sell: the SayGuard box (a UGen300 and a small host next to
the recorder), one per area of up to 12 cameras, installed by CCTV installers alongside ASUS IoT
hardware. Why it pays: one NT$6,380 accelerator per area with ten hard-hat cameras using about
40 % of it (estimate), 2.5 W typical with no server room and no cloud fees, and a new rule is one
spoken sentence, not a vendor project.

Sites already own cameras and answer for hard hats and forklift safety; what they rarely have is
someone watching every screen. We sell one SayGuard box per site area, and the CCTV installers who
already serve these sites put it in; a bigger site simply adds boxes. The accelerator is cheap per
camera, draws very little power and sends nothing to the cloud. When the site needs a new rule
about objects, zones, distances, counts or times the system already knows, the officer simply
says it instead of opening a project with a vendor.

### 16 · stage2: Four weeks on the UGen300

*On the slide:* Four week cards for Stage II (Nov 6 – Dec 4, 2026): detectors on device, GenAI on
device, a real site test at the ASUS HQ test space, and the finals package. A closing line lists
what Stage I already delivered on the PC GPU: both detectors trained, the hard-hat HEF compiled,
the full pipeline and dashboard, warehouse mode tested on six buildings the detector never trained
on, 298 unit tests and every evaluation script in the repo.

Stage II moves a system that works on a PC today onto the UGen300. The hard-hat detector is
already compiled for Hailo-10H and its INT8 accuracy checked in Hailo's emulator, but nothing has
run on the device yet. Week one runs the detectors on the UGen300, measures batch-8 throughput over
USB, and compiles the warehouse detector and measures its load. Week two brings the yes/no check,
Qwen2.5 on hailo-ollama and Whisper onto the device, with the rule tests re-run on the device
build. Week three is a real test space with a calibrated floor plan, and week four delivers the
finals package.

### 17 · risks: What it cannot do yet, and the plan

*On the slide:* A table of six limits, each with a plan: missed forklifts in two of the six test
warehouses (fine-tune on the customer's own vehicles); worker IDs that change within tens of
seconds (the dashboard never shows IDs); large halls with distant cameras (up to 12 cameras per
box, every watched zone seen by two or more); nothing run on a UGen300 yet (measure USB throughput
and warehouse-mode load in week 1); a misread or offline LLM (code checks every rule and the
officer confirms; code alone still parses 68–90 % of rules); and the double-check filtering a real
violation, 5 % in our test (filtered alerts stay in the feed, dimmed).

These are the limits we know of today, each with a plan. The biggest is forklifts in unfamiliar
buildings: in two of the six test warehouses the detector missed most of them, so we plan a short
fine-tune on each customer's own vehicles. Worker IDs change too often to show, so the dashboard
counts and alerts without them. Large halls need the coverage spec. Nothing has run on the
UGen300 yet, so throughput over USB and the load of warehouse mode are the first measurements of
Stage II. And because every AI answer is typed and checked, a misread rule or a filtered alert
stays visible to the officer instead of failing silently.

### 18 · close: SayGuard

*On the slide:* "Every camera watched. Rules about known objects and zones, one sentence away."
Links: code at github.com/Rober1208/sayguard, the demo video (link still to fill in) and the
team, 逼寶隊. References: the Hailo Model Zoo and GenAI Model Zoo, the ASUS UGen300 specification, Roboflow
Hard Hats v2, NVIDIA PhysicalAI-SmartSpaces, Ultralytics YOLOv8 and YOLO-World, OpenAI Whisper
and CLIP, Qwen2.5, TypeSafe AI's Jev as the design idea, and Mixkit stock footage.

Thank you. To sum up: SayGuard is designed so that one UGen300 per site area watches every camera,
the safety officer adds rules by saying them, and every AI decision is a typed answer that code
can check. That is why small models are enough for a 2.5-watt chip. The code, the evaluation scripts and the results
behind every number we measured in this deck are in the repository at github.com/Rober1208/sayguard,
together with the Stage II plan for moving it all onto the UGen300.
