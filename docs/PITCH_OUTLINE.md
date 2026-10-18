# Stage I proposal deck — outline (English, ≤ 20 pages of body)

Required sections (rules §6.1): context & problem, solution (tools/tech/method),
HW/SW architecture, expected benefits & outputs, references + GitHub link.
Scoring weights drive emphasis: business potential & feasibility 35 %, innovation 30 %,
practical value 25 %, storytelling 10 %.

1. **Title** — SayGuard: one UGen300, ten cameras, safety rules you can say. Team, track (Battlefield Lightning · Workplace AI · Onsite).
2. **The problem** — sites already have cameras, nobody watches them; PPE violations caught after the fact; existing "AI cameras" need a new model (data + retraining) for every new hazard; cloud video raises privacy and bandwidth issues.
3. **Who pays** — construction contractors, factories, warehouses; Taiwan Occupational Safety and Health Act duty of care; safety officers' pain: false alarms and configuration effort.
4. **The idea in one picture** — existing cameras → one UGen300 box → alerts + daily report; manager speaks a rule.
5. **Live demo stills** — dashboard, alert with VLM verdict, spoken rule card, report.
6. **Why one accelerator is enough** — batching all streams through one detector; YOLOv8s 252 FPS (batch 8) on Hailo-10H vs 100 FPS demand → ~40 % load; table for 5/10/15 fps per camera, max cameras.
7. **Rules you can say** — pipeline Whisper → Qwen2.5-1.5B (JSON schema) → deterministic checker → human confirm; examples in Mandarin/English; why the checker (safety-critical, small model).
8. **No retraining for new hazards** — open-vocabulary detection (YOLO-World) from the rule's noun; distance from head size (0.23 m prior).
9. **Fewer false alarms** — persistence + track voting + VLM second opinion only on alerts; measured filter rate [x/y].
10. **Privacy by design** — on-device, faces pixelated from the head boxes the detector already finds, only event snapshots stored.
11. **Architecture (HW)** — cameras/NVR → host (mini PC or Raspberry Pi 5 class) → UGen300 (USB-C 10 Gbps or M.2); power 2.5 W accelerator.
12. **Architecture (SW)** — modules diagram: sources → batched detector → ByteTrack → rule engine → alerts store → VLM verifier → web UI/API; LLM services via Ollama-compatible API = hailo-ollama on device.
13. **Model map PC → UGen300** — table from `docs/HAILO_PORT.md` with Hailo zoo numbers.
14. **Detector results** — dataset (19,745 images, CC BY 4.0), mAP50 / mAP50-95 / per-class P/R on the held-out test split, speed.
15. **Prototype results** — measured throughput on PC, end-to-end alert latency, spoken-rule parse accuracy on the test sentences, report generation time.
16. **Business model** — hardware kit (UGen300 + host) + per-site software subscription; cost per camera channel (accelerator ≈ NT$6,380 / 10 ch ≈ NT$640 per channel) vs per-camera cloud AI fees or a GPU server; energy per channel.
17. **Go-to-market & roadmap** — pilot on a site (Stage II at ASUS HQ test space), integrations (NVR/VMS, LINE/Teams alerts), more PPE classes (vest, harness), multi-site dashboard.
18. **Stage II plan (4 weeks)** — week 1 HEF compile + HailoRT backend; week 2 GenAI on hailo-ollama; week 3 on-site tuning, latency/power measurement; week 4 polish + finals demo.
19. **Risks & mitigations** — USB vs PCIe throughput (measure, reduce fps per camera), LLM parse errors (checker + confirm), VLM latency (only on alerts, queue), lighting/occlusion (persistence, VLM).
20. **References & links** — GitHub repo, Hailo Model Zoo pages, dataset, models, licences.
