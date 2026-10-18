# Third-party data, models and code

The competition rules require entrants to hold the rights to every image, video, model
and dataset they use. Everything below is used under its published license; nothing
restricted is redistributed in this repository.

## Data

| Item | Source | License | Use |
|---|---|---|---|
| Hard Hats v2 (19,745 images, classes `hardhat` / `no-hardhat`) | Roboflow Universe "hard-hats-fhbh5", mirrored as `keremberke/hard-hat-detection` on Hugging Face | CC BY 4.0 (attribution) | Training/validation/test of the hard-hat detector |
| Demo camera footage | Mixkit (see `docs/VIDEO_SOURCES.md`) | Mixkit Stock Video Free License | Simulated CCTV streams; clips are **not** committed, `scripts/fetch_videos.py` re-downloads them |

Faces in every dashboard tile and saved snapshot are pixelated on-device (`sayguard/privacy.py`).

## Models

| Model | License |
|---|---|
| YOLOv8s (COCO pre-trained) and our fine-tuned weights, YOLO-World v2-S weights | AGPL-3.0 (Ultralytics) |
| CLIP ViT-B/32 text encoder (for YOLO-World class embeddings) | MIT (OpenAI) |
| Whisper-Small | MIT (OpenAI) |
| Qwen2.5-1.5B-Instruct | Apache-2.0 (Alibaba Qwen) |
| Qwen2-VL-2B-Instruct | Apache-2.0 (Alibaba Qwen) |

## Code dependencies

Ultralytics (AGPL-3.0), FastAPI/Starlette (MIT), Uvicorn (BSD-3),
OpenCV (Apache-2.0), PyTorch (BSD-3), Transformers (Apache-2.0), Ollama (MIT),
opencc-python-reimplemented (Apache-2.0), HailoRT (Hailo license, Stage II).

Because the detector code path uses Ultralytics (AGPL-3.0), this repository is released
under the AGPL-3.0 as well (see `LICENSE`), which keeps the combination compliant. Note that the
competition's IP clause (winners grant ASUS/Intel/Hailo a licence to the entry) does not
change third-party licences.
