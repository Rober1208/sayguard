"""Does the second opinion need a generative VLM?  "Is this person wearing a hard hat?" is
a binary question, so compare three non-generative options with the Qwen2-VL-2B check on
exactly the same held-out crops as scripts/eval_vlm.py:

  clip_zeroshot - CLIP ViT-B/32, image vs. two text prompts, no training
  clip_probe    - frozen CLIP ViT-B/32 image encoder + a logistic-regression head trained on
                  crops from the train split (the "frozen encoder + probe" recipe)
  resnet18      - ImageNet ResNet18 fine-tuned as a 2-class crop classifier

CLIP ViT-B/32 and ResNet18 both have Hailo-10H builds in the Hailo Model Zoo.

    python scripts/eval_yesno.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import eval_vlm as V  # noqa: E402  (same sampling and crop as the VLM evaluation)
from sayguard.config import work_dir  # noqa: E402

YES_PROMPTS = ["a photo of a person wearing a hard hat", "a photo of a construction worker wearing a safety helmet",
               "a person with a helmet on their head"]
NO_PROMPTS = ["a photo of a person not wearing a hard hat", "a photo of a person with a bare head",
              "a construction worker without a helmet"]


def crops(split: str, per_class: int, seed: int, min_px: int = 24) -> tuple[list, np.ndarray]:
    """[(RGB crop)], labels (1 = wearing a hard hat, 0 = bare head), as in eval_vlm."""
    items = V.sample(work_dir() / "datasets" / "helmet" / "images" / split, per_class, min_px, seed)
    out, y = [], []
    for path, c, (cx, cy, w, h) in items:
        img = cv2.imread(str(path))
        H, W = img.shape[:2]
        box = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
        if box[3] - box[1] < min_px:
            continue
        out.append(cv2.cvtColor(V.alert_crop(img, box), cv2.COLOR_BGR2RGB))
        y.append(1 if c == 0 else 0)
    return out, np.array(y)


def report(name: str, pred: np.ndarray, y: np.ndarray, ms: float) -> dict:
    helmet, head = y == 1, y == 0
    return {"method": name, "accuracy": round(float((pred == y).mean()), 3),
            "false_alarm_filter_rate": round(float((pred[helmet] == 1).mean()), 3),   # wearing -> filtered
            "violation_confirm_rate": round(float((pred[head] == 0).mean()), 3),     # bare head -> confirmed
            "ms_per_crop_pc": round(ms, 2)}


def main() -> None:
    import torch
    from PIL import Image

    from sayguard.resources import require
    require("eval_yesno", gpu_gb=2.0, ram_gb=2.0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)

    test_x, test_y = crops("test", 100, seed=0)          # identical to eval_vlm.py (94 helmet, 99 head)
    train_x, train_y = crops("train", 1500, seed=1)
    val_x, val_y = crops("val", 300, seed=2)
    print(f"crops: train {len(train_x)}, val {len(val_x)}, test {len(test_x)} "
          f"(test helmet {int(test_y.sum())}, head {int((1 - test_y).sum())})")
    results = []

    # ---- CLIP (zero-shot and frozen encoder + probe) --------------------------------
    import clip
    model, preprocess = clip.load("ViT-B/32", device=dev)
    model.eval()

    @torch.inference_mode()
    def embed(images: list) -> torch.Tensor:
        feats = []
        for i in range(0, len(images), 128):
            batch = torch.stack([preprocess(Image.fromarray(im)) for im in images[i:i + 128]]).to(dev)
            f = model.encode_image(batch).float()
            feats.append(f / f.norm(dim=-1, keepdim=True))
        return torch.cat(feats)

    with torch.inference_mode():
        t = clip.tokenize(YES_PROMPTS + NO_PROMPTS).to(dev)
        tf = model.encode_text(t).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
        yes_t = tf[:len(YES_PROMPTS)].mean(0)
        no_t = tf[len(YES_PROMPTS):].mean(0)
        text = torch.stack([no_t / no_t.norm(), yes_t / yes_t.norm()])       # index 0 = bare, 1 = wearing
    embed(test_x[:16])                                    # warm-up; timing = all test crops, batches of 128
    t0 = time.perf_counter()
    te = embed(test_x)
    if dev == "cuda":
        torch.cuda.synchronize()
    clip_ms = (time.perf_counter() - t0) * 1000 / len(test_x)
    zs = (te @ text.T).argmax(1).cpu().numpy()
    results.append(report("clip_zeroshot", zs, test_y, clip_ms))

    tr, va = embed(train_x), embed(val_x)
    # the shipped head's exact recipe and seeding, so this row equals docs/results/probe_train.json
    # (imported here, not at the top: train_probe imports crops() from this file)
    from train_probe import SCALE, fit
    acc, wd, W, b = fit(tr.cpu().numpy(), train_y, va.cpu().numpy(), val_y, dev)
    probe = (te.cpu().numpy() * SCALE @ W.T + b).argmax(1)
    r = report("clip_probe", probe, test_y, clip_ms)
    r["val_accuracy"], r["weight_decay"] = round(acc, 3), wd
    results.append(r)
    del model
    torch.cuda.empty_cache()

    # ---- ResNet18 fine-tuned crop classifier ------------------------------------------
    import torchvision
    from torchvision import transforms as T
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Linear(net.fc.in_features, 2)
    net = net.to(dev)
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = T.Compose([T.ToPILImage(), T.Resize((160, 160)), T.RandomHorizontalFlip(),
                          T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
    eval_tf = T.Compose([T.ToPILImage(), T.Resize((160, 160)), T.ToTensor(), norm])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    idx = list(range(len(train_x)))
    for epoch in range(5):
        net.train()
        random.shuffle(idx)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            xb = torch.stack([train_tf(train_x[j]) for j in b]).to(dev)
            yb = torch.tensor(train_y[b], device=dev)
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(xb), yb).backward()
            opt.step()
    net.eval()

    @torch.inference_mode()
    def classify(images: list) -> np.ndarray:
        return np.concatenate([net(torch.stack([eval_tf(im) for im in images[i:i + 128]]).to(dev)).argmax(1).cpu().numpy()
                               for i in range(0, len(images), 128)])
    classify(test_x[:16])                                 # warm-up
    t0 = time.perf_counter()
    rn = classify(test_x)
    if dev == "cuda":
        torch.cuda.synchronize()
    rn_ms = (time.perf_counter() - t0) * 1000 / len(test_x)
    r = report("resnet18", rn, test_y, rn_ms)
    r["val_accuracy"] = round(float((classify(val_x) == val_y).mean()), 3)
    results.append(r)

    vlm_file = work_dir() / "results" / "vlm_check.json"
    if vlm_file.exists():
        v = json.loads(vlm_file.read_text(encoding="utf-8"))
        results.append({"method": "qwen2_vl_2b (generative)", "accuracy": v["accuracy"],
                        "false_alarm_filter_rate": v["false_alarm_filter_rate"],
                        "violation_confirm_rate": v["violation_confirm_rate"],
                        "ms_per_crop_pc": v["latency_ms"]["mean"]})
    out = {"test_crops": {"helmet": int(test_y.sum()), "head": int((1 - test_y).sum())},
           "train_crops": len(train_x), "results": results,
           "note": ("PC timings after warm-up: CLIP/ResNet = all test crops in batches of 128, total time / crops "
                    "(throughput); VLM = one crop per call (vlm_check.json). One-crop CLIP probe latency: "
                    "probe_train.json. Same held-out crops as eval_vlm.py.")}
    for d in (work_dir() / "results", ROOT / "docs" / "results"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "yesno_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{'method':28s} {'acc':>6s} {'filter FA':>10s} {'confirm':>8s} {'ms/crop':>8s}")
    for r in results:
        print(f"{r['method']:28s} {r['accuracy']:6.3f} {r['false_alarm_filter_rate']:10.3f} "
              f"{r['violation_confirm_rate']:8.3f} {r['ms_per_crop_pc']:8.2f}")


if __name__ == "__main__":
    main()
