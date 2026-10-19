"""Train the hard-hat yes/no head the alert verifier uses (verifier.backend: clip_probe).

Frozen CLIP ViT-B/32 image encoder -> unit-norm 512-d feature -> 2x512 linear head.  The
crops are the head-and-shoulders crops an alert produces (eval_vlm.alert_crop) from the
train split; weight decay is picked on val; the score is reported on the same held-out
test crops as scripts/eval_yesno.py and scripts/eval_vlm.py.  Features come from the
serving code (sayguard.verify.ClipProbeBackend), so training and the live check match.

    python scripts/train_probe.py            # -> @work/models/helmet_probe.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_yesno import crops  # noqa: E402
from sayguard.config import load_config, resolve, work_dir  # noqa: E402

SCALE = 10.0


def fit(tr: np.ndarray, ytr: np.ndarray, va: np.ndarray, yva: np.ndarray, device: str):
    """The eval_yesno.py recipe: AdamW, 300 full-batch steps, weight decay chosen on val."""
    import torch
    torch.manual_seed(0)
    X, Y, V = (torch.tensor(a, device=device) for a in (tr * SCALE, ytr, va * SCALE))
    best = None
    for wd in (0.0, 1e-4, 1e-3, 1e-2):
        head = torch.nn.Linear(X.shape[1], 2).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=wd)
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(head(X), Y).backward()
            opt.step()
        with torch.inference_mode():
            acc = float((head(V).argmax(1).cpu().numpy() == yva).mean())
        if best is None or acc > best[0]:
            best = (acc, wd, head.weight.detach().cpu().numpy(), head.bias.detach().cpu().numpy())
    return best


def main() -> None:
    from PIL import Image

    from sayguard.resources import require
    from sayguard.verify import ClipProbeBackend, LinearProbe

    require("train_probe", gpu_gb=1.0, ram_gb=2.0)
    vcfg = load_config().get("verifier", {})
    clip_path = str(resolve(vcfg.get("clip", "@work/models/clip/ViT-B-32.pt")))
    out_path = Path(resolve(vcfg.get("probe", "@work/models/helmet_probe.npz")))
    enc = ClipProbeBackend(clip_path, str(out_path))
    enc.load_encoder()

    def unit(f):
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    data = {}
    for split, per, seed in (("train", 1500, 1), ("val", 300, 2), ("test", 100, 0)):   # as eval_yesno
        x, y = crops(split, per, seed=seed)
        data[split] = (unit(enc.embed([Image.fromarray(im) for im in x])), y)
        print(split, len(y), "crops")
    acc, wd, W, b = fit(*data["train"], *data["val"], enc.device)
    probe = LinearProbe(W, b, SCALE)
    te, yte = data["test"]
    pred = (probe.prob_wearing(te) >= 0.5).astype(int)
    res = {"encoder": "CLIP ViT-B/32 (OpenAI), frozen", "head": "linear 512->2 on 10x unit features",
           "train_crops": int(len(data["train"][1])), "val_accuracy": round(acc, 3), "weight_decay": wd,
           "test_crops": {"helmet": int(yte.sum()), "head": int((1 - yte).sum())},
           "test_accuracy": round(float((pred == yte).mean()), 3),
           "false_alarm_filter_rate": round(float((pred[yte == 1] == 1).mean()), 3),
           "violation_confirm_rate": round(float((pred[yte == 0] == 0).mean()), 3)}
    probe.save(out_path, val_accuracy=np.float32(acc), weight_decay=np.float32(wd))

    # Live-path latency: one crop per call through judge(), exactly as an alert is checked
    # (eval_vlm.py times the VLM the same way).  The 3.5 ms/crop in yesno_check.json is
    # batched throughput (all test crops in batches of 128, total time / crops) and is not comparable.
    import time

    import torch
    enc._probe = probe
    test_x, _ = crops("test", 100, seed=0)
    bgr = [np.ascontiguousarray(im[:, :, ::-1]) for im in test_x]
    for im in bgr[:5]:
        enc.judge(im)                                  # warm-up (CUDA kernels, allocator)
    ms = []
    for im in bgr:
        t0 = time.perf_counter()
        enc.judge(im)                                  # returns a Python float: the GPU work is done
        ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    res["latency_ms_one_crop"] = {"mean": round(sum(ms) / len(ms), 1), "p95": round(ms[int(0.95 * (len(ms) - 1))], 1),
                                  "device": enc.device, "gpu": torch.cuda.get_device_name(0) if enc.device == "cuda" else None}
    for d in (work_dir() / "results", ROOT / "docs" / "results"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "probe_train.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    print("saved", out_path)


if __name__ == "__main__":
    main()
