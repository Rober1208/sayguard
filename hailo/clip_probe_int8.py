"""Does the CLIP "wearing a hard hat?" probe survive INT8 on the Hailo-10H?

Two halves, two environments:

  dump      (SayGuard venv, Windows/Linux)  the exact crops of scripts/eval_yesno.py, resized and
            center-cropped by CLIP's own transforms, plus PyTorch CLIP ViT-B/32 features of those
            same pixels -> yesno_crops.npz
  emulate   (Hailo DFC venv, WSL)  Model Zoo clip_vit_b_32_image_encoder HAR -> optimize with
            crops as calibration -> features from the SDK_FP_OPTIMIZED and SDK_QUANTIZED
            emulators -> retrain the linear probe and score it on the same 193 test crops

    python hailo/clip_probe_int8.py dump
    python hailo/clip_probe_int8.py emulate --har <clip_vit_b_32_image_encoder.har>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("SAYGUARD_WORK", "D:/ugen-work" if os.name == "nt" else "/mnt/d/ugen-work"))
NPZ = WORK / "results" / "yesno_crops.npz"
ZOO = Path(os.environ.get("HAILO_MODEL_ZOO", Path.home() / "hailo" / "hailo_model_zoo"))
ZOO_ALLS = ZOO / "hailo_model_zoo/cfg/alls/generic/clip_vit_b_32_image_encoder.alls"


def dump() -> None:
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    import clip
    from eval_yesno import crops
    from PIL import Image
    from torchvision import transforms as T

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    # the weights file the app uses (models/clip, shared with YOLO-World); absolute path = no download
    model, preprocess = clip.load(str(WORK / "models" / "clip" / "ViT-B-32.pt"), device=dev)
    model.eval()
    # CLIP's own geometry (Resize(224, bicubic) + CenterCrop(224)) without normalization: the chip
    # gets raw 0-255 RGB and normalizes on device (norm1 in the model script).  The PyTorch
    # features are computed from these same 224x224 pixels, so any difference is the compiler's.
    geom = T.Compose(preprocess.transforms[:2])
    for split, per, seed in (("test", 100, 0), ("train", 1500, 1), ("val", 300, 2)):   # as eval_yesno
        x, y = crops(split, per, seed=seed)
        raw = np.stack([np.asarray(geom(Image.fromarray(im)).convert("RGB")) for im in x])
        with torch.inference_mode():
            f = torch.cat([model.encode_image(torch.stack([preprocess(Image.fromarray(r)) for r in raw[i:i + 128]]).to(dev)).float()
                           for i in range(0, len(raw), 128)])
        out[f"{split}_x"], out[f"{split}_y"], out[f"{split}_torch"] = raw, y, f.cpu().numpy()
        print(split, raw.shape, f.shape)
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ, **out)
    print("saved", NPZ)


def unit(f: np.ndarray) -> np.ndarray:
    f = f.reshape(len(f), -1).astype(np.float32)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def probe(tr, ytr, va, yva):
    """The eval_yesno.py recipe: linear head on 10x unit features, AdamW, pick weight decay on val."""
    import torch
    torch.manual_seed(0)
    best = None
    for wd in (0.0, 1e-4, 1e-3, 1e-2):
        head = torch.nn.Linear(tr.shape[1], 2)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=wd)
        X, Y = torch.tensor(tr * 10), torch.tensor(ytr)
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(head(X), Y).backward()
            opt.step()
        with torch.no_grad():
            acc = float((head(torch.tensor(va * 10)).argmax(1).numpy() == yva).mean())
        if best is None or acc > best[0]:
            best = (acc, wd, head)
    return best


def score(head, te, yte) -> dict:
    import torch
    with torch.no_grad():
        p = head(torch.tensor(te * 10)).argmax(1).numpy()
    return {"accuracy": round(float((p == yte).mean()), 3),
            "false_alarm_filter_rate": round(float((p[yte == 1] == 1).mean()), 3),
            "violation_confirm_rate": round(float((p[yte == 0] == 0).mean()), 3)}


def emulate(har: str, batch: int) -> None:
    from hailo_sdk_client import ClientRunner, InferenceContext

    d = np.load(NPZ)
    runner = ClientRunner(har=har)
    if "optimized" not in Path(har).name:
        runner.load_model_script(ZOO_ALLS.read_text())
        # 64 train crops (the model script's calibset_size), 32 per class, evenly spread
        y = d["train_y"]
        pick = np.concatenate([idx[np.linspace(0, len(idx) - 1, 32).astype(int)]
                               for idx in (np.flatnonzero(y == 0), np.flatnonzero(y == 1))])
        calib = d["train_x"][pick].astype(np.float32)
        print("calibration", calib.shape, "per class", np.bincount(y[pick]), flush=True)
        runner.optimize(calib)
        har = str(Path(har).with_name(Path(har).stem + "_optimized.har"))
        runner.save_har(har)

    import tensorflow as tf
    splits = ("train", "val", "test")
    allx = np.concatenate([d[f"{s}_x"] for s in splits])          # uint8, 3.7k x 224 x 224 x 3 = 0.56 GB
    bounds = np.cumsum([0] + [len(d[f"{s}_x"]) for s in splits])

    def dataset():
        # one infer call per context: calling infer per batch leaks memory in the emulator
        def gen():
            for i in range(len(allx)):
                yield allx[i].astype(np.float32), {"id": np.int32(i)}
        return tf.data.Dataset.from_generator(gen, output_signature=(
            tf.TensorSpec((224, 224, 3), tf.float32), {"id": tf.TensorSpec((), tf.int32)}))

    feats = {}
    for name, ctx_type in (("fp_optimized", InferenceContext.SDK_FP_OPTIMIZED),
                           ("quantized", InferenceContext.SDK_QUANTIZED)):
        t0 = time.time()
        with runner.infer_context(ctx_type) as ctx:
            f = unit(np.asarray(runner.infer(ctx, dataset(), batch_size=batch)))
        for k, split in enumerate(splits):
            feats[(name, split)] = f[bounds[k]:bounds[k + 1]]
        print(name, f"{time.time() - t0:.0f}s", flush=True)

    torch_f = {s: unit(d[f"{s}_torch"]) for s in ("train", "val", "test")}
    y = {s: d[f"{s}_y"] for s in ("train", "val", "test")}
    res = {"crops": {s: int(len(y[s])) for s in y},
           "cosine_to_pytorch": {n: round(float((feats[(n, "test")] * torch_f["test"]).sum(1).mean()), 4)
                                 for n in ("fp_optimized", "quantized")}}
    pc = probe(torch_f["train"], y["train"], torch_f["val"], y["val"])
    res["pc_probe_on_pc_features"] = score(pc[2], torch_f["test"], y["test"])
    res["pc_probe_on_int8_features"] = score(pc[2], feats[("quantized", "test")], y["test"])
    q = probe(feats[("quantized", "train")], y["train"], feats[("quantized", "val")], y["val"])
    res["int8_probe_on_int8_features"] = score(q[2], feats[("quantized", "test")], y["test"])
    res["note"] = ("Model Zoo clip_vit_b_32_image_encoder + its model script; 64 site crops as calibration; "
                   "bit-exact INT8 emulation (SDK_QUANTIZED). Same crops and probe recipe as scripts/eval_yesno.py.")
    for out in (WORK / "results", ROOT / "docs" / "results"):
        out.mkdir(parents=True, exist_ok=True)
        (out / "clip_probe_int8.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dump", "emulate"])
    ap.add_argument("--har")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    dump() if a.stage == "dump" else emulate(a.har, a.batch)


if __name__ == "__main__":
    main()
