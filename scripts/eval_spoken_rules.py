"""Spoken rules end to end: the held-out rule sentences are spoken, transcribed by the shipped speech
step (sayguard.asr.SpeechToText: Whisper-Small with the site's words as decoder prompt), parsed by the
shipped parser (Qwen2.5-1.5B + checker) and scored with the same field checks as scripts/eval_rules.py.

The speech is SYNTHETIC: Microsoft neural voices through edge-tts (two Taiwan Mandarin voices, two US
English voices), each sentence in both voices of its language, once clean and once mixed with
synthetic site noise (pink noise, engine hum, random impacts) at 10 dB SNR.  It measures what the
speech step adds to the typed results; it is not a test with real officers on a real site.

The typed sentences are parsed again in the same run, so typed and spoken are compared like for like.
Nothing here is tuned: the sentence sets are the held-out ones and every setting is the shipped one.

    python scripts/eval_spoken_rules.py           # needs Ollama (qwen2.5:1.5b), ffmpeg, edge-tts, a GPU helps
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import eval_rules as E  # noqa: E402
from sayguard.asr import SAMPLE_RATE, SpeechToText, decode_audio  # noqa: E402
from sayguard.config import load_config, resolve, work_dir  # noqa: E402
from sayguard.llm import LLMClient  # noqa: E402
from sayguard.rules import parser as P  # noqa: E402
from sayguard.server import site_vocabulary  # noqa: E402

# zh-TW-YunJheNeural (male) returned no audio from the service in Oct 2026, so both Taiwan voices are female
VOICES = {"zh": ["zh-TW-HsiaoChenNeural", "zh-TW-HsiaoYuNeural"], "en": ["en-US-AriaNeural", "en-US-GuyNeural"]}
SNR_DB = 10.0


def site_noise(n: int, seed: int) -> np.ndarray:
    """Synthetic construction/warehouse noise: pink noise, a slowly varying engine hum and impacts."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    pink = np.fft.irfft(np.fft.rfft(white) / np.sqrt(np.maximum(f, 20.0)), n)
    t = np.arange(n) / SAMPLE_RATE
    base = rng.uniform(45, 90)
    hum = sum(np.sin(2 * np.pi * base * k * t + rng.uniform(0, 6.3)) / k for k in range(1, 6))
    hum *= 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t)
    clank = np.zeros(n)
    for start in rng.integers(0, max(1, n - 4000), size=max(1, n // SAMPLE_RATE)):
        k = np.arange(4000)
        clank[start:start + 4000] += np.exp(-k / 600) * np.sin(2 * np.pi * rng.uniform(800, 2500) * k / SAMPLE_RATE)
    mix = pink / np.std(pink) + 0.7 * hum / np.std(hum) + 0.8 * clank / (np.std(clank) + 1e-9)
    return (mix / np.std(mix)).astype(np.float32)


def with_noise(speech: np.ndarray, seed: int) -> np.ndarray:
    noise = site_noise(len(speech), seed)
    p_s = float(np.mean(speech ** 2)) + 1e-12
    noise *= np.sqrt(p_s / (10 ** (SNR_DB / 10)))
    out = speech + noise
    return (out / max(1.0, float(np.max(np.abs(out))))).astype(np.float32)


def wav_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


async def tts(text: str, voice: str, tries: int = 4) -> bytes:
    """MP3 bytes from the online voice; retried, since the service drops a request now and then."""
    import edge_tts
    for i in range(tries):
        try:
            out = b""
            async for chunk in edge_tts.Communicate(text, voice).stream():
                if chunk["type"] == "audio":
                    out += chunk["data"]
            if out:
                return out
        except edge_tts.exceptions.NoAudioReceived:
            pass
        await asyncio.sleep(2 + 3 * i)
    raise RuntimeError(f"no audio from {voice} after {tries} tries")


def cer(ref: str, hyp: str) -> float:
    """Character error rate after dropping spaces and punctuation, case-insensitive."""
    norm = lambda s: [c for c in s.lower() if c.isalnum()]  # noqa: E731
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[len(h)] / max(1, len(r))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="sentences per set (smoke test)")
    a = ap.parse_args()

    site_cfg, wh_cfg = load_config(), load_config(str(ROOT / "configs" / "warehouse.yaml"))
    data = json.loads((ROOT / "tests" / "site_rule_sentences.json").read_text(encoding="utf-8"))
    keys = ("type", "zones", "object", "threshold", "time_window", "min_duration_s")
    sets = [("hard-hat held-out", site_cfg, E.HELDOUT, E.matches),
            ("warehouse held-out v1", wh_cfg, [(c["text"], {k: c[k] for k in keys}) for c in data["heldout"]], E.site_matches),
            ("warehouse held-out v2", wh_cfg, [(c["text"], {k: c[k] for k in keys}) for c in data["heldout2"]], E.site_matches)]
    llm = LLMClient(site_cfg["llm"]["base_url"], site_cfg["llm"]["model"], 120)
    if not llm.available():
        raise SystemExit("LLM server not reachable: start Ollama first")
    out = {"about": __doc__.split("\n\n")[1].replace("\n", " "), "llm": site_cfg["llm"]["model"],
           "asr": str(site_cfg["asr"]["model"]), "voices": VOICES, "noise_snr_db": SNR_DB, "sets": {}}
    for name, cfg, cases, check in sets:
        cases = cases[:a.limit]
        cams = [{"id": c["id"], "name": c["name"], "zone": c["zone"]} for c in cfg["cameras"]]
        E.CAMS[:] = cams
        zones = cfg["zones"]
        stt = SpeechToText(str(resolve(cfg["asr"]["model"])), "zh", vocabulary=site_vocabulary(cfg))
        score = {"typed": 0}
        n_audio, cers, rows = {}, {}, []
        for k, (text, exp) in enumerate(cases):
            lang = "zh" if P._is_zh(text) else "en"
            typed = P.parse(text, cams, zones, llm=llm).rule
            ok_typed, _ = check(typed.model_dump() if typed else None, exp)
            score["typed"] += ok_typed
            row = {"text": text, "lang": lang, "typed_ok": ok_typed, "spoken": []}
            for v, voice in enumerate(VOICES[lang]):
                clean = decode_audio(asyncio.run(tts(text, voice)), "tts.mp3")
                for cond, audio in (("clean", clean), ("noisy", with_noise(clean, seed=1000 * k + v))):
                    key = f"{cond}"
                    t0 = time.perf_counter()
                    heard = stt.transcribe(wav_bytes(audio), "speech.wav", lang)["text"]
                    asr_ms = (time.perf_counter() - t0) * 1000
                    rule = P.parse(heard, cams, zones, llm=llm).rule
                    ok, bad = check(rule.model_dump() if rule else None, exp)
                    score[key] = score.get(key, 0) + ok
                    n_audio[key] = n_audio.get(key, 0) + 1
                    cers.setdefault(key, []).append(cer(text, heard))
                    row["spoken"].append({"voice": voice, "condition": cond, "heard": heard, "ok": ok,
                                          "wrong_fields": bad, "asr_ms": round(asr_ms)})
            rows.append(row)
            print(f"[{name}] {k + 1}/{len(cases)} typed={ok_typed} " +
                  " ".join(f"{s['condition'][0]}{int(s['ok'])}" for s in row["spoken"]), flush=True)
        n = len(cases)
        res = {"sentences": n, "typed_correct": score["typed"], "typed_accuracy": round(score["typed"] / n, 3)}
        for key in n_audio:
            res[f"spoken_{key}"] = {"recordings": n_audio[key], "correct": score[key],
                                    "accuracy": round(score[key] / n_audio[key], 3),
                                    "mean_cer": round(float(np.mean(cers[key])), 3)}
        res["rows"] = rows
        out["sets"][name] = res
        print(name, {k: v for k, v in res.items() if k != "rows"}, flush=True)
    for d in (work_dir() / "results", ROOT / "docs" / "results"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "rule_parsing_spoken.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
