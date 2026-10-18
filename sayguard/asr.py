"""Speech -> text for spoken safety rules.

Whisper-Small is in the Hailo GenAI Model Zoo for Hailo-10H, so in Stage II the
same model moves onto the UGen300; on the PC it runs through transformers.  It
loads on the first request (most sessions never speak a rule) and only after the
memory guard agrees; when the GPU is full but RAM is not, it runs on the CPU
instead of refusing.  Browsers record webm/opus, ogg, mp4 or wav, so ffmpeg turns
everything into the 16 kHz mono float32 Whisper expects.  Whisper tends to write
Mandarin in Simplified characters; site managers here read Traditional (Taiwan).
torch/transformers are imported lazily so decoding can be tested without them.

The language is chosen per request (the dashboard sends its UI language).  A forced
wrong language makes Whisper translate or garble the rule, and its own detector picks
from ~100 languages, so a short accented sentence can come out as Japanese or
Cantonese; "auto" therefore only compares the Mandarin and English language tokens.
"""
from __future__ import annotations

import gc
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import resources

log = logging.getLogger("sayguard.asr")

SAMPLE_RATE = 16000
MIN_SECONDS = 0.3
MAX_SECONDS = 30.0
FFMPEG_TIMEOUT_S = 30.0
_CJK = re.compile(r"[\u3400-\u9fff]")
_SUFFIX = re.compile(r"\.[a-z0-9]{1,8}")
AUTO_LANGUAGES = ("zh", "en")   # what "auto" chooses between: the site speaks Mandarin or English


def _ffmpeg(src: str, data: bytes | None, timeout_s: float) -> subprocess.CompletedProcess:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", src,
           "-t", f"{MAX_SECONDS:g}", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    stdin = {"input": data} if data is not None else {"stdin": subprocess.DEVNULL}
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout_s, check=False,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **stdin)
    except FileNotFoundError:
        raise ValueError("ffmpeg is not on PATH; it is needed to decode audio") from None
    except subprocess.TimeoutExpired:
        raise ValueError(f"ffmpeg did not decode the audio within {timeout_s:.0f} s") from None


def _ffmpeg_from_file(data: bytes, suffix: str, timeout_s: float) -> subprocess.CompletedProcess:
    """Safety net for inputs ffmpeg can only probe when it can seek (odd mp4/mov muxes)."""
    suffix = suffix if _SUFFIX.fullmatch(suffix) else ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="sayguard_asr_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return _ffmpeg(path, None, timeout_s)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _release_partial() -> None:
    """Return memory a failed model load left behind (never imports torch just for this)."""
    torch = sys.modules.get("torch")
    try:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _error_text(proc: subprocess.CompletedProcess) -> str:
    lines = proc.stderr.decode("utf-8", "replace").strip().splitlines()
    return lines[-1] if lines else f"ffmpeg exit code {proc.returncode}"


def decode_audio(data: bytes, filename: str = "", timeout_s: float = FFMPEG_TIMEOUT_S) -> np.ndarray:
    """Any audio ffmpeg can read -> mono 16 kHz float32 samples, truncated to 30 s."""
    if not data:
        raise ValueError("empty audio")
    proc = _ffmpeg("pipe:0", data, timeout_s)
    if proc.returncode != 0 or not proc.stdout:
        proc = _ffmpeg_from_file(data, Path(filename or "").suffix.lower(), timeout_s)
    if proc.returncode != 0:
        raise ValueError(f"could not decode audio ({filename or 'upload'}): {_error_text(proc)}")
    raw = proc.stdout[: len(proc.stdout) // 4 * 4]
    audio = np.frombuffer(raw, dtype="<f4")
    if audio.size < MIN_SECONDS * SAMPLE_RATE:
        raise ValueError(f"audio too short ({audio.size / SAMPLE_RATE:.2f} s); "
                         f"need at least {MIN_SECONDS} s of speech")
    return audio[: int(MAX_SECONDS * SAMPLE_RATE)].astype(np.float32)


class SpeechToText:
    def __init__(self, model_dir: str, language: str | None = "zh", vocabulary: dict[str, str] | None = None):
        """vocabulary: {"zh": "...", "en": "..."} site words passed to Whisper as a decoder prompt.
        Without it short rules lose their key words ("A區" came back as "H" in testing)."""
        self.model_dir = model_dir
        self.language = language or None
        self.vocabulary = {k: v for k, v in (vocabulary or {}).items() if v}
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._converter = None
        self._state = "idle"
        self._device: str | None = None
        self._reason = ""
        self._calls = 0
        self._total_ms = 0.0

    def status(self) -> dict:
        avg = self._total_ms / self._calls if self._calls else 0.0
        return {"state": self._state, "device": self._device, "reason": self._reason,
                "calls": self._calls, "avg_ms": round(avg, 1)}

    def transcribe(self, data: bytes, filename: str = "audio.webm", language: str | None = None) -> dict:
        """Decode first so a bad upload never triggers a model load; ms excludes the one-off load.
        language: "zh" | "en" | "auto" (Mandarin or English, detected); None -> the configured default."""
        language = self._pick_language(language)
        t0 = time.perf_counter()
        audio = decode_audio(data, filename)
        decode_s = time.perf_counter() - t0
        with self._lock:
            self._ensure_loaded()
            t1 = time.perf_counter()
            try:
                text, lang = self._infer(audio, language)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                raise resources.ResourceError(f"ASR ran out of memory on {self._device}: {e}") from e
            ms = int((decode_s + time.perf_counter() - t1) * 1000)
            self._calls += 1
            self._total_ms += ms
        return {"text": text, "language": lang, "ms": ms, "device": self._device}

    def _pick_language(self, language: str | None) -> str:
        """Request value -> a Whisper language code or "auto"; ValueError (HTTP 400) for anything else.
        A configured default of None means "auto" too."""
        if not language:
            return self.language or "auto"
        code = language.strip().lower().replace("_", "-").split("-")[0]   # "zh-TW" -> "zh"
        if code == "auto" or code in AUTO_LANGUAGES:
            return code
        raise ValueError(f"unsupported language {language!r}; use one of {', '.join(AUTO_LANGUAGES)} or auto")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._state, self._reason = "loading", ""
        failed: str | None = None
        try:
            device, note = self._pick_device()
            self._load(device)
        except resources.ResourceError as e:
            self._state, self._reason = "error", str(e)
            raise
        except Exception as e:
            failed = f"{type(e).__name__}: {e}"
            log.warning("ASR model failed to load", exc_info=True)
        if failed is not None:
            # Clean up only after the handler has exited: until then e.__traceback__ pins the
            # from_pretrained frames and the half-loaded model, so gc/empty_cache free nothing
            # and the VRAM stays reserved for every later require() (same fix as verify.py).
            self._model = self._processor = None
            self._state, self._reason = "error", failed
            _release_partial()
            raise resources.ResourceError(f"ASR model failed to load: {failed}")
        self._state, self._device, self._reason = "ready", device, note

    def _pick_device(self) -> tuple[str, str]:
        import torch
        try:
            resources.require("ASR", gpu_gb=1.5, ram_gb=3.0)
        except resources.ResourceError as gpu_err:
            try:
                resources.require("ASR (CPU)", ram_gb=3.0)
            except resources.ResourceError:
                raise gpu_err from None
            return "cpu", f"running on CPU: {gpu_err}"
        return ("cuda", "") if torch.cuda.is_available() else ("cpu", "no CUDA device")

    def _load(self, device: str) -> None:
        import torch
        from opencc import OpenCC
        with resources.IMPORT_LOCK:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if not Path(self.model_dir).is_dir():
            raise resources.ResourceError(f"ASR model folder not found: {self.model_dir}")
        dtype = torch.float16 if device == "cuda" else torch.float32
        processor = WhisperProcessor.from_pretrained(self.model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_dir, dtype=dtype, device_map=device)
        model.eval()
        self._converter = OpenCC("s2twp")
        self._processor, self._model = processor, model

    def _detect_zh_en(self, feats) -> str | None:
        """Mandarin or English, whichever language token scores higher.  Same single decoder
        step as WhisperGenerationMixin.detect_language (start token -> logits of the next
        token), but the argmax is taken over <|zh|> and <|en|> only.  None (let generate()
        decide) if the checkpoint has no such tokens, e.g. an English-only model."""
        import torch

        gc = self._model.generation_config
        lang_to_id = getattr(gc, "lang_to_id", None) or {}
        ids = [lang_to_id.get(f"<|{c}|>") for c in AUTO_LANGUAGES]
        if None in ids:
            return None
        start = torch.full((feats.shape[0], 1), gc.decoder_start_token_id, dtype=torch.long, device=self._model.device)
        logits = self._model(input_features=feats[:, :, :3000], decoder_input_ids=start, use_cache=False).logits[:, -1]
        return AUTO_LANGUAGES[int(logits[0, ids].argmax())]

    def _infer(self, audio: np.ndarray, language: str) -> tuple[str, str | None]:
        import torch

        feats = self._processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE,
                                                  return_tensors="pt").input_features
        feats = feats.to(self._model.device, dtype=self._model.dtype)
        with torch.inference_mode():
            lang = self._detect_zh_en(feats) if language == "auto" else language
            vocab = self.vocabulary.get(lang or "", "")
            kwargs = {}
            if vocab:
                kwargs["prompt_ids"] = self._processor.get_prompt_ids(vocab, return_tensors="pt").to(self._model.device)
            ids = self._model.generate(feats, language=lang, task="transcribe", **kwargs)
        text = self._processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if vocab and text.startswith(vocab.strip()):   # some versions return the prompt with the transcript
            text = text[len(vocab.strip()):].strip()
        if _CJK.search(text):
            text = self._converter.convert(text).strip()
        return text, lang
