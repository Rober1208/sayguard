"""ffmpeg decoding for the ASR endpoint, tested without torch/transformers so it
runs while the GPU is busy: browsers send arbitrary sample rates and channel
counts, and Whisper only accepts 16 kHz mono float32 of at most 30 s."""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import wave

import numpy as np
import pytest

from sayguard import asr

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def make_wav(seconds: float, rate: int = 8000, channels: int = 2, freq: float = 440.0) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    tone = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    frames = np.repeat(tone[:, None], channels, axis=1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames.tobytes())
    return buf.getvalue()


def test_import_stays_light():
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_resamples_to_16k_mono_float32():
    audio = asr.decode_audio(make_wav(1.0), "clip.wav")
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(audio.size - asr.SAMPLE_RATE) <= 160
    assert 0.3 < float(np.abs(audio).max()) <= 1.0  # stereo downmix is -3 dB per channel


def test_mono_keeps_amplitude():
    audio = asr.decode_audio(make_wav(1.0, rate=16000, channels=1), "mono.wav")
    assert audio.size == asr.SAMPLE_RATE
    assert 0.45 < float(np.abs(audio).max()) < 0.55


@pytest.mark.parametrize("name,codec", [("rec.webm", "libopus"), ("rec.ogg", "libopus"), ("rec.mp4", "aac")])
def test_browser_containers(tmp_path, name, codec):
    """What MediaRecorder produces: Chrome/Firefox webm or ogg with Opus, Safari mp4 with AAC."""
    src, dst = tmp_path / "in.wav", tmp_path / name
    src.write_bytes(make_wav(1.0, rate=48000, channels=1))
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                           "-c:a", codec, str(dst)], capture_output=True, timeout=30)
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg cannot encode {codec}: {proc.stderr[-200:]!r}")
    audio = asr.decode_audio(dst.read_bytes(), name)
    assert abs(audio.size - asr.SAMPLE_RATE) < 0.1 * asr.SAMPLE_RATE
    assert float(np.abs(audio).max()) > 0.3


def test_truncates_to_30_seconds():
    audio = asr.decode_audio(make_wav(31.0, channels=1), "long.wav")
    assert audio.size == int(asr.MAX_SECONDS * asr.SAMPLE_RATE)


def test_too_short_is_rejected():
    with pytest.raises(ValueError, match="too short"):
        asr.decode_audio(make_wav(0.1), "short.wav")


def test_garbage_is_rejected():
    with pytest.raises(ValueError, match="could not decode"):
        asr.decode_audio(b"definitely not audio" * 50, "noise.webm")


def test_empty_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        asr.decode_audio(b"", "empty.webm")
