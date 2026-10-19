"""Dashboard metrics: frames analysed per second and the UGen300 load estimate."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from sayguard.pipeline import Pipeline


def test_frames_per_second_has_no_fencepost_error(monkeypatch):
    # 10 cameras, one 10-frame batch per 10 Hz tick: 100 frames/s, not 105 (N batches over N-1 intervals)
    fake = SimpleNamespace(_win=deque(), stats={"frames_per_s": 0.0, "det_batch_ms": 0.0, "batch_size": 0.0,
                                                "det_capacity_fps": 0.0})
    now = [100.0]
    monkeypatch.setattr("sayguard.pipeline.time.time", lambda: now[0])
    for i in range(60):
        now[0] = 100.0 + i * 0.1
        Pipeline._record_throughput(fake, 10, 0.035)
    assert fake.stats["frames_per_s"] == pytest.approx(100.0, abs=0.5)


def test_first_batch_reports_no_rate(monkeypatch):
    fake = SimpleNamespace(_win=deque(), stats={"frames_per_s": 0.0, "det_batch_ms": 0.0, "batch_size": 0.0,
                                                "det_capacity_fps": 0.0})
    monkeypatch.setattr("sayguard.pipeline.time.time", lambda: 5.0)
    Pipeline._record_throughput(fake, 10, 0.035)
    assert fake.stats["frames_per_s"] == 0.0


def _estimate(imgsz):
    cams = {f"c{i}": SimpleNamespace(fps=10.0) for i in range(10)}
    fake = SimpleNamespace(states=cams, _stale=lambda st, now: False,
                           cfg={"detector": {"imgsz": imgsz}, "hailo_reference": {"fps_batch1": 166, "fps_batch8": 252}})
    return Pipeline.hailo_estimate(fake)


def test_load_estimate_only_for_the_reference_input_size():
    hard_hat = _estimate(640)
    assert hard_hat["comparable"] and hard_hat["load_batch8_pct"] == pytest.approx(39.7, abs=0.1)
    warehouse = _estimate(960)           # the 960 px site detector: Hailo's 640 px figures do not apply
    assert not warehouse["comparable"] and warehouse["load_batch8_pct"] is None
    assert warehouse["demand_fps"] == 100.0
