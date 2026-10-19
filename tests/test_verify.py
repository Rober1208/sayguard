"""Alert verifier: probe math, backend selection, and the worker's verdict bookkeeping.
No model is loaded: the worker runs against a fake backend."""
from __future__ import annotations

import time

import numpy as np
import pytest

from sayguard.types import Event
from sayguard.verify import (ClipProbeBackend, LinearProbe, Verifier, VlmBackend, make_backend,
                             parse_yes_no)


def ident(p):
    return p


def test_parse_yes_no():
    assert parse_yes_no("Yes.") is True
    assert parse_yes_no(" no, the person") is False
    assert parse_yes_no("是的") is True
    assert parse_yes_no("沒有") is False
    assert parse_yes_no("maybe") is None


def test_probe_uses_unit_features_and_scale():
    W = np.zeros((2, 4), np.float32)
    W[1, 0] = 1.0                       # class 1 (wearing) reads feature 0
    probe = LinearProbe(W, np.zeros(2), scale=10.0)
    f = np.array([[3.0, 0, 0, 0], [0, 5.0, 0, 0]])
    p = probe.prob_wearing(f)
    # unit(f) @ W.T * 10 -> logits (0, 10) and (0, 0): magnitude of the raw feature must not matter
    assert p[0] == pytest.approx(1 / (1 + np.exp(-10)), rel=1e-5)
    assert p[1] == pytest.approx(0.5)
    assert np.allclose(probe.prob_wearing(f * 100), p)


def test_probe_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    probe = LinearProbe(rng.normal(size=(2, 512)), rng.normal(size=2), 10.0)
    path = tmp_path / "probe.npz"
    probe.save(path, val_accuracy=np.float32(0.9))
    again = LinearProbe.load(path)
    x = rng.normal(size=(5, 512))
    assert np.allclose(probe.prob_wearing(x), again.prob_wearing(x))


def test_make_backend_selection():
    assert isinstance(make_backend({"backend": "clip_probe"}, ident), ClipProbeBackend)
    assert isinstance(make_backend({"backend": "vlm", "vlm_model": "m"}, ident), VlmBackend)
    assert isinstance(make_backend({}, ident), ClipProbeBackend)
    # a config written before the probe existed: `vlm: {model: ...}` keeps meaning the VLM
    old = make_backend({"enabled": True, "model": "qwen"}, ident)
    assert isinstance(old, VlmBackend) and old.model_dir == "qwen"
    with pytest.raises(ValueError):
        make_backend({"backend": "gpt"}, ident)


class FakeBackend:
    label = "Fake"
    device = "cpu"

    def __init__(self, answers, fail=False):
        self.answers, self.fail, self.loads = list(answers), fail, 0

    def load(self):
        self.loads += 1
        if self.fail:
            raise FileNotFoundError("no weights")

    def judge(self, crop):
        return self.answers.pop(0), "p=0.9"

    def unload(self):
        pass


class Store:
    def __init__(self):
        self.marks = {}

    def set_verified(self, eid, verified, note):
        self.marks[eid] = (verified, note)


def ev(i, rule_type="no_helmet"):
    return Event(id=f"e{i}", ts=0.0, cam_id="cam01", cam_name="Gate", zone="entrance", rule_id="r",
                 rule_name="r", rule_type=rule_type, severity="high", message="m")


def run(v, jobs, n):
    v.start()
    for e, crop in jobs:
        v.submit(e, crop)
    t0 = time.time()
    while len(v.store.marks) < n and time.time() - t0 < 5:
        time.sleep(0.01)
    v.stop()


def test_worker_verdicts():
    store = Store()
    crop = np.zeros((32, 32, 3), np.uint8)
    v = Verifier(FakeBackend([False, True, None]), store)
    run(v, [(ev(1), crop), (ev(2), crop), (ev(3), crop), (ev(4, "zone"), crop), (ev(5), np.zeros((4, 4, 3)))], 5)
    m = store.marks
    assert m["e1"][0] is True and "confirmed" in m["e1"][1]          # bare head -> violation confirmed
    assert m["e2"][0] is False and "false alarm" in m["e2"][1]       # wearing -> filtered
    assert m["e3"][0] is None and "unclear" in m["e3"][1]
    assert m["e4"] == (None, "not checked")                          # only hard-hat alerts
    assert m["e5"][0] is None and "no usable crop" in m["e5"][1]
    s = v.status()
    assert (s["checked"], s["confirmed"], s["false_alarms"], s["backend"]) == (3, 1, 1, "Fake")


def test_worker_unavailable_backend_leaves_alerts_unchecked():
    store = Store()
    fb = FakeBackend([], fail=True)
    v = Verifier(fb, store)
    run(v, [(ev(1), np.zeros((32, 32, 3), np.uint8)), (ev(2), np.zeros((32, 32, 3), np.uint8))], 2)
    assert all(verdict is None and "skipped" in note for verdict, note in store.marks.values())
    assert v.status()["state"] == "unavailable"
    assert fb.loads == 1                     # no reload storm: retries wait RETRY_S


def test_disabled_verifier_marks_without_loading():
    store = Store()
    fb = FakeBackend([])
    v = Verifier(fb, store, enabled=False)
    v.submit(ev(1), np.zeros((32, 32, 3), np.uint8))
    assert store.marks["e1"] == (None, "Fake disabled") and fb.loads == 0
