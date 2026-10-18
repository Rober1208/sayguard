"""CameraTracker on synthetic head boxes.

The tracker decides which detections are the same worker, and the no-helmet rule
counts that worker's label history, so identity across frames, the helmet/head
label flip, forgetting and the merge of overlapping helmet+head boxes are pinned
here. Pure numpy; no detector or model is involved.
"""
from __future__ import annotations

import pytest

from sayguard.tracking import FORGET_S, HISTORY, LOST_S, CameraTracker
from sayguard.types import Detection, Person

DT = 0.1                  # 10 fps
T0 = 1_000.0


def head(x: float, y: float = 100.0, size: float = 40.0, label: str = "head", conf: float = 0.8) -> Detection:
    return Detection((x, y, x + size, y + size), label, conf)


def ids(people: list[Person]) -> list[int]:
    return [p.track_id for p in people]


def test_same_head_keeps_id_across_frames():
    tr = CameraTracker(10.0)
    p0, b0 = tr.update([head(100)], T0)
    p1, b1 = tr.update([head(102)], T0 + DT)
    assert ids(p0) == ids(p1) == [1]
    assert [b.track_id for b in b0] == [b.track_id for b in b1] == [1]
    assert p1[0].observations == 2
    assert b1[0].xyxy == (102.0, 100.0, 142.0, 140.0)       # the detection's own box is returned


def test_two_heads_get_two_ids():
    tr = CameraTracker()
    p0, _ = tr.update([head(100), head(400)], T0)
    p1, _ = tr.update([head(401), head(101)], T0 + DT)       # order of detections does not matter
    assert ids(p0) == [1, 2]
    assert {p.track_id: p.head[0] for p in p1} == {1: 101.0, 2: 401.0}


def test_new_track_is_reported_on_its_first_frame():
    tr = CameraTracker()
    tr.update([head(100)], T0)
    people, boxes = tr.update([head(100), head(600)], T0 + DT)
    assert ids(people) == [1, 2]
    assert [b.track_id for b in boxes] == [1, 2]


@pytest.mark.parametrize("size", [40.0, 24.0])   # 24 px heads overlap < MATCH_IOU at 20 px/frame
def test_moving_head_keeps_id(size: float):
    tr = CameraTracker()
    for i in range(30):
        people, _ = tr.update([head(100 + 20 * i, size=size)], T0 + i * DT)
        assert ids(people) == [1]


def test_two_heads_walking_past_each_other_keep_ids():
    tr = CameraTracker()
    for i in range(12):
        a, b = head(100 + 20 * i, y=100), head(400 - 20 * i, y=150)   # 10 px vertical gap
        people, _ = tr.update([a, b], T0 + i * DT)
        assert {p.track_id: p.head[1] for p in people} == {1: 100.0, 2: 150.0}


def test_label_flip_keeps_id_and_updates_ratio():
    tr = CameraTracker()
    labels = ["helmet", "helmet", "head", "helmet", "head", "head"]
    for i, lab in enumerate(labels):
        people, boxes = tr.update([head(100, label=lab)], T0 + i * DT)
        assert ids(people) == [1]
        assert people[0].helmet_now == (lab == "helmet")
        assert boxes[0].label == lab
        bare = sum(1 for x in labels[: i + 1] if x == "head")
        assert people[0].observations == i + 1
        assert people[0].no_helmet_ratio == pytest.approx(bare / (i + 1))


def test_history_is_capped():
    tr = CameraTracker()
    for i in range(HISTORY + 5):
        people, _ = tr.update([head(100, label="helmet" if i < 5 else "head")], T0 + i * DT)
    assert people[0].observations == HISTORY
    assert people[0].no_helmet_ratio == 1.0                 # the 5 helmet frames fell out of the window


def test_track_kept_within_lost_s_and_dropped_after():
    tr = CameraTracker()
    tr.update([head(100)], T0)
    people, _ = tr.update([head(100)], T0 + LOST_S - 0.1)   # short dropout: same worker
    assert ids(people) == [1]
    people, _ = tr.update([head(100)], T0 + 2 * LOST_S + 0.1)
    assert ids(people) == [2]                               # gone > LOST_S: new id, fresh history
    assert people[0].observations == 1


def test_history_forgotten_after_forget_s():
    tr = CameraTracker()
    tr.update([head(100)], T0)
    tr.update([head(500)], T0 + 1.0)
    assert set(tr.hist) == {1, 2}
    tr.update([], T0 + FORGET_S + 0.5)                      # track 1 last seen > FORGET_S ago
    assert set(tr.hist) == {2}


def test_empty_frames_keep_tracks():
    tr = CameraTracker()
    assert tr.update([], T0) == ([], [])
    tr.update([head(100)], T0 + DT)
    assert tr.update([], T0 + 2 * DT) == ([], [])
    people, _ = tr.update([head(101)], T0 + 3 * DT)
    assert ids(people) == [1]


def test_non_ppe_labels_are_ignored():
    tr = CameraTracker()
    people, boxes = tr.update([Detection((0, 0, 50, 50), "forklift", 0.9), head(100)], T0)
    assert ids(people) == [1]
    assert [b.label for b in boxes] == ["head"]


def test_overlapping_helmet_and_head_merge_into_one_track():
    tr = CameraTracker()
    dets = [head(100, label="head", conf=0.55), head(102, label="helmet", conf=0.85)]
    people, boxes = tr.update(dets, T0)
    assert ids(people) == [1]
    assert [(b.label, b.conf) for b in boxes] == [("helmet", 0.85)]   # higher score wins
    assert people[0].helmet_now and people[0].no_helmet_ratio == 0.0


def test_low_confidence_extends_but_never_starts_a_track():
    tr = CameraTracker()
    people, _ = tr.update([head(100, conf=0.2)], T0)
    assert people == []
    tr.update([head(100, conf=0.8)], T0 + DT)
    people, boxes = tr.update([head(102, conf=0.2), head(600, conf=0.2)], T0 + 2 * DT)
    assert ids(people) == [1]
    assert boxes[0].conf == 0.2
