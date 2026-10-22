"""RuleEngine on synthetic tracks.

The engine decides when a camera raises an alert, so its timing (min duration,
dropout grace, cooldown), scope filters, time windows and the head-height
distance scale are pinned here with hand-built Person/Detection objects; no
detector or model is involved.  rules.json survival (crash mid-save, locked or
damaged file, invalid items) is pinned too: a bad file must never stop startup.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from sayguard.config import DEFAULT_CONFIG, ROOT
from sayguard.rules import CameraInfo, Rule, RuleEngine, TimeWindow
from sayguard.rules import engine as engine_mod
from sayguard.rules.engine import GRACE_S, MAX_GRACE_S, NO_HELMET_MIN_OBS, NO_HELMET_RATIO, grace_for, rect_gap
from sayguard.types import HEAD_HEIGHT_M, Detection, Person

FRAME = (1920, 1080)
T0 = 1_000_000.0          # arbitrary epoch; only time-window tests need a real local time
HEAD_46 = (100.0, 100.0, 146.0, 146.0)   # 46 px tall head -> 0.23 m / 46 px = 0.005 m/px
CAMS = [
    CameraInfo("cam01", "Gate A", "entrance"),
    CameraInfo("cam03", "Scaffold North", "zone_a"),
    CameraInfo("cam07", "Warehouse Aisle", "warehouse"),
    CameraInfo("cam10", "Site Office", "office"),
]


def person(tid: int = 1, head: tuple = HEAD_46, ratio: float = 1.0, obs: int = 10) -> Person:
    return Person(track_id=tid, head=head, helmet_now=ratio < 0.5, no_helmet_ratio=ratio, observations=obs)


def make_engine(*rules: Rule, path: Path | None = None) -> RuleEngine:
    eng = RuleEngine(CAMS, path)
    for r in rules:
        eng.add(r)
    return eng


def fire_times(eng: RuleEngine, frames, cam: str = "cam03", frame_dt: float | None = None) -> list[float]:
    """frames: (offset_s, people, objects); returns offsets at which something fired."""
    out = []
    for dt, people, objects in frames:
        if eng.evaluate(cam, T0 + dt, people, objects, FRAME, frame_dt=frame_dt).fired:
            out.append(dt)
    return out


def steady(start: float, stop: float, step: float = 0.5, people=None, objects=()):
    n = int(round((stop - start) / step))
    return [(start + i * step, [person()] if people is None else people, list(objects)) for i in range(n + 1)]


def local_ts(h: int, m: int = 0) -> float:
    """Epoch seconds for a fixed local wall-clock time, so window tests do not depend on TZ."""
    return time.mktime((2026, 10, 18, h, m, 0, 0, 0, -1))


def test_rules_package_does_not_import_torch():
    # fresh interpreter, so modules imported by other test files cannot mask a regression
    code = ("import sys, sayguard.rules, sayguard.rules.parser; "
            "print(sorted({'torch', 'ultralytics', 'transformers'} & set(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# ---- no_helmet timing ----------------------------------------------------
def test_no_helmet_fires_once_after_min_duration():
    rule = Rule(name="hard hat", type="no_helmet", min_duration_s=2.0)
    eng = make_engine(rule)
    fired = []
    for dt, people, objects in steady(0, 4):
        res = eng.evaluate("cam03", T0 + dt, people, objects, FRAME)
        fired += [(dt, f) for f in res.fired]
        if dt < 2.0:
            assert not res.fired
            assert res.active["t1"][0][1] == pytest.approx(dt)
    assert [dt for dt, _ in fired] == [2.0]
    f = fired[0][1]
    assert (f.rule.id, f.cam_id, f.key, f.track_id) == (rule.id, "cam03", "t1", 1)
    assert f.box == person().body_box(*FRAME)


@pytest.mark.parametrize("obs, ratio, fires", [
    (NO_HELMET_MIN_OBS - 1, 1.0, False),            # too few observations of this track
    (10, NO_HELMET_RATIO - 0.01, False),            # mostly seen with a helmet
    (NO_HELMET_MIN_OBS, NO_HELMET_RATIO, True),     # both thresholds are inclusive
])
def test_no_helmet_needs_enough_evidence(obs, ratio, fires):
    eng = make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=2.0))
    frames = steady(0, 5, people=[person(obs=obs, ratio=ratio)])
    assert bool(fire_times(eng, frames)) is fires


def test_short_gap_keeps_timer():
    assert GRACE_S == 1.0
    eng = make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=2.0))
    frames = [(0.0, [person()], []), (0.5, [person()], []),
              (1.0, [], []),                       # one missed detection
              (1.4, [person()], []),               # back 0.9 s after last sighting (< GRACE_S)
              (1.9, [person()], []), (2.0, [person()], []), (2.5, [person()], [])]
    assert fire_times(eng, frames) == [2.0]


def test_long_gap_resets_timer():
    eng = make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=2.0))
    frames = [(0.0, [person()], []), (0.5, [person()], []),
              (1.0, [], []), (1.6, [], [])]        # gone for 1.1 s > GRACE_S
    frames += steady(2.0, 5.0)
    assert fire_times(eng, frames) == [4.0]


def test_gap_without_intermediate_frames_resets_timer():
    eng = make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=2.0))
    frames = [(0.0, [person()], []), (0.5, [person()], [])] + steady(2.0, 5.0)
    assert fire_times(eng, frames) == [4.0]


# ---- grace follows the camera's frame interval -------------------------------------
@pytest.mark.parametrize("frame_dt, grace", [
    (None, GRACE_S),            # interval not known yet (st.fps still 0)
    (0.1, GRACE_S),             # 10 fps: the 1 s floor applies
    (0.4, GRACE_S),
    (1.0, 2.5),                 # 2.5 frame intervals
    (1.5, 3.75),
    (2.0, MAX_GRACE_S),
    (30.0, MAX_GRACE_S),        # capped: a real stall still resets the timer
    (0.0, GRACE_S), (-1.0, GRACE_S), (float("nan"), GRACE_S), (float("inf"), GRACE_S),
])
def test_grace_for_frame_interval(frame_dt, grace):
    assert MAX_GRACE_S == 5.0
    assert grace_for(frame_dt) == pytest.approx(grace)


def test_slow_camera_still_fires_with_its_frame_interval():
    # processed every 1.5 s (CPU fallback / GPU contention): each frame used to start a new timer
    frames = steady(0, 12, step=1.5)
    assert fire_times(make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=2.0)), frames) == []
    eng = make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=2.0))
    assert fire_times(eng, frames, frame_dt=1.5) == [3.0]


def test_slow_camera_missed_detection_keeps_timer():
    # the grace must also apply in _expire, or the empty frame drops the hit
    eng = make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=2.0))
    frames = [(0.0, [person()], []), (1.5, [], []), (3.0, [person()], []), (4.5, [person()], [])]
    assert fire_times(eng, frames, frame_dt=1.5) == [3.0]
    eng = make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=2.0))
    assert fire_times(eng, frames) == []          # default 1 s grace: 1.5 s apart always resets


def test_grace_is_capped_for_very_slow_cameras():
    eng = make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=2.0))
    assert fire_times(eng, steady(0, 18, step=6.0), frame_dt=10.0) == []   # 6 s gaps > 5 s cap


def test_grace_is_per_camera():
    eng = make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=2.0))
    slow, fast = [], []
    for dt, people, objects in steady(0, 6, step=1.5):
        slow += [dt] if eng.evaluate("cam03", T0 + dt, people, objects, FRAME, frame_dt=1.5).fired else []
        fast += [dt] if eng.evaluate("cam07", T0 + dt, people, objects, FRAME).fired else []
    assert (slow, fast) == ([3.0], [])


# ---- cooldown ------------------------------------------------------------
def test_cooldown_while_condition_persists():
    eng = make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=0, cooldown_s=30))
    assert fire_times(eng, steady(0, 61, step=1.0)) == [0.0, 30.0, 60.0]


def test_cooldown_survives_subject_leaving_and_returning():
    eng = make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=0, cooldown_s=30))
    frames = [(0.0, [person()], [])] + steady(1, 5, step=1.0, people=[]) + steady(6, 31, step=1.0)
    assert fire_times(eng, frames) == [0.0, 30.0]


def test_cooldown_is_per_subject():
    eng = make_engine(Rule(name="hard hat", type="no_helmet", min_duration_s=0, cooldown_s=30))
    r0 = eng.evaluate("cam03", T0, [person(1)], [], FRAME)
    r1 = eng.evaluate("cam03", T0 + 1, [person(1), person(2, head=(400, 100, 446, 146))], [], FRAME)
    assert [f.track_id for f in r0.fired] == [1]
    assert [f.track_id for f in r1.fired] == [2]


# ---- scope filters ---------------------------------------------------------
@pytest.mark.parametrize("scope, expected", [
    ({}, {"cam01", "cam03", "cam07", "cam10"}),
    ({"zones": ["zone_a"]}, {"cam03"}),
    ({"zones": ["zone_a", "warehouse"]}, {"cam03", "cam07"}),
    ({"zones_exclude": ["office"]}, {"cam01", "cam03", "cam07"}),
    ({"cameras": ["cam01"]}, {"cam01"}),
    ({"cameras": ["cam01", "cam10"], "zones_exclude": ["office"]}, {"cam01"}),
    ({"enabled": False}, set()),
])
def test_scope_filters(scope, expected):
    eng = make_engine(Rule(name="presence", type="person_present", min_duration_s=0, **scope))
    fired = {c.id for c in CAMS if eng.evaluate(c.id, T0, [person()], [], FRAME).fired}
    assert fired == expected


@pytest.mark.parametrize("scope, expected", [
    ({}, ["cam01", "cam03", "cam07", "cam10"]),
    ({"zones": ["zone_a", "warehouse"]}, ["cam03", "cam07"]),
    ({"zones_exclude": ["office"]}, ["cam01", "cam03", "cam07"]),
    ({"cameras": ["cam10", "cam01"]}, ["cam01", "cam10"]),                 # camera order, not rule order
    ({"cameras": ["cam01"], "zones": ["zone_b"]}, []),                    # "cam01 B區...": cam01 is the entrance
    ({"zones": ["no_such_zone"]}, []),
    ({"cameras": ["cam99"]}, []),
    ({"enabled": False}, ["cam01", "cam03", "cam07", "cam10"]),          # scope only: enabled is ignored
    ({"time_window": TimeWindow(start="03:00", end="03:01")}, ["cam01", "cam03", "cam07", "cam10"]),
])
def test_scope_cameras(scope, expected):
    eng = make_engine()
    assert eng.scope_cameras(Rule(name="presence", type="person_present", **scope)) == expected


# ---- time windows ----------------------------------------------------------
@pytest.mark.parametrize("window, h, m, inside", [
    (("22:00", "06:00"), 23, 0, True),
    (("22:00", "06:00"), 3, 0, True),
    (("22:00", "06:00"), 22, 0, True),     # start is inclusive
    (("22:00", "06:00"), 5, 59, True),
    (("22:00", "06:00"), 6, 0, False),     # end is exclusive
    (("22:00", "06:00"), 12, 0, False),
    (("22:00", "06:00"), 21, 59, False),
    (("08:00", "17:00"), 8, 0, True),
    (("08:00", "17:00"), 16, 59, True),
    (("08:00", "17:00"), 17, 0, False),
    (("08:00", "17:00"), 7, 59, False),
    (("00:00", "24:00"), 0, 0, True),      # 24:00 is stored as 00:00: all day, not never
    (("00:00", "24:00"), 12, 0, True),
    (("00:00", "24:00"), 23, 59, True),
])
def test_time_window(window, h, m, inside):
    tw = TimeWindow(start=window[0], end=window[1])
    rule = Rule(name="night", type="person_present", min_duration_s=0, time_window=tw)
    eng = make_engine(rule)
    ts = local_ts(h, m)
    assert tw.contains(time.localtime(ts)) is inside
    assert bool(eng.evaluate("cam07", ts, [person()], [], FRAME).fired) is inside


def test_time_window_normalises_hhmm():
    tw = TimeWindow(start="6", end="24:00")
    assert (tw.start, tw.end) == ("06:00", "00:00")
    with pytest.raises(ValueError):
        TimeWindow(start="25:00", end="06:00")


# ---- people / object rules ---------------------------------------------------
def test_person_count_over():
    eng = make_engine(Rule(name="crowd", type="person_count_over", threshold=3, min_duration_s=0))
    three = [person(i, head=(100 * i, 100, 100 * i + 46, 146)) for i in range(1, 4)]
    assert not eng.evaluate("cam03", T0, three, [], FRAME).fired
    res = eng.evaluate("cam03", T0 + 0.5, three + [person(9, head=(900, 100, 946, 146))], [], FRAME)
    assert len(res.fired) == 1
    f = res.fired[0]
    assert (f.key, f.track_id, f.box, f.detail) == ("cam", None, None, "4 people > 3")


def test_person_present():
    eng = make_engine(Rule(name="presence", type="person_present", min_duration_s=0))
    assert not eng.evaluate("cam01", T0, [], [], FRAME).fired
    res = eng.evaluate("cam01", T0 + 0.5, [person()], [], FRAME)
    assert [(f.key, f.track_id, f.detail) for f in res.fired] == [("cam", None, "1 person(s)")]


def test_object_present():
    eng = make_engine(Rule(name="ladder", type="object_present", object="Ladder", min_duration_s=0))
    forklift = Detection((300.0, 300.0, 600.0, 500.0), "forklift", 0.8, source="ov")
    ladder = Detection((10.4, 20.6, 50.0, 200.0), "ladder", 0.5, source="ov")
    assert not eng.evaluate("cam03", T0, [], [forklift], FRAME).fired
    res = eng.evaluate("cam03", T0 + 0.5, [], [forklift, ladder], FRAME)
    assert len(res.fired) == 1
    assert res.fired[0].box == (10, 20, 50, 200)
    assert res.fired[0].detail == "ladder x1"


def test_objects_requested_per_camera():
    eng = make_engine(Rule(name="ladder", type="object_present", object="ladder", zones=["zone_a"]),
                      Rule(name="fork", type="object_near_person", object="forklift", threshold=3,
                           cameras=["cam07"]),
                      Rule(name="hat", type="no_helmet"))
    assert eng.objects_for_camera("cam03") == {"ladder"}
    assert eng.objects_for_camera("cam07") == {"forklift"}
    assert eng.objects_for_camera("cam01") == set()
    assert eng.all_objects() == {"ladder", "forklift"}


def forklift_beside(p: Person, gap_px: float) -> Detection:
    body = p.body_box(*FRAME)
    x1 = body[2] + gap_px
    return Detection((x1, body[1] + 50.0, x1 + 300.0, body[3] - 50.0), "forklift", 0.7, source="ov")


def test_head_height_gives_metres_per_pixel():
    assert person().head_h == 46
    assert person().metres_per_px == pytest.approx(HEAD_HEIGHT_M / 46)
    assert person().metres_per_px == pytest.approx(0.005)
    assert rect_gap((0, 0, 10, 10), (5, 5, 20, 20)) == 0
    assert rect_gap((0, 0, 10, 10), (13, 14, 20, 20)) == pytest.approx(5.0)


@pytest.mark.parametrize("threshold, fires", [(3.0, True), (1.5, True), (0.5, False)])
def test_object_near_person_uses_head_scale(threshold, fires):
    p = person()
    fork = forklift_beside(p, 200)          # 200 px * 0.005 m/px = 1.0 m
    assert rect_gap(p.body_box(*FRAME), fork.xyxy) == pytest.approx(200)
    eng = make_engine(Rule(name="fork", type="object_near_person", object="forklift",
                           threshold=threshold, min_duration_s=0))
    res = eng.evaluate("cam07", T0, [p], [fork], FRAME)
    assert bool(res.fired) is fires
    if fires:
        f = res.fired[0]
        assert (f.key, f.track_id, f.box) == ("t1", 1, p.body_box(*FRAME))
        assert f.detail == "forklift 1.0 m"
        (obox, pbox, dist), = res.links
        assert obox == tuple(int(v) for v in fork.xyxy) and pbox == p.body_box(*FRAME)
        assert dist == pytest.approx(1.0)
    else:
        assert res.links == [] and res.active == {}


def test_object_near_person_picks_nearest_and_ignores_other_labels():
    p = person()
    far, near = forklift_beside(p, 1000), forklift_beside(p, 400)   # 5 m and 2 m
    ladder = Detection(near.xyxy, "ladder", 0.9, source="ov")
    eng = make_engine(Rule(name="fork", type="object_near_person", object="forklift",
                           threshold=3, min_duration_s=0))
    assert not eng.evaluate("cam07", T0, [p], [far, ladder], FRAME).fired
    res = eng.evaluate("cam07", T0 + 0.5, [p], [far, near], FRAME)
    assert res.links[0][2] == pytest.approx(2.0)
    assert res.fired[0].detail == "forklift 2.0 m"


# ---- rule management and persistence -------------------------------------------
def saved(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_add_remove_enable_bump_version_and_persist(tmp_path):
    path = tmp_path / "state" / "rules.json"
    eng = RuleEngine(CAMS, path)
    assert eng.version == 0 and not path.exists()

    r = eng.add(Rule(name="倉庫堆高機", type="object_near_person", object="forklift", threshold=3))
    assert eng.version == 1
    assert [d["id"] for d in saved(path)] == [r.id]
    assert "倉庫堆高機" in path.read_text(encoding="utf-8")   # written as UTF-8, not \u escapes

    assert eng.set_enabled(r.id, False) is r and eng.version == 2
    assert saved(path)[0]["enabled"] is False
    assert eng.set_enabled("nope", True) is None and eng.version == 2

    assert eng.remove(r.id) is True and eng.version == 3
    assert saved(path) == []
    assert eng.remove(r.id) is False and eng.version == 3


def test_disabled_rule_stops_firing_and_clears_on_remove():
    rule = Rule(name="hard hat", type="no_helmet", min_duration_s=0, cooldown_s=0)
    eng = make_engine(rule)
    assert eng.evaluate("cam03", T0, [person()], [], FRAME).fired
    eng.set_enabled(rule.id, False)
    assert not eng.evaluate("cam03", T0 + 0.5, [person()], [], FRAME).fired
    eng.set_enabled(rule.id, True)
    eng.evaluate("cam03", T0 + 1.0, [person()], [], FRAME)
    eng.remove(rule.id)
    assert not any(k[0] == rule.id for k in eng._hits)
    assert eng.list() == []


def test_load_uses_defaults_once_then_the_saved_file(tmp_path):
    path = tmp_path / "rules.json"
    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))["default_rules"]
    eng = RuleEngine(CAMS, path)
    eng.load(defaults)
    assert eng.version == 1 and path.exists()
    (r,) = eng.list()
    assert (r.type, r.zones_exclude, r.severity) == ("no_helmet", ["office"], "high")
    assert not eng.applies(r, eng.cameras["cam10"]) and eng.applies(r, eng.cameras["cam03"])

    tw = TimeWindow(start="22:00", end="06:00")
    eng.add(Rule(name="night", type="person_present", zones=["warehouse"], time_window=tw))

    again = RuleEngine(CAMS, path)
    again.load([{"name": "ignored", "type": "person_present"}])   # file exists -> defaults ignored
    assert [x.model_dump() for x in again.list()] == [x.model_dump() for x in eng.list()]
    assert again.list()[1].time_window == tw


def test_load_without_file_or_defaults(tmp_path):
    eng = RuleEngine(CAMS, tmp_path / "rules.json")
    eng.load()
    assert eng.list() == [] and saved(tmp_path / "rules.json") == []
    mem = RuleEngine(CAMS)          # no rules_path: in-memory only
    mem.load([{"name": "hat", "type": "no_helmet"}])
    assert len(mem.list()) == 1 and mem.version == 1


# ---- crash-safe persistence -----------------------------------------------------------
DEFAULTS = [{"name": "hat", "type": "no_helmet"}]


def set_aside_files(path: Path) -> list[Path]:
    return sorted(path.parent.glob(path.name + ".bad-*"))


def test_save_failure_mid_write_keeps_the_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "rules.json"
    eng = make_engine(Rule(name="hat", type="no_helmet"), path=path)
    before = path.read_text(encoding="utf-8")

    def power_cut(fd):
        raise OSError("disk went away")
    monkeypatch.setattr(engine_mod.os, "fsync", power_cut)
    with pytest.raises(OSError):
        eng.add(Rule(name="crowd", type="person_count_over", threshold=3))
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["rules.json"]       # no stray .tmp


def test_save_retries_while_the_file_is_locked(tmp_path, monkeypatch):
    path = tmp_path / "rules.json"
    real_replace, calls, sleeps = engine_mod.os.replace, [], []

    def locked_twice(src, dst):
        calls.append(dst)
        if len(calls) <= 2:
            raise PermissionError(13, "in use by another process")
        real_replace(src, dst)
    monkeypatch.setattr(engine_mod.os, "replace", locked_twice)
    monkeypatch.setattr(engine_mod.time, "sleep", sleeps.append)
    r = make_engine(Rule(name="hat", type="no_helmet"), path=path).list()[0]
    assert len(calls) == 3 and len(sleeps) == 2
    assert [d["id"] for d in saved(path)] == [r.id]


def test_save_gives_up_when_the_file_stays_locked(tmp_path, monkeypatch):
    path = tmp_path / "rules.json"
    eng = make_engine(Rule(name="hat", type="no_helmet"), path=path)
    before = path.read_text(encoding="utf-8")

    def always_locked(src, dst):
        raise PermissionError(13, "in use by another process")
    monkeypatch.setattr(engine_mod.os, "replace", always_locked)
    monkeypatch.setattr(engine_mod.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        eng.add(Rule(name="crowd", type="person_count_over", threshold=3))
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["rules.json"]


@pytest.mark.parametrize("content", [
    b'[{"name": "hat", "type": "no_he',          # truncated by a crash mid-write
    b"\xff\xfe\x00\x00 not utf-8",
    b"",
    b'{"name": "hat", "type": "no_helmet"}',     # valid JSON, but not a list of rules
])
def test_unreadable_rules_file_is_set_aside_and_defaults_used(tmp_path, caplog, content):
    path = tmp_path / "rules.json"
    path.write_bytes(content)
    eng = RuleEngine(CAMS, path)
    with caplog.at_level(logging.ERROR, logger="sayguard.rules"):
        eng.load(DEFAULTS)
    assert [r.name for r in eng.list()] == ["hat"]
    assert [d["name"] for d in saved(path)] == ["hat"]
    (bad,) = set_aside_files(path)
    assert bad.read_bytes() == content
    assert any(rec.levelno == logging.ERROR and bad.name in rec.getMessage() for rec in caplog.records)


def test_invalid_rules_are_skipped_and_the_original_kept(tmp_path, caplog):
    path = tmp_path / "rules.json"
    good = Rule(name="crowd", type="person_count_over", threshold=3).model_dump()
    items = [good, {"name": "old", "type": "removed_rule_type"}, "not an object", {"type": "no_helmet"}]
    path.write_text(json.dumps(items), encoding="utf-8")
    eng = RuleEngine(CAMS, path)
    with caplog.at_level(logging.WARNING, logger="sayguard.rules"):
        eng.load(DEFAULTS)                                 # file exists -> defaults ignored
    assert [r.id for r in eng.list()] == [good["id"]]
    assert saved(path) == [Rule(**good).model_dump()]
    (bad,) = set_aside_files(path)
    assert json.loads(bad.read_text(encoding="utf-8")) == items
    assert sum(rec.levelno == logging.WARNING for rec in caplog.records) == 3


def test_invalid_default_rule_is_skipped(tmp_path):
    path = tmp_path / "rules.json"
    eng = RuleEngine(CAMS, path)
    eng.load([{"name": "typo", "type": "no_hlemet"}] + DEFAULTS)
    assert [r.name for r in eng.list()] == ["hat"]
    assert set_aside_files(path) == []


def test_track_id_switch_at_same_place_does_not_realert():
    """A new track id where a person was just alerted (occlusion, id switch, looped clip) is the same person."""
    from sayguard.rules.engine import CameraInfo, RuleEngine
    from sayguard.rules.model import Rule
    from sayguard.types import Person
    eng = RuleEngine([CameraInfo("cam01", "Gate", "entrance")])
    eng.add(Rule(name="hat", type="no_helmet", min_duration_s=0.0, cooldown_s=30))

    def bare(tid, x=100):
        return Person(track_id=tid, head=(x, 100, x + 40, 146), helmet_now=False, no_helmet_ratio=1.0, observations=5)

    fired = [f for t in range(3) for f in eng.evaluate("cam01", 1000.0 + t * 0.1, [bare(1)], [], (1280, 720)).fired]
    assert len(fired) == 1
    fired = [f for t in range(3) for f in eng.evaluate("cam01", 1001.0 + t * 0.1, [bare(7)], [], (1280, 720)).fired]
    assert fired == []                         # same spot, new id -> suppressed
    fired = eng.evaluate("cam01", 1002.0, [bare(8, x=900)], [], (1280, 720)).fired
    assert fired == []                         # one violator again within 15 s on this camera -> no new alert
    fired = eng.evaluate("cam01", 1003.0, [bare(8, x=900), bare(10, x=500)], [], (1280, 720)).fired
    assert len(fired) >= 1                     # more people violating than at the last alert -> alert
    fired = eng.evaluate("cam01", 1040.0, [bare(9)], [], (1280, 720)).fired
    assert len(fired) == 1                     # after the cooldown the same spot alerts again

# ---- a failed save changes nothing; values that cannot be saved are refused ---------------
def test_failed_save_leaves_the_running_rules_unchanged(tmp_path, monkeypatch):
    eng = make_engine(Rule(name="hat", type="no_helmet"), path=tmp_path / "rules.json")
    before, version = dict(eng.rules), eng.version
    hat = eng.list()[0]

    def full_disk(path, data):
        raise OSError("no space left")
    monkeypatch.setattr(engine_mod, "_atomic_write", full_disk)
    with pytest.raises(OSError):
        eng.add(Rule(name="crowd", type="person_count_over", threshold=3))
    with pytest.raises(OSError):
        eng.remove(hat.id)
    with pytest.raises(OSError):
        eng.set_enabled(hat.id, False)
    assert eng.rules == before and eng.version == version and hat.enabled


def test_lone_surrogate_is_refused_before_it_can_break_saving():
    with pytest.raises(ValueError):
        Rule(name="bad \ud83d", type="no_helmet")


def test_empty_camera_list_means_every_camera():
    assert Rule(name="x", type="no_helmet", cameras=[]).cameras == ["*"]


@pytest.mark.parametrize("hhmm, ok", [("24:00", True), ("23:59", True), ("24:30", False), ("25:00", False)])
def test_time_window_hours(hhmm, ok):
    if ok:
        TimeWindow(start="22:00", end=hhmm)
    else:
        with pytest.raises(ValueError):
            TimeWindow(start="22:00", end=hhmm)


def test_negative_durations_are_refused():
    with pytest.raises(ValueError):
        Rule(name="x", type="no_helmet", min_duration_s=-1)
    with pytest.raises(ValueError):
        Rule(name="x", type="no_helmet", cooldown_s=-1)


def test_incomplete_rule_in_the_file_is_loaded_switched_off(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps([{"name": "crowd", "type": "person_count_over"}]), encoding="utf-8")
    eng = RuleEngine(CAMS, path)
    eng.load()
    assert len(eng.list()) == 1 and eng.list()[0].enabled is False
