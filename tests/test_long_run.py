"""A site-mode service runs for days: what it keeps per tick must not grow with how long it has
run, and Windows must not move it onto the efficient cores as background work
(sayguard.resources.keep_full_speed: site mode slowed from about 93 to about 50 frames/s)."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from sayguard import resources
from sayguard.pipeline import Pipeline
from sayguard.rules import CameraInfo, Rule, RuleEngine
from sayguard.rules.engine import SiteObject, SiteSubject
from sayguard.sitetrack import Obs, SiteTracker

BOX = (100.0, 100.0, 140.0, 220.0)


def test_keep_full_speed_turns_execution_speed_throttling_off():
    ok = resources.keep_full_speed()
    if sys.platform != "win32":
        assert ok is False and resources.power_throttling() is None
        return
    assert ok
    assert resources.power_throttling() == (resources.POWER_THROTTLING_EXECUTION_SPEED, 0)


def test_live_pipeline_asks_for_full_speed(monkeypatch):
    calls, started = [], []
    monkeypatch.setattr(resources, "keep_full_speed", lambda: calls.append(1) or True)
    fake = SimpleNamespace(states={}, verifier=SimpleNamespace(start=lambda: started.append("verifier")),
                           _thread=SimpleNamespace(start=lambda: started.append("loop")))
    Pipeline.start(fake)
    assert calls == [1] and started == ["verifier", "loop"]


def test_site_state_follows_who_is_on_the_floor_not_how_long_it_ran():
    """Two minutes at 10 Hz: people walk in and out, cameras switch track ids, alerts fire.
    The tracker and the rule engine may hold what the people on the floor now need, never one
    entry per person or id ever seen."""
    rng = np.random.default_rng(7)
    cams = ("A", "B", "C")
    tr = SiteTracker()
    eng = RuleEngine([CameraInfo(c, c, "warehouse") for c in cams])
    eng.add(Rule(name="near", type="object_near_person", object="forklift", threshold=3.0, min_duration_s=1.0,
                 cooldown_s=20.0))
    eng.add(Rule(name="aisle", type="person_present", zones=["aisle"], min_duration_s=1.0, cooldown_s=20.0))
    forklift = SiteObject("forklift", (10.0, 10.0), ["warehouse"], ["A"])
    people: dict[int, dict] = {}
    next_pid = next_tid = 0
    peak: dict[str, int] = {}
    ticks = 1200
    for i in range(ticks):
        ts = i * 0.1
        if len(people) < 4 and rng.random() < 0.1:                      # someone walks in ...
            people[next_pid] = {"xy": rng.uniform(0, 20, 2), "v": rng.normal(0, 0.1, 2),
                                "leave": i + int(rng.integers(30, 120)),
                                "tids": {c: (next_tid := next_tid + 1) for c in rng.choice(cams, 2, replace=False)}}
            next_pid += 1
        for pid in [p for p, d in people.items() if d["leave"] <= i]:   # ... and leaves
            del people[pid]
        obs = []
        for d in people.values():
            d["xy"] = np.clip(d["xy"] + d["v"], 0, 20)
            for c in d["tids"]:
                if rng.random() < 0.01:                                  # the camera switches track id
                    d["tids"][c] = (next_tid := next_tid + 1)
                x, y = d["xy"] + rng.normal(0, 0.05, 2)
                obs.append(Obs(c, d["tids"][c], (float(x), float(y)), BOX))
        workers = tr.update(obs, ts)
        subs = [SiteSubject(w.gid, (float(w.xy[0]), float(w.xy[1])),
                            ["warehouse"] + (["aisle"] if w.xy[0] < 6 else []), w.cams) for w in workers]
        eng.evaluate_site(ts, subs, [forklift], frame_dt=0.1)
        if i >= ticks // 2:
            sizes = {"workers": len(tr.workers), "assign": len(tr.assign), "assign_seen": len(tr.assign_seen),
                     "linked_at": len(tr.linked_at), "close": len(tr._close), "pending": len(tr._pending),
                     "coseen": len(tr._coseen) + len(tr._coseen_n), "hits": len(eng._hits),
                     "last_fire": len(eng._last_fire), "site_seen": len(eng._site_seen),
                     "site_fired": sum(len(v) for v in eng._site_fired.values())}
            peak = {k: max(v, peak.get(k, 0)) for k, v in sizes.items()}
    assert next_pid > 40 and next_tid > 120 and tr.next_gid > 35      # plenty of people and ids came and went
    # at most 4 people on the floor, each seen by 2 cameras: a few entries each, kept for seconds
    bounds = {"workers": 12, "assign": 24, "assign_seen": 24, "linked_at": 24, "close": 66, "pending": 8,
              "coseen": 40, "hits": 16, "last_fire": 16, "site_seen": 16, "site_fired": 16}
    assert {k: v for k, v in peak.items() if v > bounds[k]} == {}, peak
