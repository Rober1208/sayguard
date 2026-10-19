"""Site-mode rule evaluation: one subject per worker, floor zones, metres."""
from __future__ import annotations

from sayguard.rules import CameraInfo, Rule, RuleEngine
from sayguard.rules.engine import SITE, SiteObject, SiteSubject

CAMS = [CameraInfo("c1", "A", "warehouse", zones=["aisle"]), CameraInfo("c2", "B", "warehouse"),
        CameraInfo("c3", "C", "warehouse", zones=["dock"])]


def engine(*rules):
    e = RuleEngine(CAMS)
    for r in rules:
        e.add(r)
    return e


def worker(gid, x, y, zones=("warehouse",), cams=("c1", "c2", "c3")):
    return SiteSubject(gid, (x, y), list(zones), list(cams))


def forklift(x, y):
    return SiteObject("forklift", (x, y), ["warehouse"], ["c1", "c2"])


def run(e, ticks, dt=0.1):
    fired = []
    for i, (subs, objs) in enumerate(ticks):
        fired += e.evaluate_site(i * dt, subs, objs, frame_dt=dt).fired
    return fired


NEAR = Rule(name="near", type="object_near_person", object="forklift", threshold=3.0, min_duration_s=1.0,
            cooldown_s=30.0, zones=["warehouse"])


def test_worker_seen_by_three_cameras_alerts_once():
    fired = run(engine(NEAR), [([worker(7, 0, 0)], [forklift(2, 0)])] * 30)
    assert [(f.key, f.track_id, f.cam_id) for f in fired] == [("g7", 7, SITE)]
    assert "2.0 m" in fired[0].detail


def test_two_workers_both_alert_without_the_per_camera_gap():
    fired = run(engine(NEAR), [([worker(1, 0, 0), worker(2, 4, 0)], [forklift(2, 0)])] * 30)
    assert sorted(f.track_id for f in fired) == [1, 2]


def test_distance_uses_nearest_ground_contact():
    near = SiteObject("forklift", (10, 0), ["warehouse"], ["c1"], distance_to=lambda xy: 1.5)
    fired = run(engine(NEAR), [([worker(3, 0, 0)], [near])] * 20)
    assert fired and "1.5 m" in fired[0].detail


def test_far_worker_and_short_visits_do_not_alert():
    assert not run(engine(NEAR), [([worker(1, 0, 0)], [forklift(5, 0)])] * 30)
    assert not run(engine(NEAR), [([worker(1, 0, 0)], [forklift(2, 0)])] * 5)     # 0.4 s < 1 s


def test_floor_zone_scope_and_exclusion():
    aisle = Rule(name="aisle", type="person_present", zones=["aisle"], min_duration_s=0.5)
    e = engine(aisle)
    fired = run(e, [([worker(1, 0, 0, zones=("aisle", "warehouse")), worker(2, 9, 9)], [])] * 10)
    assert [f.track_id for f in fired] == [1]
    no_dock = Rule(name="nodock", type="person_present", zones=["warehouse"], zones_exclude=["dock"],
                   min_duration_s=0.5)
    fired = run(engine(no_dock), [([worker(1, 0, 0, zones=("dock", "warehouse")), worker(2, 9, 9)], [])] * 10)
    assert [f.track_id for f in fired] == [2]


def test_count_over_is_one_site_alert():
    rule = Rule(name="crowd", type="person_count_over", threshold=2, zones=["warehouse"], min_duration_s=0.5)
    fired = run(engine(rule), [([worker(i, i, 0) for i in range(4)], [])] * 10)
    assert [f.key for f in fired] == [SITE] and "4 people > 2" in fired[0].detail


def test_camera_filter_applies_to_the_cameras_seeing_the_worker():
    rule = Rule(name="cam3only", type="person_present", cameras=["c3"], min_duration_s=0.5)
    fired = run(engine(rule), [([worker(1, 0, 0, cams=("c1",)), worker(2, 5, 5, cams=("c2", "c3"))], [])] * 10)
    assert [f.track_id for f in fired] == [2]


def test_scope_uses_every_zone_a_camera_sees():
    e = engine()
    assert e.scope_cameras(Rule(name="d", type="person_present", zones=["dock"])) == ["c3"]
    assert e.scope_cameras(Rule(name="w", type="person_present", zones=["warehouse"])) == ["c1", "c2", "c3"]
    assert e.scope_cameras(Rule(name="x", type="person_present", zones=["nowhere"])) == []


def test_objects_for_site_ignores_camera_zones():
    e = engine(NEAR, Rule(name="ladder", type="object_present", object="ladder", zones=["dock"]))
    assert e.objects_for_site() == {"forklift", "ladder"}


def test_rule_objects_match_detector_labels():
    from sayguard.rules.model import canon_label
    assert canon_label("Pallet trucks") == canon_label("pallet-truck") == "pallet_truck"
    assert canon_label("AMR") == "mobile_robot" and canon_label("forklifts") == "forklift"
    rule = Rule(name="pt", type="object_near_person", object="pallet trucks", threshold=2.0, min_duration_s=0.5)
    obj = SiteObject("pallet_truck", (1, 0), ["warehouse"], ["c1"])
    assert run(engine(rule), [([worker(1, 0, 0)], [obj])] * 10)


def test_new_global_id_at_the_same_spot_does_not_alert_again():
    ticks = [([worker(1, 0, 0)], [forklift(2, 0)])] * 20            # alerted as W-1 ...
    ticks += [([worker(9, 0.3, 0)], [forklift(2, 0)])] * 20          # ... re-identified as W-9, same place
    ticks += [([worker(9, 0.3, 0), worker(4, 0, 6)], [forklift(2, 0), forklift(1, 6)])] * 20   # someone else
    fired = run(engine(NEAR), ticks)
    assert [f.track_id for f in fired] == [1, 4]


def test_per_camera_evaluation_keeps_the_old_duplicate_guards():
    e = engine(NEAR)
    fired = []
    box = (100, 100, 140, 220)
    for i in range(60):                                             # one person, track id split in three
        tid = 1 if i < 20 else 2 if i < 40 else 3
        s = SiteSubject(tid, (0, 0), ["warehouse"], ["c1"], box=box)
        fired += e.evaluate_site(i * 0.1, [s], [forklift(2, 0)], frame_dt=0.1, cam_id="c1").fired
    assert len(fired) == 1 and fired[0].key == "t1" and fired[0].cam_id == "c1"


def test_site_wide_alert_records_the_cameras_involved():
    rule = Rule(name="crowd", type="person_count_over", threshold=1, zones=["warehouse"], min_duration_s=0.2)
    e = engine(rule)
    res = None
    for i in range(5):
        res = e.evaluate_site(i * 0.1, [worker(1, 0, 0, cams=("c1",)), worker(2, 5, 0, cams=("c3",))], [], frame_dt=0.1)
    assert res.site_scope[rule.id] == ["c1", "c3"]


def test_duration_carries_over_when_the_global_id_changes():
    rule = Rule(name="aisle", type="person_present", zones=["aisle"], min_duration_s=3.0)
    inside = ("aisle", "warehouse")
    ticks = [([worker(1, 0, 0, zones=inside)], [])] * 20               # 2 s as W-1 ...
    ticks += [([worker(2, 0.2, 0, zones=inside)], [])] * 15            # ... then 1.5 s as W-2
    fired = run(engine(rule), ticks)
    assert [f.track_id for f in fired] == [2]


def test_re_identified_worker_keeps_the_original_cooldown():
    rule = Rule(name="aisle", type="person_present", zones=["aisle"], min_duration_s=1.0, cooldown_s=10.0)
    inside, away = ("aisle", "warehouse"), ("warehouse",)
    ticks = [([worker(1, 0, 0, zones=inside)], [])] * 20               # alerted as W-1 at 1 s
    ticks += [([worker(1, 9, 9, zones=away)], [])] * 20                # leaves
    ticks += [([worker(2, 0.2, 0, zones=inside)], [])] * 20            # back as W-2 at 4-6 s: same person, quiet
    ticks += [([worker(2, 9, 9, zones=away)], [])] * 60                # leaves again
    ticks += [([worker(2, 0.2, 0, zones=inside)], [])] * 20            # back at 12 s: the 10 s since W-1's alert are over
    fired = run(engine(rule), ticks)
    assert [f.track_id for f in fired] == [1, 2]


def test_alerted_worker_that_walked_on_and_changed_id_is_not_alerted_again():
    rule = Rule(name="floor", type="person_present", zones=["aisle"], min_duration_s=1.0, cooldown_s=60.0)
    inside = ("aisle", "warehouse")
    walk = [([worker(1, 0.25 * i, 0, zones=inside)], []) for i in range(30)]       # alerted at 1 s, walks 7 m
    handover = [([worker(7, 7.4, 0, zones=inside)], [])] * 20                     # then shows up as W-7
    fired = run(engine(rule), walk + handover)
    assert [f.track_id for f in fired] == [1]
    gone = [([], [])] * 50                                                         # 5 s nobody: not a handover
    fired = run(engine(rule), walk + gone + handover)
    assert [f.track_id for f in fired] == [1, 7]


def test_next_worker_down_the_same_aisle_is_alerted_too():
    rule = Rule(name="aisle", type="person_present", zones=["aisle"], min_duration_s=1.0, cooldown_s=60.0)
    inside, away = ("aisle", "warehouse"), ("warehouse",)
    first = [([worker(1, 0, 0, zones=inside)], [])] * 20                         # W-1 alerted at 1 s
    later = [([worker(1, 20, 0, zones=away), worker(2, 0.3, 0, zones=inside)], [])] * 20   # W-1 still tracked
    fired = run(engine(rule), first + [([worker(1, 20, 0, zones=away)], [])] * 50 + later)
    assert [f.track_id for f in fired] == [1, 2]


def test_second_id_on_the_alerted_worker_is_not_alerted():
    rule = Rule(name="aisle", type="person_present", zones=["aisle"], min_duration_s=1.0, cooldown_s=60.0)
    inside = ("aisle", "warehouse")
    ticks = [([worker(1, 0, 0, zones=inside)], [])] * 20
    ticks += [([worker(1, 0, 0, zones=inside), worker(5, 0.4, 0.2, zones=inside)], [])] * 30
    fired = run(engine(rule), ticks)
    assert [f.track_id for f in fired] == [1]


def test_coming_back_into_a_zone_is_a_new_entry_despite_the_cooldown():
    from sayguard.rules.engine import SITE_REENTRY_S
    rule = Rule(name="aisle", type="person_present", zones=["aisle"], min_duration_s=1.0, cooldown_s=60.0)
    inside, away = ("aisle", "warehouse"), ("warehouse",)
    stay = [([worker(1, 0, 0, zones=inside)], [])] * 20
    out_long = [([worker(1, 5, 5, zones=away)], [])] * int(10 * (SITE_REENTRY_S + 1))
    out_short = [([worker(1, 5, 5, zones=away)], [])] * 20                       # 2 s: same visit
    assert [f.track_id for f in run(engine(rule), stay + out_long + stay)] == [1, 1]
    assert [f.track_id for f in run(engine(rule), stay + out_short + stay)] == [1]
