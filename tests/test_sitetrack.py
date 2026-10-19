"""Site geometry and the cross-camera worker identity, on synthetic floor positions."""
from __future__ import annotations

import numpy as np
import pytest

from sayguard.site import CameraCalib, Site, Zone, foot_point, touches_bottom
from sayguard.sitetrack import DETACH_TICKS, MERGE_TICKS, Obs, SiteTracker, cluster_vehicles

BOX = (100.0, 100.0, 140.0, 220.0)


def ob(cam, tid, x, y, feet=True):
    return Obs(cam=cam, track_id=tid, xy=(x, y), box=BOX, feet_visible=feet)


def run(tr, ticks, t0=0.0, dt=0.1):
    out = None
    for i, obs in enumerate(ticks):
        out = tr.update(obs, t0 + i * dt)
    return out


def test_two_cameras_one_worker():
    tr = SiteTracker()
    ws = run(tr, [[ob("A", 1, 0, 0), ob("B", 7, 0.4, 0.2)]] * 3)
    assert len(ws) == 1 and ws[0].cams == ["A", "B"]


def test_two_people_stay_apart():
    tr = SiteTracker()
    ws = run(tr, [[ob("A", 1, 0, 0), ob("B", 1, 0.3, 0), ob("A", 2, 4, 0), ob("B", 2, 4.2, 0.1)]] * 3)
    assert len(ws) == 2 and all(w.cams == ["A", "B"] for w in ws)


def test_track_id_switch_keeps_identity():
    tr = SiteTracker()
    gid = run(tr, [[ob("A", 1, 0, 0)]] * 3)[0].gid
    ws = run(tr, [[ob("A", 9, 0.1, 0)]] * 2, t0=0.3)          # camera A lost track 1, started 9 nearby
    assert [w.gid for w in ws] == [gid]


def test_walking_into_the_next_camera_keeps_identity():
    tr = SiteTracker()
    ticks = [[ob("A", 1, 0.2 * i, 0)] for i in range(10)]                 # A sees the walk ...
    ticks += [[ob("B", 4, 2.0 + 0.2 * i, 0)] for i in range(10)]          # ... then only B
    seen = set()
    for i, obs in enumerate(ticks):
        seen |= {w.gid for w in tr.update(obs, i * 0.1)}
    assert len(seen) == 1


def test_people_who_walked_together_split_up():
    tr = SiteTracker()
    run(tr, [[ob("A", 1, 0, 0), ob("B", 2, 0.5, 0)]] * 3)                 # joined: one worker
    assert len(tr.workers) == 1
    ws = run(tr, [[ob("A", 1, 0, 0), ob("B", 2, 6, 0)]] * (DETACH_TICKS + 1), t0=0.3)
    assert len(ws) == 2


def test_late_merge_keeps_older_id():
    tr = SiteTracker()
    first = run(tr, [[ob("A", 1, 0, 0), ob("B", 2, 2.6, 0)]] * 3)         # too far apart to cluster
    assert len(first) == 2
    ws = run(tr, [[ob("A", 1, 0, 0), ob("B", 2, 0.3, 0)]] * (MERGE_TICKS + 2), t0=0.3)
    assert len(ws) == 1 and ws[0].gid == min(w.gid for w in first)


def test_occluded_feet_do_not_pull_the_position():
    tr = SiteTracker()
    ws = run(tr, [[ob("A", 1, 0, 0), ob("B", 1, 0.2, 0), ob("C", 1, 0.9, 0, feet=False)]])
    assert len(ws) == 1 and abs(ws[0].xy[0] - 0.1) < 1e-6


def test_lost_worker_expires_and_ids_are_not_reused():
    tr = SiteTracker(lost_s=1.0)
    g1 = run(tr, [[ob("A", 1, 0, 0)]] * 2)[0].gid
    run(tr, [[]] * 15, t0=0.2)
    assert not tr.workers
    g2 = run(tr, [[ob("A", 2, 0, 0)]] * 2, t0=2.0)[0].gid
    assert g2 > g1


def test_a_single_camera_flicker_never_becomes_a_worker():
    tr = SiteTracker()
    assert run(tr, [[ob("A", 1, 0, 0)]]) == []                       # one tick: not confirmed
    far = Obs("A", 2, xy=(9.0, 9.0), box=BOX, sigma=2.5)               # far end of the view
    assert run(tr, [[far]] * 3, t0=0.1) == []                          # needs 4 ticks
    assert len(run(tr, [[far]], t0=0.4)) == 1
    assert len(run(tr, [[ob("A", 3, 5, 5), ob("B", 3, 5.2, 5)]])) == 1  # two cameras agree: at once


def test_uncertain_far_view_joins_instead_of_duplicating():
    tr = SiteTracker()
    run(tr, [[ob("A", 1, 0, 0)]] * 3)
    far = Obs("B", 7, xy=(2.4, 0.0), box=BOX, sigma=1.8)              # 2.4 m off, but very uncertain
    ws = run(tr, [[ob("A", 1, 0, 0), far]] * 6, t0=0.3)
    assert len(ws) == 1 and ws[0].cams == ["A", "B"] and abs(ws[0].xy[0]) < 0.3


def test_vehicle_clustering_and_distance():
    other = (300.0, 100.0, 340.0, 220.0)
    vs = cluster_vehicles([("forklift", "A", (5.0, 5.0), BOX), ("forklift", "A", (5.6, 5.0), BOX),   # one box
                           ("forklift", "B", (6.0, 5.2), BOX), ("forklift", "A", (20.0, 0.0), other)])
    assert len(vs) == 2
    near = min(vs, key=lambda v: v.xy[0])
    assert near.cams == ["A", "B"] and len(near.points) == 3
    # nearest ground contact 2.0 m away, minus half a forklift's width
    assert near.distance_to(np.array([3.0, 5.0])) == pytest.approx(2.0 - near.pad)


def test_two_boxes_of_one_camera_stay_two_vehicles():
    other = (300.0, 100.0, 340.0, 220.0)
    vs = cluster_vehicles([("forklift", "A", (5.0, 5.0), BOX), ("forklift", "A", (6.0, 5.0), other)])
    assert len(vs) == 2


def test_hidden_feet_view_joins_through_its_segment():
    tr = SiteTracker()
    good = ob("A", 1, 0.0, 0.0)
    hidden = Obs("B", 1, xy=(4.0, 0.0), box=BOX, seg=((0.1, 0.0), (4.0, 0.0)))   # bottom cut by a shelf
    ws = run(tr, [[good, hidden]] * 3)
    assert len(ws) == 1 and ws[0].cams == ["A", "B"]
    assert abs(ws[0].xy[0]) < 0.2                                     # not dragged towards 4 m


def test_people_who_only_passed_each_other_are_not_merged():
    tr = SiteTracker()
    ticks = [[ob("A", 1, 0, 0), ob("A", 2, 3, 0)]] * 6            # camera A sees both people at once
    ticks += [[ob("A", 1, 0, 0), ob("C", 5, 3, 0)]] * 3           # C takes over the second person
    ticks += [[ob("A", 1, 0, 0), ob("C", 5, 3 - 0.5 * i, 0)] for i in range(1, 6)]   # walks up to 0.5 m
    ticks += [[ob("A", 1, 0, 0), ob("C", 5, 0.3, 0)]] * (MERGE_TICKS + 3)            # stands next to them
    gids = set()
    for i, obs in enumerate(ticks):
        gids = {w.gid for w in tr.update(obs, i * 0.1)}
    assert len(gids) == 2                                         # disjoint cameras, close, but two people


def test_zone_and_plan_geometry():
    z = Zone("aisle", "Aisle", np.array([[0, 0], [4, 0], [4, 2], [0, 2]], float))
    assert z.contains(1, 1) and not z.contains(5, 1)
    site = Site(cameras={}, scale=10.0, translation=(5.0, 2.0), plan_size=(200, 100), zones=[z])
    px = site.to_plan(1.0, 3.0)
    assert px == (60.0, 50.0)                                        # y axis flipped on the plan
    assert site.from_plan(*px) == pytest.approx((1.0, 3.0))
    assert site.zone_of(1, 1) == "aisle" and site.zone_of(9, 9) is None
    z2 = site.zone_from_config({"id": "p", "plan_polygon": [[50, 80], [90, 80], [90, 60]]})
    assert z2.polygon[0] == pytest.approx(site.from_plan(50, 80))


def test_camera_homography_round_trip():
    H = np.array([[100.0, 0, 960], [0, 100.0, 540], [0, 0.001, 1]])
    cam = CameraCalib("c", H, np.linalg.inv(H), 1920, 1080)
    uv = cam.to_image(2.0, 1.5)
    assert cam.to_world(*uv) == pytest.approx((2.0, 1.5))
    half = cam.to_image(2.0, 1.5, dst_wh=(960, 540))                 # streams are downscaled
    assert cam.to_world(*half, src_wh=(960, 540)) == pytest.approx((2.0, 1.5))


def test_foot_point_and_cut_off_boxes():
    assert foot_point((10, 20, 30, 120)) == (20, 120)
    assert touches_bottom((0, 0, 10, 719), 720) and not touches_bottom((0, 0, 10, 600), 720)


def test_worker_hidden_behind_a_rack_keeps_their_id():
    tr = SiteTracker()
    gid = run(tr, [[ob("A", 1, 0.0, 0.0)]] * 3)[0].gid
    run(tr, [[]] * 40, t0=0.3)                                         # 4 s unseen
    ws = run(tr, [[ob("B", 4, 3.0, 0.0)]] * 2, t0=4.3)                 # reappears 3 m on
    assert [w.gid for w in ws] == [gid]
