"""EventStore aggregates and retention on a temporary database with real snapshot files
(no models): the dashboard's 1 Hz counts must match what a row scan would give, and a
retention pass must delete old rows, their JPEGs and the emptied day folders only."""
from __future__ import annotations

import time

import numpy as np
import pytest

from sayguard.events import EventStore
from sayguard.types import Event

DAY = 86400.0


def at(day: int, h: int, m: int = 0) -> float:
    """Local wall-clock time on 2026-10-<day>, so day folders do not depend on the timezone."""
    return time.mktime((2026, 10, day, h, m, 0, 0, 0, -1))


def ev(i: int, ts: float, rule_type: str = "no_helmet", verified: bool | None = None) -> Event:
    return Event(id=f"e{i:03d}", ts=ts, cam_id="cam01", cam_name="Gate A", zone="entrance", rule_id="r1",
                 rule_name="rule", rule_type=rule_type, severity="high", message="msg", verified=verified,
                 box=(1, 2, 3, 4), extra={"detail": "x"})


@pytest.fixture()
def store(tmp_path) -> EventStore:
    return EventStore(tmp_path / "db" / "events.db", tmp_path / "snaps")


def add_with_snapshot(store: EventStore, e: Event) -> None:
    e.snapshot = store.save_snapshot(e, np.zeros((8, 8, 3), np.uint8))
    store.add(e)


def test_counts_empty_window_is_all_zero(store):
    assert store.counts(0, time.time()) == {"total": 0, "confirmed": 0, "false_alarms": 0,
                                            "pending": 0, "not_applicable": 0}


def test_counts_split_by_vlm_state(store):
    t = at(18, 12)
    rows = [("no_helmet", True), ("no_helmet", True), ("no_helmet", False), ("no_helmet", None),
            ("object_near_person", None), ("person_present", None), ("person_count_over", None)]
    for i, (rt, v) in enumerate(rows):
        store.add(ev(i, t + i, rt, v))
    store.add(ev(50, t - 2 * DAY))                                 # outside the window
    c = store.counts(t - DAY, t + 60)
    assert c == {"total": 7, "confirmed": 2, "false_alarms": 1, "pending": 1, "not_applicable": 3}
    assert all(type(v) is int for v in c.values())
    assert c["total"] == c["confirmed"] + c["false_alarms"] + c["pending"] + c["not_applicable"]
    store.set_verified("e003", False, "VLM: helmet visible")
    assert store.counts(t - DAY, t + 60)["pending"] == 0
    assert store.counts(t, t + 1)["total"] == 1                     # t1 is exclusive


def test_between_pages_and_columns(store):
    t = at(18, 8)
    for i in range(23):
        store.add(ev(i, t + (i // 2)))                              # pairs share a timestamp
    full = store.between(t, t + 100, page=5)
    assert [e["id"] for e in full] == [f"e{i:03d}" for i in range(23)]
    assert full[0]["box"] == [1, 2, 3, 4] and full[0]["extra"] == {"detail": "x"} and full[0]["verified"] is None
    slim = store.between(t, t + 100, ("id", "verified", "rule_type"), page=4)
    assert len(slim) == 23 and set(slim[0]) == {"id", "verified", "rule_type"}
    with pytest.raises(ValueError):
        store.between(t, t + 1, ("id; DROP TABLE events",))


def test_save_snapshot_recreates_a_deleted_snapshots_folder(store, tmp_path):
    store.snap_dir.rmdir()
    e = ev(1, at(18, 9))
    add_with_snapshot(store, e)
    assert (tmp_path / "snaps" / "20261018" / "e001.jpg").is_file()


def test_purge_deletes_old_rows_files_and_empty_day_folders(store, tmp_path):
    old = [ev(1, at(10, 9)), ev(2, at(10, 17)), ev(3, at(12, 8))]
    keep = [ev(4, at(17, 23)), ev(5, at(18, 7))]
    for e in old + keep:
        add_with_snapshot(store, e)
    unrelated = tmp_path / "snaps" / "20261011"                     # an old folder with a stray file
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("keep me")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")
    rogue = ev(6, at(11, 9))
    rogue.snapshot = str(outside)                                   # a path outside the snapshots folder
    store.add(rogue)
    v0 = store.version

    assert store.purge_older_than(at(16, 0)) == 4
    assert store.version == v0 + 1
    assert sorted(e["id"] for e in store.between(0, at(19, 0))) == ["e004", "e005"]
    for e in old:
        assert not (tmp_path / "snaps" / time.strftime("%Y%m%d", time.localtime(e.ts)) / f"{e.id}.jpg").exists()
    days = sorted(p.name for p in (tmp_path / "snaps").iterdir())
    assert days == ["20261011", "20261017", "20261018"]             # emptied 10th/12th removed, stray kept
    assert outside.exists()
    for e in keep:
        assert (tmp_path / "snaps" / time.strftime("%Y%m%d", time.localtime(e.ts)) / f"{e.id}.jpg").is_file()

    assert store.purge_older_than(at(16, 0)) == 0                   # idempotent
    assert store.version == v0 + 1


def test_purge_keeps_the_cutoff_day_folder(store, tmp_path):
    e = ev(1, at(18, 6))
    add_with_snapshot(store, e)
    assert store.purge_older_than(at(18, 12)) == 1
    assert (tmp_path / "snaps" / "20261018").is_dir()               # may still receive today's snapshots
