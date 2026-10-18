"""Alert store: SQLite (stdlib) + anonymised JPEG snapshots on disk.

One connection is shared by the camera thread (add), the verifier (set_verified)
and the API (reads), serialised by one lock.  Anything the dashboard polls or a
report scans must therefore keep that lock short: counts() aggregates in SQL
instead of decoding rows, and between() reads in keyset pages, taking the lock
per page.  Nothing is kept forever: purge_older_than() drops old rows together
with their snapshot files (a day of demo alerts is 0.6-0.9 GB of JPEGs).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from .types import Event

log = logging.getLogger("sayguard.events")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, ts REAL, cam_id TEXT, cam_name TEXT, zone TEXT,
  rule_id TEXT, rule_name TEXT, rule_type TEXT, severity TEXT, message TEXT,
  track_id INTEGER, box TEXT, snapshot TEXT, verified INTEGER, verify_note TEXT, extra TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""
COLUMNS = ("id", "ts", "cam_id", "cam_name", "zone", "rule_id", "rule_name", "rule_type", "severity", "message",
           "track_id", "box", "snapshot", "verified", "verify_note", "extra")
PAGE_ROWS = 5000
# Only hard-hat alerts are sent to the VLM; every other rule type keeps verified=NULL for good.
VLM_CHECKED_TYPE = "no_helmet"


class EventStore:
    def __init__(self, db_path: Path, snapshots_dir: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.snap_dir = snapshots_dir
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self.version = 0

    def save_snapshot(self, ev: Event, img: np.ndarray) -> str:
        day = time.strftime("%Y%m%d", time.localtime(ev.ts))
        d = self.snap_dir / day
        d.mkdir(parents=True, exist_ok=True)   # parents: the snapshots folder itself may have been cleared
        p = d / f"{ev.id}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return str(p)

    def add(self, ev: Event) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ev.id, ev.ts, ev.cam_id, ev.cam_name, ev.zone, ev.rule_id, ev.rule_name, ev.rule_type,
                 ev.severity, ev.message, ev.track_id, json.dumps(ev.box), ev.snapshot,
                 None if ev.verified is None else int(ev.verified), ev.verify_note, json.dumps(ev.extra)))
            self._db.commit()
            self.version += 1

    def set_verified(self, event_id: str, verified: bool | None, note: str) -> None:
        with self._lock:
            self._db.execute("UPDATE events SET verified=?, verify_note=? WHERE id=?",
                             (None if verified is None else int(verified), note, event_id))
            self._db.commit()
            self.version += 1

    def _row(self, r: sqlite3.Row) -> dict:
        d = dict(r)
        if "box" in d:
            d["box"] = json.loads(d["box"]) if d["box"] else None
        if "extra" in d:
            d["extra"] = json.loads(d["extra"]) if d["extra"] else {}
        if "verified" in d:
            d["verified"] = None if d["verified"] is None else bool(d["verified"])
        return d

    def recent(self, limit: int = 50, since: float | None = None) -> list[dict]:
        with self._lock:
            if since is None:
                rows = self._db.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
            else:
                rows = self._db.execute("SELECT * FROM events WHERE ts>=? ORDER BY ts DESC LIMIT ?",
                                        (since, limit)).fetchall()
        return [self._row(r) for r in rows]

    def get(self, event_id: str) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return self._row(r) if r else None

    def between(self, t0: float, t1: float, columns: Iterable[str] | None = None,
                page: int = PAGE_ROWS) -> list[dict]:
        """Rows with t0 <= ts < t1, oldest first.  `columns` limits what is read and
        decoded (a report needs 9 of 16); pages continue from the last (ts, id) so the
        lock is released between pages and add() from the camera thread never waits long."""
        cols = tuple(columns) if columns is not None else COLUMNS
        unknown = set(cols) - set(COLUMNS)
        if unknown:
            raise ValueError(f"unknown event columns: {sorted(unknown)}")
        select = ", ".join(dict.fromkeys(("ts", "id", *cols)))   # ts/id are the paging key
        out: list[dict] = []
        last: tuple[float, str] | None = None
        while True:
            with self._lock:
                if last is None:
                    rows = self._db.execute(f"SELECT {select} FROM events WHERE ts>=? AND ts<? "
                                            "ORDER BY ts, id LIMIT ?", (t0, t1, page)).fetchall()
                else:
                    rows = self._db.execute(f"SELECT {select} FROM events WHERE ts>=? AND (ts, id) > (?, ?) "
                                            "AND ts<? ORDER BY ts, id LIMIT ?",
                                            (last[0], *last, t1, page)).fetchall()
            for r in rows:
                d = self._row(r)
                out.append(d if columns is None else {k: d[k] for k in cols})
            if len(rows) < page:
                return out
            last = (rows[-1]["ts"], rows[-1]["id"])

    def counts(self, t0: float, t1: float) -> dict[str, int]:
        """Alert totals for t0 <= ts < t1 in one aggregate query (the dashboard polls this
        every second).  pending = hard-hat alerts the VLM has not judged yet;
        not_applicable = alerts of rule types the VLM never checks.  COALESCE because
        SUM over zero rows is NULL."""
        with self._lock:
            r = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(verified = 1), 0), COALESCE(SUM(verified = 0), 0), "
                "COALESCE(SUM(verified IS NULL AND rule_type = ?), 0), "
                "COALESCE(SUM(verified IS NULL AND IFNULL(rule_type, '') != ?), 0) "
                "FROM events WHERE ts>=? AND ts<?", (VLM_CHECKED_TYPE, VLM_CHECKED_TYPE, t0, t1)).fetchone()
        return dict(zip(("total", "confirmed", "false_alarms", "pending", "not_applicable"), map(int, r)))

    def purge_older_than(self, ts: float) -> int:
        """Delete events with ts < `ts` and their snapshots; returns the number of rows.
        Files are removed after the lock is released, and only inside snap_dir (the path
        comes from the database).  Day folders are only removed when empty and older
        than the cutoff's day, so today's folder is never pulled from under save_snapshot."""
        with self._lock:
            snaps = [r[0] for r in self._db.execute(
                "SELECT snapshot FROM events WHERE ts<? AND snapshot IS NOT NULL", (ts,)).fetchall()]
            n = self._db.execute("DELETE FROM events WHERE ts<?", (ts,)).rowcount
            self._db.commit()
            if n:
                self.version += 1
        root = self.snap_dir.resolve()
        for s in snaps:
            p = Path(s)
            try:
                if p.resolve().is_relative_to(root):
                    p.unlink(missing_ok=True)
            except OSError as e:
                log.warning("could not delete snapshot %s: %s", p, e)
        cutoff_day = time.strftime("%Y%m%d", time.localtime(ts))
        if not self.snap_dir.is_dir():   # cleared by hand: save_snapshot recreates it on the next alert
            return max(n, 0)
        for d in self.snap_dir.iterdir():
            if d.is_dir() and len(d.name) == 8 and d.name.isdigit() and d.name < cutoff_day:
                try:
                    d.rmdir()   # fails (and is skipped) while anything is still inside
                except OSError:
                    pass
        return max(n, 0)
