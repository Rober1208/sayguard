"""SyncedFileGroup: every camera gets the same frame index, and all rewind together."""
from __future__ import annotations

import time

import cv2
import numpy as np

from sayguard.sources import SyncedFileGroup


def write_clip(path, n):
    """n frames whose brightness encodes the frame number (10 * k)."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for k in range(n):
        w.write(np.full((48, 64, 3), 10 * k, np.uint8))
    w.release()


def frame_no(pkt):
    return int(round(float(pkt.frame.mean()) / 10))


def test_cameras_stay_frame_locked_and_loop_together(tmp_path):
    write_clip(tmp_path / "a.avi", 10)
    write_clip(tmp_path / "b.avi", 8)                 # the shorter clip decides the loop length
    g = SyncedFileGroup({"a": str(tmp_path / "a.avi"), "b": str(tmp_path / "b.avi")}, fps=200).start()
    seen = []
    try:
        t0 = time.time()
        last = None
        while time.time() - t0 < 5 and len(seen) < 20:
            a, b = g.views["a"].latest(), g.views["b"].latest()
            if a is not None and b is not None and a.index != last:
                last = a.index
                seen.append((a.index, b.index, a.pos, b.pos, a.loop, b.loop, frame_no(a), frame_no(b)))
            time.sleep(0.001)
    finally:
        g.stop()
    assert len(seen) >= 12
    for ia, ib, pa, pb, la, lb, fa, fb in seen:
        assert ia == ib and pa == pb and la == lb      # one tick never mixes moments
        assert fa == fb == pa                          # the frame shown is frame `pos` in both files
        assert pa < 8                                  # rewound at the end of the 8-frame clip
    assert any(s[4] >= 1 for s in seen)                # and it did loop
    assert g.views["b"].native_fps == 10.0             # views expose the VideoSource interface
