"""Face pixelation regions from the PPE detector's head boxes."""
from __future__ import annotations

import numpy as np

from sayguard.privacy import FaceHold, anonymise, face_region


def test_hard_hat_box_covers_the_face_below_the_hat():
    # a hard-hat box frames the hat only (about 0.73 as tall as wide): the face is below it
    x1, y1, x2, y2 = 100, 100, 140, 129
    fx1, fy1, fx2, fy2 = face_region((x1, y1, x2, y2), True, 1280, 720)
    w = x2 - x1
    assert fy1 <= y2 - 0.2 * (y2 - y1)            # from the brim
    assert fy2 >= y2 + 0.9 * w                    # down to the chin
    assert fx1 <= x1 - 0.1 * w and fx2 >= x2 + 0.1 * w   # wider than the hat for a turned head


def test_bare_head_box_covers_its_lower_part():
    fx1, fy1, fx2, fy2 = face_region((100, 100, 140, 148), False, 1280, 720)
    assert fy1 <= 106 and fy2 >= 148 and fx1 <= 100 and fx2 >= 140


def test_regions_stay_inside_the_frame():
    assert face_region((0, 690, 40, 719), True, 1280, 720) == (0, 701, 47, 720)


def test_anonymise_changes_the_pixels_under_the_face():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    before = img.copy()
    anonymise(img, [((80, 40, 120, 69), True)])
    assert not np.array_equal(img[75:105, 85:115], before[75:105, 85:115])   # the face under the hat
    assert np.array_equal(img[150:, :], before[150:, :])                      # far below: untouched


def test_face_stays_masked_through_a_short_detector_miss():
    hold = FaceHold(hold_s=1.0, grow=0.15)
    head = ((100.0, 100.0, 140.0, 129.0), True)
    assert hold.update(0.0, [head]) == [head]
    held = hold.update(0.3, [])                       # missed for 0.3 s: still masked, a little larger
    assert len(held) == 1 and held[0][1] is True
    (x1, y1, x2, y2), _ = held[0]
    assert x1 < 100 and y1 < 100 and x2 > 140 and y2 > 129
    assert hold.update(1.2, []) == []                 # gone for more than a second: no longer held


def test_a_new_detection_replaces_the_held_box():
    hold = FaceHold()
    hold.update(0.0, [((100, 100, 140, 129), False)])
    moved = ((104, 101, 144, 130), False)             # the same head, found again nearby
    assert hold.update(0.2, [moved]) == [moved]


def test_a_restarted_clock_clears_the_hold():
    hold = FaceHold()
    hold.update(50.0, [((100, 100, 140, 129), False)])
    assert hold.update(1.0, []) == []

