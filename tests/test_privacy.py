"""Face pixelation regions from the PPE detector's head boxes."""
from __future__ import annotations

import numpy as np

from sayguard.privacy import anonymise, face_region


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
