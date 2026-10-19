# Site-mode evaluation split (written 2026-10-19 00:50, before any result on the new scenes)

Spec: one UGen300 per area of at most 12 cameras; watched zones inside 2+ camera coverage (person >= 40 px).

- Detector training (existing): W000, W002, W004, W006, W008, W011, W014, W016 (W016 cams 9,10 = detector val)
- If we retrain for small warehouses, add ONLY: W009, W017
- Parameter validation (confidence filters, forklift margin): W012 (small), W020 (large, out of spec)
- Test, never trained or tuned on: W010, W013, W015, W018, W019, W022
- Frozen run first: current v3 detector + current parameters on every scene, reported as-is.

## Update 2026-10-19 01:42 (after the frozen run)
- Scenes come in pairs with the same building and cameras: 000/001, 002/003, 004/005, 006/007, 008/009, 017/018, 020/021.
  W020 is the W000-W007 building (other cameras): validation only, not an unseen building.
- So the retrain pool is void: W009 = W008's twin (already trained), W017 = W018's twin (a test scene). No retraining.
- Rule logic changed after the frozen run, developed and checked only on W009/W012/W017/W020: presence-aware
  same-worker guard; site presence rules re-alert on a new entry after SITE_REENTRY_S = 10 s (5 vs 10 compared on
  those scenes: same coverage, 10 s gives fewer alerts). Test scenes are re-run once with it; frozen numbers kept.
