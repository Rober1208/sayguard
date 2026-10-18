"""Measure spoken-rule parsing accuracy on the labelled sentences in tests/test_parser.py
(Mandarin and English), three ways:

  llm_only  - the on-device LLM's JSON as is (after mapping names to ids)
  rules     - the deterministic extractor alone (what runs when the LLM is down)
  llm+check - what SayGuard ships: LLM proposal checked against explicit evidence

A sentence counts as correct only if every labelled field matches.  Needs the
Ollama-compatible server from configs/site.yaml.

    python scripts/eval_rules.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from sayguard.config import load_config, work_dir  # noqa: E402
from sayguard.llm import LLMClient  # noqa: E402
from sayguard.rules import parser as P  # noqa: E402
from sayguard.rules.model import Rule  # noqa: E402


FACTS = ("type", "zones", "cameras", "object", "threshold", "time_window")
ALL = ["*"]


def _e(type_: str, zones=(), cameras=ALL, obj=None, th=None, tw=None, dur=None) -> dict:
    d = {"type": type_, "zones": list(zones), "cameras": list(cameras), "object": obj, "threshold": th,
         "time_window": {"start": tw[0], "end": tw[1]} if tw else None}
    if dur is not None:
        d["min_duration_s"] = dur
    return d


# Written after the extractor was finished and never used to tune it: an honest held-out set.
# "after X" means X until 06:00 by SayGuard's convention.
HELDOUT = [
    ("倉庫同時超過四個人就通知我", _e("person_count_over", ["warehouse"], th=4)),
    ("工作坊裡沒戴安全帽的人要馬上提醒", _e("no_helmet", ["workshop"])),
    ("下班後辦公室不能有人", _e("person_present", ["office"], tw=("18:00", "07:00"))),
    ("如果有人離挖土機兩公尺以內就警告", _e("object_near_person", obj="excavator", th=2)),
    ("B區看到梯子要通知", _e("object_present", ["zone_b"], obj="ladder")),
    ("入口晚上十一點到早上五點不准有人進入", _e("person_present", ["entrance"], tw=("23:00", "05:00"))),
    ("5號攝影機有人沒戴安全帽超過十秒就警告", _e("no_helmet", cameras=["cam05"], dur=10)),
    ("Notify me when anyone enters the office after 7 pm", _e("person_present", ["office"], tw=("19:00", "06:00"))),
    ("Warn me if there is a forklift in Zone B", _e("object_present", ["zone_b"], obj="forklift")),
    ("Alert if more than six workers are on the scaffold", _e("person_count_over", cameras=["cam04"], th=6)),
    ("If someone stands within one metre of the excavator, alert immediately",
     _e("object_near_person", obj="excavator", th=1)),
    ("Tell me when people at the loading dock are not wearing hard hats", _e("no_helmet", cameras=["cam08"])),
    ("在焊接區看到有人沒戴安全帽就通知", _e("no_helmet", cameras=["cam06"])),
    ("Zone A: alert me if a truck is within 5 metres of a worker", _e("object_near_person", ["zone_a"], obj="truck", th=5)),
    ("大門口超過兩個人逗留就提醒", _e("person_count_over", ["entrance"], th=2)),
    ("Alert me if anyone is in the warehouse between 10 pm and 6 am",
     _e("person_present", ["warehouse"], tw=("22:00", "06:00"))),
    ("A區有人超過三十秒沒戴安全帽就警告", _e("no_helmet", ["zone_a"], dur=30)),
    ("Let me know if smoke appears in the workshop", _e("object_present", ["workshop"], obj="smoke")),
    ("如果B區人數多於5人請提醒", _e("person_count_over", ["zone_b"], th=5)),
    ("Raise an alarm when a ladder shows up at the site gate", _e("object_present", cameras=["cam01"], obj="ladder")),
]


def factual(expected: dict) -> dict:
    """Only what the sentence states: severity and the default durations are policy, not facts
    (a duration counts only when the sentence gives one, i.e. not the 1 s / 2 s defaults)."""
    out = {k: v for k, v in expected.items() if k in FACTS}
    if expected.get("min_duration_s") not in (None, 1.0, 2.0):
        out["min_duration_s"] = expected["min_duration_s"]
    return out


CAMS: list[dict] = []


def scope(d: dict) -> list[str]:
    """Which cameras the rule actually applies to.  cameras=[cam01] and zones=[entrance] are the
    same rule on this site, so scope is compared instead of the raw camera/zone fields."""
    r = Rule(name="x", type=d.get("type") or "no_helmet", cameras=d.get("cameras") or ["*"],
             zones=d.get("zones") or [], zones_exclude=d.get("zones_exclude") or [])
    return sorted(c["id"] for c in CAMS if r.covers(c["id"], c["zone"]))


def matches(rule: dict | None, expected: dict) -> tuple[bool, list[str]]:
    if rule is None:
        return False, ["no rule"]
    exp = factual(expected)
    bad = [k for k, v in exp.items() if k not in ("cameras", "zones") and rule.get(k) != v]
    if scope(rule) != scope(exp):
        bad.append("scope")
    return not bad, bad


def llm_only(text: str, cams: list[dict], zones: dict, llm: LLMClient) -> dict | None:
    try:
        raw = llm.chat(P.llm_prompt(text, cams, zones), fmt=P.llm_schema(), temperature=0.0, max_tokens=300)
        d = P._clean_llm(json.loads(raw), cams, zones)
        d = {k: v for k, v in d.items() if v is not None or k in ("object", "threshold", "time_window")}
        return Rule(name=d.pop("name", "x") or "x", **{k: v for k, v in d.items() if k != "name"}).model_dump()
    except Exception:
        return None


def main() -> None:
    import test_parser as T

    cfg = load_config()
    cams = [{"id": c["id"], "name": c["name"], "zone": c["zone"]} for c in cfg["cameras"]]
    CAMS[:] = cams
    zones = cfg["zones"]
    llm = LLMClient(cfg["llm"]["base_url"], cfg["llm"]["model"], 120)
    if not llm.available():
        raise SystemExit("LLM server not reachable: start Ollama (or hailo-ollama) first")
    sets = {"dev (used to build the extractor)": [(t, e) for t, e in T.CASES if e is not None],
            "held-out (never used for tuning)": HELDOUT}
    out = {"llm": cfg["llm"]["model"], "fields_compared": ["type", "object", "threshold", "time_window", "min_duration_s when stated", "scope (cameras the rule applies to)"], "sets": {}}
    for name, cases in sets.items():
        score = {"llm_only": 0, "rules": 0, "llm+check": 0}
        ms: list[float] = []
        rows = []
        for text, exp in cases:
            a = llm_only(text, cams, zones, llm)
            b = P.parse(text, cams, zones, llm=None).rule
            t0 = time.perf_counter()
            c = P.parse(text, cams, zones, llm=llm).rule
            ms.append((time.perf_counter() - t0) * 1000)
            res = {k: matches(r.model_dump() if hasattr(r, "model_dump") else r, exp)
                   for k, r in (("llm_only", a), ("rules", b), ("llm+check", c))}
            for k, (ok, _) in res.items():
                score[k] += ok
            rows.append({"text": text, **{k: {"ok": ok, "wrong_fields": bad} for k, (ok, bad) in res.items()}})
        n = len(cases)
        out["sets"][name] = {"sentences": n, "correct": score,
                             "accuracy": {k: round(v / n, 3) for k, v in score.items()},
                             "llm+check_latency_ms": {"mean": round(sum(ms) / n), "max": round(max(ms))},
                             "rows": rows}
        print(f"{name}: {n} sentences (zh + en), model {cfg['llm']['model']}")
        for k, v in score.items():
            print(f"  {k:10s} {v}/{n} = {v / n:.0%}")
        print(f"  llm+check latency: mean {round(sum(ms) / n)} ms, max {round(max(ms))} ms")
    for d in (work_dir() / "results", ROOT / "docs" / "results"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "rule_parsing.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
