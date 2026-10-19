"""Measure the daily report's LLM-drafted actions on synthetic days: does every action talk about
a problem the camera it names really had?

Three ways of asking Qwen2.5-1.5B for the 3 recommended actions, on the same seeded days:

  free_text   - the earlier prompt: 3 markdown bullets from a few facts (busiest camera, peak
                hour, camera with most false alarms ...), checked only for invented numbers
  structured  - build_report(actions="free"), sayguard.report._llm_actions: the model picks 3 issues
                by number from the list of (camera, problem) pairs that happened and writes one
                action for each; the camera and the problem are filled in by code, and an action
                that names another camera, a number not in the stats or another kind of problem
                gets the template text instead
  menu        - the default, sayguard.report._llm_menu_actions: the model picks the 3 issues and,
                for each, one of three vetted fixes for that kind of problem; it writes no text

An action is "misattributed" when it names a camera together with a problem that camera did not
have that day (hard hats, false alarms, vehicles, crowding: keyword topics in English and
Chinese).  Needs the Ollama-compatible server from configs/site.yaml.

    python scripts/eval_report.py [--days 20]

The blind-judge scores in docs/results/report_actions_judged.json belong to the run judged there
(docs/results/report_actions_judging/); a new run writes report_actions.json with new samples and no judges.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sayguard import report as R  # noqa: E402
from sayguard.config import load_config, work_dir  # noqa: E402
from sayguard.events import EventStore  # noqa: E402
from sayguard.llm import LLMClient  # noqa: E402
from sayguard.types import Event  # noqa: E402

RULES = [  # (id, name, type, severity, message)
    ("helmet", "Hard hat required on site", "no_helmet", "high", "Person without a hard hat"),
    ("near", "Keep 3 m from the excavator", "object_near_person", "high", "Person near an excavator"),
    ("crowd", "Entrance: more than 2 people", "person_count_over", "medium", "More than 2 people"),
]
TOPICS = {   # topic -> (keywords, which day facts make it true for a camera)
    "hard_hat": (r"安全帽|hard ?hats?|helmets?|ppe", "helmet"),
    "false_alarm": (r"誤報|假警報|false alarms?|misfire", "false"),
    "vehicle": (r"挖土機|堆高機|excavators?|forklifts?|vehicles?|machine", "near"),
    "crowding": (r"人數|擁擠|聚集|crowd|too many people|headcount", "crowd"),
}


def synthetic_day(rng: random.Random, cams: list[dict], now: float) -> list[Event]:
    out, i = [], 0
    hot = rng.sample(cams, rng.randint(3, min(6, len(cams))))
    for c in hot:
        for rule in rng.sample(RULES, rng.randint(1, 2)):
            n = int(rng.paretovariate(1.2) * rng.randint(2, 15))
            for _ in range(min(n, 400)):
                verified = True if rule[2] == "no_helmet" else None
                out.append(Event(id=f"e{i}", ts=now - rng.uniform(60, 23.5 * 3600), cam_id=c["id"], cam_name=c["name"],
                                 zone=c["zone"], rule_id=rule[0], rule_name=rule[1], rule_type=rule[2],
                                 severity=rule[3], message=rule[4], verified=verified))
                i += 1
    for c in rng.sample(hot, rng.randint(0, 2)):    # false alarms the double-check filtered
        for _ in range(rng.randint(3, 40)):
            h = RULES[0]
            out.append(Event(id=f"e{i}", ts=now - rng.uniform(60, 23.5 * 3600), cam_id=c["id"], cam_name=c["name"],
                             zone=c["zone"], rule_id=h[0], rule_name=h[1], rule_type=h[2], severity=h[3],
                             message=h[4], verified=False))
            i += 1
    return out


LABEL = {"helmet": "people without hard hats", "near": "people within 3 m of the excavator",
         "crowd": "more than 2 people at the entrance", "false": "false hard-hat alarms the double-check filtered"}


def readable_truth(events: list[Event]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for e in events:
        k = LABEL["false" if e.verified is False else e.rule_id]
        out.setdefault(e.cam_name, {}).setdefault(k, 0)
        out[e.cam_name][k] += 1
    return out


def facts_by_camera(events: list[Event]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for e in events:
        out.setdefault(e.cam_name, set()).add("false" if e.verified is False else e.rule_id)
    return out


def misattributed(text: str, truth: dict[str, set[str]], own: str | None = None) -> tuple[bool, bool]:
    """-> (names a camera, claims a problem one of the named cameras did not have)."""
    named = [c for c in truth if c.lower() in text.lower()] or ([own] if own else [])
    if not named:
        return False, False
    bad = any(re.search(kw, text, re.I) and not any(fact in truth[c] for c in named)
              for kw, fact in TOPICS.values())
    return True, bad


def free_text_actions(llm: LLMClient, s: dict, lang: str) -> list[str] | None:
    """The earlier prompt (SayGuard before the structured actions), kept here as the baseline."""
    t = R.TEXT[R._lang(lang)]
    facts = {"busiest_camera": s["busiest_camera"], "busiest_camera_main_rule": s["busiest_camera_top_rule"],
             "peak_hour": s["peak_hour"], "camera_with_most_hard_hat_violations": s["no_helmet_top_camera"],
             "second_camera": list(s["by_camera"])[1] if len(s["by_camera"]) > 1 else None,
             "camera_with_most_false_alarms": next(iter(s["false_alarms_by_camera"]), None)}
    system = (f"You help a construction safety officer. Reply in {t['lang_name']} with exactly 3 markdown bullet "
              "lines ('- ...'), each one concrete action under 30 words that names a camera and, where useful, the "
              "peak hour from the JSON. No numbers other than the hour given, no headings, no other text.")
    try:
        raw = llm.chat([{"role": "system", "content": system},
                        {"role": "user", "content": t["ask_actions"] + "\n\n" + json.dumps(facts, ensure_ascii=False)}],
                       temperature=0.3, max_tokens=220)
    except Exception:
        return None
    text = R._clean(raw, lang)
    bullets = [m.group(1).strip() for m in re.finditer(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+(.+)$", text)]
    if len(bullets) < 3 or R.unsupported_numbers("\n".join(bullets[:3]), facts):
        return None
    return bullets[:3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=20)
    a = ap.parse_args()
    cfg = load_config()
    cams = [{"id": c["id"], "name": c["name"], "zone": c["zone"]} for c in cfg["cameras"]]
    llm = LLMClient(cfg["llm"]["base_url"], cfg["llm"]["model"], 120)
    if not llm.available():
        raise SystemExit("LLM server not reachable: start Ollama (or hailo-ollama) first")
    rng = random.Random(20261019)
    now = time.time()
    score = {k: {"days": 0, "accepted": 0, "actions": 0, "name_a_camera": 0, "misattributed": 0}
             for k in ("free_text", "structured", "menu")}
    rows = []
    d = work_dir() / "tmp_eval_report"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    if True:
        for day in range(a.days):
            lang = "en" if day % 2 == 0 else "zh"
            events = synthetic_day(rng, cams, now)
            store = EventStore(Path(d) / f"day{day}.db", Path(d) / "snaps")
            for e in events:
                store.add(e)
            stats = R.compute_stats(list(store.between(now - 24 * 3600, now + 1)), cams, now - 24 * 3600, now)
            truth = facts_by_camera(events)
            ft = free_text_actions(llm, stats, lang)
            st, note = R._llm_actions(llm, stats, lang)
            mn, note_m = R._llm_menu_actions(llm, stats, lang)
            own = [x["camera"] for x in R.action_issues(stats, lang)]
            row = {"day": day, "lang": lang, "truth": readable_truth(events)}
            templates = {R.TEXT[R._lang(lang)][f"fix_{k}"] for k in ("hard_hat", "vehicle", "crowding", "zone", "false_alarm")}
            for k, acts in (("free_text", ft), ("structured", st), ("menu", mn)):
                sc = score[k]
                sc["days"] += 1
                if not acts:
                    row[k] = None
                    continue
                sc["accepted"] += 1
                res = []
                for j, act in enumerate(acts):
                    cam = None
                    if k in ("structured", "menu"):
                        cam = re.match(r"\*\*(.+?)\*\*", act).group(1)
                    named, bad = misattributed(act, truth, cam)
                    sc["actions"] += 1
                    sc["name_a_camera"] += named
                    sc["misattributed"] += bad
                    src = "template" if k == "structured" and any(act.endswith(x) for x in templates) else "llm"
                    res.append({"text": act, "misattributed": bad, "source": src})
                row[k] = res
            row["structured_note"] = note
            row["menu_note"] = note_m
            rows.append(row)
            print(f"day {day} ({lang}): free_text {'ok' if ft else 'rejected'}, structured {'ok' if st else note}, "
                  f"menu {'ok' if mn else note_m}", flush=True)
    for k, sc in score.items():
        sc["misattributed_pct"] = round(100 * sc["misattributed"] / max(1, sc["actions"]), 1)
        sc["accepted_pct"] = round(100 * sc["accepted"] / max(1, sc["days"]), 1)
    out = {"llm": cfg["llm"]["model"], "days": a.days, "seed": 20261019, "score": score, "rows": rows}
    for d in (work_dir() / "results", ROOT / "docs" / "results"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "report_actions.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(score, indent=1))


if __name__ == "__main__":
    main()
