"""Report stats and fallbacks on a synthetic event store (no models, no network)."""
from __future__ import annotations

import json
import time

import pytest

from sayguard.events import EventStore
from sayguard.report import action_issues, build_report, compute_stats, unsupported_numbers
from sayguard.types import Event

CAMS = [{"id": "cam01", "name": "Gate A", "zone": "entrance"},
        {"id": "cam03", "name": "Scaffold North", "zone": "zone_a"},
        {"id": "cam10", "name": "Site Office", "zone": "office"}]
HELMET = ("r_helmet", "Hard hat required on site", "no_helmet", "high", "Person without a hard hat")
FORK = ("r_fork", "Forklift proximity", "object_near_person", "medium", "Person within 3 m of a forklift")


def at(h: int, m: int) -> float:
    """Local wall-clock time on a fixed day, so hour buckets do not depend on the timezone."""
    return time.mktime((2026, 10, 18, h, m, 0, 0, 0, -1))


NOW = at(18, 0)
ROWS = [  # (cam, rule, verified, hh, mm)
    ("cam01", HELMET, True, 8, 5),
    ("cam01", HELMET, True, 8, 20),
    ("cam01", HELMET, None, 8, 45),
    ("cam01", HELMET, False, 9, 10),
    ("cam01", FORK, True, 13, 30),
    ("cam01", FORK, None, 14, 0),
    ("cam01", HELMET, False, 15, 15),
    ("cam03", HELMET, True, 10, 0),
    ("cam03", FORK, True, 10, 30),
    ("cam03", HELMET, None, 16, 40),
    ("cam03", HELMET, False, 8, 30),
    ("cam03", FORK, False, 11, 0),
]


def _event(i: int, cam: str, rule: tuple, verified: bool | None, ts: float) -> Event:
    rid, name, rtype, sev, msg = rule
    cam_name = next(c["name"] for c in CAMS if c["id"] == cam)
    return Event(id=f"e{i:02d}", ts=ts, cam_id=cam, cam_name=cam_name, zone="", rule_id=rid, rule_name=name,
                 rule_type=rtype, severity=sev, message=msg, verified=verified)


@pytest.fixture()
def store(tmp_path) -> EventStore:
    s = EventStore(tmp_path / "events.db", tmp_path / "snaps")
    for i, (cam, rule, ver, h, m) in enumerate(ROWS):
        s.add(_event(i, cam, rule, ver, at(h, m)))
    s.add(_event(99, "cam03", HELMET, True, NOW - 30 * 3600))   # outside the 24 h window
    return s


class FakeLLM:
    model = "fake-llm"

    def __init__(self, reply: str | None = None, error: Exception | None = None):
        self.reply, self.error, self.calls = reply, error, []

    def chat(self, messages, fmt=None, temperature=0.2, max_tokens=512):
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        if self.error:
            raise self.error
        return self.reply


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_template_stats_and_markdown(store, lang):
    r = build_report(store, None, CAMS, lang, 24.0, now=NOW)
    s = r["stats"]
    assert r["generator"] == "template"
    assert (s["total"], s["confirmed"], s["false_alarms"], s["unchecked"], s["violations"]) == (12, 5, 4, 2, 8)
    assert s["not_applicable"] == 1                                # the unverified forklift alert
    assert s["by_camera"] == {"Gate A": 5, "Scaffold North": 3}
    assert s["false_alarms_by_camera"] == {"Gate A": 2, "Scaffold North": 2}
    assert s["by_rule"] == {"Hard hat required on site": 5, "Forklift proximity": 3}
    assert s["by_severity"] == {"high": 5, "medium": 3, "low": 0}
    assert s["hourly"][8] == 3 and sum(s["hourly"]) == 8          # the 08:30 false alarm is not counted
    assert s["peak_hour"] == "08:00-09:00" and s["peak_hour_count"] == 3
    assert s["busiest_camera"] == "Gate A" and s["top_rule"] == "Hard hat required on site"
    assert s["first_alert"].endswith("08:05") and s["last_alert"].endswith("16:40")
    assert s["cameras_total"] == 3 and s["cameras_with_violations"] == 2
    assert 1 <= len(s["examples"]) <= 3
    md = r["markdown"]
    assert "Gate A" in md
    assert "on-device" in md
    heads = ["摘要", "重點發現", "建議措施"] if lang == "zh" else ["Summary", "Key findings", "Recommended actions"]
    assert all(f"## {h}" in md for h in heads)
    actions = md.split(f"## {heads[2]}")[1].split("---")[0]
    assert sum(1 for line in actions.splitlines() if line.startswith("- ")) == 3
    if lang == "en":
        assert "4 alerts were filtered as false alarms by the double-check" in md
        assert "(5 confirmed by the double-check, 2 not yet checked, 1 not subject to the double-check)" in md
    else:
        assert "4 則經複核判定為誤報" in md
        assert "（複核確認 5 則、尚待複核 2 則、1 則屬不經複核的規則）" in md


def test_short_window_targets_hard_hat_camera(store):
    r = build_report(store, None, CAMS, "en", 6.0, now=NOW)
    s = r["stats"]
    assert (s["total"], s["false_alarms"], s["violations"]) == (4, 1, 3)
    assert s["busiest_camera"] == "Gate A" and s["busiest_camera_top_rule"] == "Forklift proximity"
    assert s["no_helmet_top_camera"] == "Scaffold North"
    assert "1 alert was filtered as a false alarm by the double-check" in r["markdown"]
    assert "spare hard hats and a reminder sign at Scaffold North" in r["markdown"]


def test_rules_never_double_checked_are_not_pending(tmp_path):
    """Only hard-hat alerts are double-checked: forklift alerts are violations, never 'awaiting re-check'."""
    s = EventStore(tmp_path / "e.db", tmp_path / "snaps")
    for i, (h, m) in enumerate([(9, 0), (9, 30), (10, 15)]):
        s.add(_event(i, "cam01", FORK, None, at(h, m)))
    r = build_report(s, None, CAMS, "en", now=NOW)
    st = r["stats"]
    assert (st["violations"], st["unchecked"], st["not_applicable"], st["confirmed"]) == (3, 0, 3, 0)
    assert {e["status"] for e in st["examples"]} == {"not_applicable"}
    md = r["markdown"]
    assert "0 not yet checked, 3 not subject to the double-check" in md
    actions = md.split("## Recommended actions")[1]
    assert "duty officer" not in actions                              # no 'review unchecked snapshots' action
    assert "(not subject to the double-check)" in md and "(not yet checked)" not in md
    facts = FakeLLM(GOOD_ZH)
    build_report(s, facts, CAMS, "en", now=NOW, mode="llm")
    assert '"violations_not_yet_checked": 0' in facts.calls[0]["messages"][-1]["content"]
    assert '"violations_not_subject_to_double_check": 3' in facts.calls[0]["messages"][-1]["content"]


def test_window_is_capped_at_a_week(store):
    r = build_report(store, None, CAMS, "en", 24 * 31, now=NOW)
    assert r["stats"]["hours"] == 168
    assert r["stats"]["total"] == 13                                  # now includes the 30 h old alert


def test_empty_store(tmp_path):
    s = EventStore(tmp_path / "e.db", tmp_path / "snaps")
    for lang in ("zh", "en"):
        r = build_report(s, None, CAMS, lang, now=NOW)
        assert r["stats"]["total"] == 0 and r["stats"]["busiest_camera"] is None
        assert r["generator"] == "template" and "## " in r["markdown"]


GOOD_ZH = """## 摘要
过去 24 小时共 12 则警报，有 4 则警报经复核判定为误报，已过滤。
## 重点发现
- Gate A 违规 5 则
- 尖峰时段 08:00-09:00
## 建议措施
- 在 Gate A 于 08:00 前进行工具箱会议
- 08:00-09:00 安排主管巡查 Gate A
- 检查 Scaffold North 镜头
"""


def test_llm_draft_is_used_and_converted(store):
    llm = FakeLLM(GOOD_ZH)
    r = build_report(store, llm, CAMS, "zh", now=NOW, mode="llm")
    assert r["generator"] == "llm", r["note"]
    call = llm.calls[0]
    assert call["temperature"] == 0.3 and call["max_tokens"] == 450
    assert '"violations_by_camera"' in call["messages"][-1]["content"]
    assert "\n\n## 重點發現" in r["markdown"] and "建議措施" in r["markdown"]   # simplified -> traditional
    assert "fake-llm" in r["markdown"]


@pytest.mark.parametrize("llm", [FakeLLM(error=RuntimeError("offline")), FakeLLM(""), FakeLLM("just one line"),
                                 FakeLLM(GOOD_ZH.replace("12 则", "99 则"))])
def test_llm_failures_fall_back_to_template(store, llm):
    r = build_report(store, llm, CAMS, "zh", now=NOW, mode="llm")
    assert r["generator"] == "template" and r["note"]
    assert r["stats"]["total"] == 12 and "Gate A" in r["markdown"]


def test_unsupported_numbers():
    facts = {"alerts_total": 12, "peak_hour": "14:00-15:00"}
    assert unsupported_numbers("12 alerts, peak 2 pm, 5-minute talk", facts) == []
    assert unsupported_numbers("13 alerts", facts) == ["13"]


class SeqLLM:
    """Returns its replies in order: first the (rejected) full draft, then the action bullets."""
    model = "seq-llm"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, messages, fmt=None, temperature=0.2, max_tokens=512):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return self.replies.pop(0)


def picks(*items) -> str:
    """The structured reply the actions step asks for: [(issue number, action text)]."""
    return json.dumps({"actions": [{"issue": i, "action": a} for i, a in items]}, ensure_ascii=False)


def issues_of(store, lang="en"):
    stats = compute_stats(list(store.between(NOW - 24 * 3600, NOW)), CAMS, NOW - 24 * 3600, NOW)
    return action_issues(stats, lang)


def test_rejected_draft_falls_back_to_llm_actions_with_template_numbers(store):
    actions = picks((0, "於尖峰前進行工具箱會議"), (1, "尖峰時段安排主管巡查"), (2, "檢查鏡頭角度"))
    llm = SeqLLM(GOOD_ZH.replace("12 则", "99 则"), actions)
    r = build_report(store, llm, CAMS, "zh", now=NOW, mode="llm", actions="free")
    assert r["generator"] == "llm+template", r["note"]
    first = issues_of(store, "zh")[0]
    assert f"**{first['camera']}**・{first['issue']}：於尖峰前進行工具箱會議" in r["markdown"]
    assert "99" not in r["markdown"]                       # the invented number never reaches the report
    assert '"violations_by_camera"' not in llm.calls[1]["messages"][-1]["content"]   # no counts given for actions
    assert "seq-llm" in r["markdown"]


def test_an_action_with_an_invented_number_gets_the_template_text(store):
    llm = SeqLLM("just one line", picks((0, "Send 12 people there"), (1, "Supervisor walk"), (2, "Check the view")))
    r = build_report(store, llm, CAMS, "en", now=NOW, mode="llm", actions="free")
    assert r["generator"] == "llm+template" and "12 people" not in r["markdown"]
    assert "Supervisor walk" in r["markdown"] and "number not in the stats" in r["note"]


def test_all_three_actions_straying_falls_back_to_the_template(store):
    llm = SeqLLM(picks((0, "Send 12 people"), (1, "Send 13 people"), (2, "Send 14 people")))
    assert build_report(store, llm, CAMS, "en", now=NOW, actions="free")["generator"] == "template"


def test_llm_actions_that_are_not_json_are_rejected(store):
    llm = SeqLLM("- Toolbox talk at Gate A\n- b\n- c")
    assert build_report(store, llm, CAMS, "en", now=NOW, actions="free")["generator"] == "template"


def test_an_action_cannot_name_another_camera(store):
    issues = issues_of(store)
    other = next(x["camera"] for x in issues if x["camera"] != issues[0]["camera"])
    llm = SeqLLM(picks((0, f"Move the spare hard hats to {other}"), (1, "Supervisor walk"), (2, "Check the view")))
    r = build_report(store, llm, CAMS, "en", now=NOW, actions="free")
    assert f"Move the spare hard hats to {other}" not in r["markdown"] and "another camera" in r["note"]


def test_an_action_about_another_kind_of_problem_gets_the_template_text(store):
    issues = issues_of(store)
    k = next(i for i, x in enumerate(issues) if x["kind"] not in ("hard_hat", "false_alarm"))
    rest = [i for i in range(len(issues)) if i != k][:2]
    llm = SeqLLM(picks((k, "Make everyone wear hard hats"), (rest[0], "Supervisor walk"), (rest[1], "Check the view")))
    r = build_report(store, llm, CAMS, "en", now=NOW, actions="free")
    assert "Make everyone wear hard hats" not in r["markdown"] and "another kind of problem" in r["note"]


def test_the_same_issue_twice_is_rejected(store):
    llm = SeqLLM(picks((0, "a"), (0, "b"), (1, "c")))
    assert build_report(store, llm, CAMS, "en", now=NOW, actions="free")["generator"] == "template"


def test_free_text_actions_name_camera_and_issue(store):
    llm = SeqLLM(picks((0, "Toolbox talk before the peak hour"), (1, "Supervisor walk"), (2, "Check the view")))
    r = build_report(store, llm, CAMS, "en", now=NOW, actions="free")
    assert r["generator"] == "llm+template" and len(llm.calls) == 1
    first = issues_of(store)[0]
    assert f"**{first['camera']}** · {first['issue']}: Toolbox talk before the peak hour" in r["markdown"]
    assert "| Gate A |" in r["markdown"]


def test_chinese_actions_use_taiwan_terms(store):
    llm = SeqLLM(picks((0, "检查摄像头角度，减少假警报"), (1, "b"), (2, "c")))
    r = build_report(store, llm, CAMS, "zh", now=NOW, actions="free")
    assert "攝影機" in r["markdown"] and "誤報" in r["markdown"] and "攝像頭" not in r["markdown"]



def plan(*items) -> str:
    return json.dumps({"actions": [{"issue": i, "fix": f} for i, f in items]})


def test_default_actions_are_vetted_fixes_the_llm_picks(store):
    from sayguard.report import TEXT
    llm = SeqLLM(plan((0, 2), (1, 0), (2, 1)))
    r = build_report(store, llm, CAMS, "en", now=NOW)
    assert r["generator"] == "llm+template" and len(llm.calls) == 1
    first = issues_of(store)[0]
    fix = TEXT["en"]["menu"][first["kind"]][2].format(peak=compute_stats(
        list(store.between(NOW - 24 * 3600, NOW)), CAMS, NOW - 24 * 3600, NOW)["peak_hour"])
    assert f"**{first['camera']}** · {first['issue']}: {fix}" in r["markdown"]


def test_a_plan_with_a_repeated_or_unknown_issue_falls_back_to_the_template(store):
    assert build_report(store, SeqLLM(plan((0, 0), (0, 1), (1, 2))), CAMS, "en", now=NOW)["generator"] == "template"
    assert build_report(store, SeqLLM(plan((0, 0), (1, 5), (2, 1))), CAMS, "en", now=NOW)["generator"] == "template"
    assert build_report(store, SeqLLM("not json"), CAMS, "en", now=NOW)["generator"] == "template"
