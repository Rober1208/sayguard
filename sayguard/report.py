"""Daily safety report built from the alert store.

Every number is computed here, deterministically, from the SQLite events; the
small local LLM (Qwen2.5-1.5B on Ollama / hailo-ollama) only turns those
numbers into prose for the safety officer.  Its draft is accepted only if it keeps
the three report sections and uses no number that is missing from the stats it was
given; otherwise, or when the LLM is offline, a template writes the same report, so
the endpoint always answers.  Alerts the double-check (sayguard.verify: CLIP probe or VLM)
rejected are reported as filtered false alarms and are never counted as violations.  Only
hard-hat alerts are double-checked, so alerts from other rules are violations "not subject
to the double-check", never "pending re-check" (they would wait forever).

The window is capped at a week: the stats are computed in Python over the rows, and a
week at demo alert rates (~90k rows of 9 small columns, read in pages) keeps both the
memory and each hold of the store lock small while the camera thread keeps writing.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .events import EventStore
    from .llm import LLMClient

SEVERITIES = ("high", "medium", "low")
MAX_HOURS = 24 * 7
CHECKED_TYPE = "no_helmet"  # the only rule type the pipeline sends to the double-check
# what compute_stats reads; box/extra/snapshot are never needed and are the bulk of a row
COLUMNS = ("id", "ts", "cam_id", "cam_name", "rule_name", "rule_type", "severity", "message", "verified")

TEXT: dict[str, dict[str, str]] = {
    "zh": {
        "lang_name": "Traditional Chinese (繁體中文, Taiwan usage)",
        "title_day": "工地安全日報",
        "title_hours": "工地安全報告（過去 {h} 小時）",
        "h_summary": "摘要", "h_findings": "重點發現", "h_actions": "建議措施",
        "no_alerts": "過去 {h} 小時內，{cams} 支攝影機皆未發出任何警報。",
        "alerts": "過去 {h} 小時共發出 {total} 則警報。",
        "fa": "其中 {fa} 則經複核判定為誤報，已過濾，不列入違規；實際違規 {viol} 則（{split}）。",
        "no_fa": "複核未發現誤報，實際違規 {viol} 則（{split}）。",
        "split": "複核確認 {conf} 則、尚待複核 {unc} 則",
        "split_na": "、{na} 則屬不經複核的規則",
        "where": "違規最多的是 **{cam}**（{cam_n} 則），尖峰時段為 {peak}（{peak_n} 則），最常觸發的規則是「{rule}」。",
        "clean": "過濾誤報後，沒有需要處理的違規。",
        "fa_phrase": "有 {fa} 則警報經複核判定為誤報，已過濾",
        "table": "| 攝影機 | 違規 | 複核確認 | 已過濾誤報 |",
        "rules": "規則分布：{items}",
        "rule_item": "「{name}」{n} 則",
        "sep": "、", "join": "",
        "severity": "嚴重度：高 {high} 則、中 {medium} 則、低 {low} 則",
        "timing": "尖峰時段 {peak}（{peak_n} 則）；首次違規 {first}，最後一次 {last}",
        "coverage": "{k}/{n} 支攝影機有違規紀錄",
        "examples": "違規範例：",
        "example": "`{time}` {camera}：{message}（{status}）",
        "confirmed": "複核確認", "unchecked": "待複核", "not_applicable": "不經複核",
        "no_violations": "沒有需要處理的違規。",
        "act_talk": "在 {start} 前於 **{cam}** 進行 5 分鐘工具箱會議（toolbox talk），重申「{rule}」。",
        "act_helmet": "在 {cam} 放置備用安全帽與提醒標示，並於 {peak} 安排主管到場抽查。",
        "act_patrol": "於 {peak} 安排主管到 {cam} 一帶巡查。",
        "act_review": "請值班人員檢視 {unc} 則尚未經複核的警報快照。",
        "act_camera": "檢查 {cam} 的鏡頭角度與照明：該處有 {n} 則警報經複核判定為誤報。",
        "act_second": "將 {cam}（{n} 則）列為下週加強宣導的第二優先地點。",
        "act_keep": "維持現行規則，並在每週工具箱會議上檢討本報告。",
        "act_coverage": "抽查各攝影機畫面是否仍涵蓋所有出入口與作業區。",
        "act_voice": "如有新的危險作業，直接用語音新增規則，例如「倉庫堆高機三公尺內不能有人」。",
        "footer": "報告期間 {start} – {end}（{h} 小時）· {gen} · 由 SayGuard 在現場產生，資料不上傳雲端。",
        "gen_llm": "敘述由本地 LLM（{model}）撰寫，數字由系統統計",
        "gen_template": "以範本產生",
        "gen_hybrid": "數字與表格由系統統計，建議措施由本地 LLM（{model}）撰寫",
        "ask_actions": "根據以下統計，用繁體中文寫出明天的 3 項具體安全改善措施。",
        "issue_fa": "誤報偏多（已由複核過濾）",
        "menu": {
            "hard_hat": ["在 {peak} 前進行 5 分鐘工具箱會議，重申安全帽規定。",
                         "在入口放置備用安全帽與提醒標示。",
                         "{peak} 安排主管到場抽查安全帽配戴。"],
            "vehicle": ["劃設與車輛保持距離的人行通道並以圍欄區隔。",
                        "車輛作業時安排專人指揮，禁止人員進入作業半徑。",
                        "向駕駛與作業人員重申車輛周圍的安全距離。"],
            "crowding": ["錯開人員進出與交班時間。",
                         "在此區張貼人數上限並設置排隊動線。",
                         "{peak} 安排專人管制此區人數。"],
            "zone": ["檢查管制區的圍欄與標示是否到位。",
                     "提醒人員此區為管制區，進入需事先核准。",
                     "檢討哪些人員確實需要進入此區。"],
            "false_alarm": ["檢查此攝影機的角度與照明，讓複核能清楚看到頭部。",
                            "清潔鏡頭並移除遮擋畫面的物品。",
                            "抽查被過濾的警報快照，確認判斷正確。"],
        },
        "fix_hard_hat": "在入口放置備用安全帽與提醒標示，並於尖峰時段抽查。",
        "fix_vehicle": "劃設與車輛保持距離的人行通道，車輛移動時安排指揮人員。",
        "fix_crowding": "錯開人員進出時間，並張貼此區域的人數上限。",
        "fix_zone": "提醒人員此區為管制區，並確認圍欄與標示到位。",
        "fix_false_alarm": "檢查此攝影機的角度與照明，讓複核能清楚看到頭部。",
        "ask": "請只根據以下 JSON 統計資料，用繁體中文撰寫今日安全報告。",
        "length": "Keep it under 350 Chinese characters.",
    },
    "en": {
        "lang_name": "English",
        "title_day": "Daily Safety Report",
        "title_hours": "Safety Report (last {h} h)",
        "h_summary": "Summary", "h_findings": "Key findings", "h_actions": "Recommended actions",
        "no_alerts": "None of the {cams} cameras raised an alert in the last {h} hours.",
        "alerts": "{total_n} raised in the last {h} hours.",
        "fa": "{fa_n} {fa_were} filtered as {fa_as} by the double-check and not counted, leaving {viol_n} ({split}).",
        "no_fa": "The double-check filtered no false alarms, so there {viol_were} {viol_n} ({split}).",
        "split": "{conf} confirmed by the double-check, {unc} not yet checked",
        "split_na": ", {na} not subject to the double-check",
        "where": "Most came from **{cam}** ({cam_n}); the peak hour was {peak} ({peak_n}) and the most frequent rule was “{rule}”.",
        "clean": "After filtering false alarms there are no violations to follow up.",
        "fa_phrase": "{fa} alerts were filtered as false alarms by the double-check",
        "table": "| Camera | Violations | Confirmed by double-check | False alarms filtered |",
        "rules": "By rule: {items}",
        "rule_item": "“{name}” {n}",
        "sep": ", ", "join": " ",
        "severity": "Severity: high {high}, medium {medium}, low {low}",
        "timing": "Peak hour {peak} ({peak_n}); first violation {first}, last {last}",
        "coverage": "{k} of {n} cameras recorded violations",
        "examples": "Examples:",
        "example": "`{time}` {camera}: {message} ({status})",
        "confirmed": "confirmed by double-check", "unchecked": "not yet checked", "not_applicable": "not subject to the double-check",
        "no_violations": "No violations to follow up.",
        "act_talk": "Hold a 5-minute toolbox talk at **{cam}** before {start}, restating “{rule}”.",
        "act_helmet": "Keep spare hard hats and a reminder sign at {cam}, and have a supervisor spot-check there during {peak}.",
        "act_patrol": "Have a supervisor walk the {cam} area during {peak}.",
        "act_review": "Have the duty officer review the {unc} alert snapshots the double-check has not judged yet.",
        "act_camera": "Check the camera angle and lighting at {cam}: {n} of its alerts were filtered as false alarms by the double-check.",
        "act_second": "Make {cam} ({n} violations) the second priority for PPE reminders next week.",
        "act_keep": "Keep the current rules and review this report at the weekly toolbox talk.",
        "act_coverage": "Spot-check that the cameras still cover every entrance and work area.",
        "act_voice": "Add rules for new hazards by voice, e.g. “Warn me if anyone is within 3 m of a forklift”.",
        "footer": "Period {start} – {end} ({h} h) · {gen} · generated on site by SayGuard; no data leaves the site.",
        "gen_llm": "narrative by the local LLM ({model}), numbers computed by SayGuard",
        "gen_template": "template report",
        "gen_hybrid": "numbers and tables computed by SayGuard, actions drafted by the local LLM ({model})",
        "ask_actions": "Based on the stats below, write 3 concrete safety actions for tomorrow in English.",
        "issue_fa": "many false alarms (filtered by the double-check)",
        "menu": {
            "hard_hat": ["Hold a 5-minute toolbox talk before {peak}, restating the hard-hat rule.",
                         "Keep spare hard hats and a reminder sign at the way in.",
                         "Have a supervisor spot-check hard hats during {peak}."],
            "vehicle": ["Mark a walkway clear of the vehicles and fence it off.",
                        "Assign a spotter while vehicles operate and keep people out of their working radius.",
                        "Brief drivers and workers on the safe distance around vehicles."],
            "crowding": ["Stagger arrivals and shift changes.",
                         "Post the maximum number of people for this area and mark a queue.",
                         "Have someone manage the number of people here during {peak}."],
            "zone": ["Check the barrier and signs of the restricted zone are in place.",
                     "Remind workers the zone is restricted and needs prior approval.",
                     "Review who really needs access to this zone."],
            "false_alarm": ["Check this camera's angle and lighting so the double-check sees heads clearly.",
                            "Clean the lens and remove anything blocking the view.",
                            "Spot-check the filtered alert snapshots to confirm the double-check was right."],
        },
        "fix_hard_hat": "Keep spare hard hats and a reminder sign at the way in, and spot-check during the peak hour.",
        "fix_vehicle": "Mark a walkway clear of the vehicles and assign a spotter while they move.",
        "fix_crowding": "Stagger arrivals and post the maximum number of people for this area.",
        "fix_zone": "Remind workers the area is restricted and check the barrier and signs are in place.",
        "fix_false_alarm": "Check this camera's angle and lighting so the double-check sees heads clearly.",
        "ask": "Write today's safety report in English using ONLY the JSON stats below.",
        "length": "Keep it under 200 words.",
    },
}


def _lang(lang: str) -> str:
    return "en" if str(lang).lower().startswith("en") else "zh"


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def _stamp(ts: float, fmt: str = "%m/%d %H:%M") -> str:
    return time.strftime(fmt, time.localtime(ts))


def _hour_label(h: int) -> str:
    return f"{h:02d}:00-{(h + 1) % 24:02d}:00"


def _ranked(c: Counter) -> dict[str, int]:
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def _en_count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _check_status(e: dict) -> str:
    """'confirmed' | 'unchecked' (hard-hat alert not judged yet) | 'not_applicable'."""
    if e["verified"] is True:
        return "confirmed"
    return "unchecked" if e.get("rule_type") == CHECKED_TYPE else "not_applicable"


def _examples(real: list[dict], cam_name, k: int = 3) -> list[dict]:
    """Confirmed and high-severity first, one per camera/rule pair where possible."""
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    order = sorted(real, key=lambda e: (e["verified"] is not True, rank.get(e["severity"], 3), -e["ts"]))
    picked: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for e in order:
        key = (e["cam_id"], e["rule_name"])
        if key not in seen:
            picked.append(e)
            seen.add(key)
            if len(picked) == k:
                break
    ids = {e["id"] for e in picked}
    picked += [e for e in order[:2 * k] if e["id"] not in ids][:k - len(picked)]
    return [{"time": _stamp(e["ts"]), "camera": cam_name(e), "rule": e["rule_name"], "message": e["message"],
             "status": _check_status(e)}
            for e in sorted(picked, key=lambda e: e["ts"])]


def compute_stats(events: list[dict], cameras: list[dict], t0: float, t1: float) -> dict[str, Any]:
    """Counts over the window.  'Violations' are alerts the double-check did not reject
    (confirmed + not yet checked + not subject to the double-check); every breakdown except
    false_alarms_by_camera uses them.  'unchecked' counts only hard-hat alerts still
    waiting for the double-check; 'not_applicable' counts alerts of rules it never checks."""
    names = {c["id"]: c.get("name") or c["id"] for c in cameras}

    def cam_name(e: dict) -> str:
        return names.get(e["cam_id"]) or e.get("cam_name") or e["cam_id"]

    events = sorted(events, key=lambda e: e["ts"])
    real = [e for e in events if e["verified"] is not False]
    false = [e for e in events if e["verified"] is False]
    by_cam = Counter(cam_name(e) for e in real)
    conf_cam = Counter(cam_name(e) for e in real if e["verified"] is True)
    fa_cam = Counter(cam_name(e) for e in false)
    detail = [{"camera": n, "violations": by_cam[n], "confirmed": conf_cam[n], "false_alarms": fa_cam[n]}
              for n in sorted(set(by_cam) | set(fa_cam), key=lambda n: (-by_cam[n], -fa_cam[n], n))]
    by_rule = _ranked(Counter(e["rule_name"] for e in real))
    sev = Counter(e["severity"] for e in real)
    by_severity = {s: sev.pop(s, 0) for s in SEVERITIES} | dict(sev)
    hourly = [0] * 24
    for e in real:
        hourly[time.localtime(e["ts"]).tm_hour] += 1
    peak = max(range(24), key=lambda h: (hourly[h], -h)) if real else None
    busiest = detail[0]["camera"] if real else None
    top_rule = next(iter(by_rule), None)
    busiest_rules = _ranked(Counter(e["rule_name"] for e in real if cam_name(e) == busiest))
    cam_rule = Counter((cam_name(e), e["rule_name"], e.get("rule_type") or "") for e in real)
    helmet_cam = _ranked(Counter(cam_name(e) for e in real if e.get("rule_type") == "no_helmet"))
    total = len(events)
    return {
        "period_start": _stamp(t0, "%Y-%m-%d %H:%M"), "period_end": _stamp(t1, "%Y-%m-%d %H:%M"),
        "hours": round((t1 - t0) / 3600, 2),
        "cameras_total": len(cameras),
        "cameras_with_violations": len(by_cam),
        "total": total,
        "confirmed": sum(1 for e in real if e["verified"] is True),
        "false_alarms": len(false),
        "unchecked": sum(1 for e in real if e["verified"] is None and e.get("rule_type") == CHECKED_TYPE),
        "not_applicable": sum(1 for e in real if e["verified"] is None and e.get("rule_type") != CHECKED_TYPE),
        "violations": len(real),
        "false_alarm_share_pct": round(100 * len(false) / total) if total else 0,
        "by_rule": by_rule,
        "by_camera": {d["camera"]: d["violations"] for d in detail if d["violations"]},
        "by_severity": by_severity,
        "by_camera_detail": detail,
        "false_alarms_by_camera": _ranked(fa_cam),
        "hourly": hourly,
        "peak_hour": _hour_label(peak) if peak is not None else None,
        "peak_hour_count": hourly[peak] if peak is not None else 0,
        "busiest_camera": busiest,
        "busiest_camera_count": by_cam[busiest] if busiest else 0,
        "top_rule": top_rule,
        "top_rule_count": by_rule.get(top_rule, 0) if top_rule else 0,
        "first_alert": _stamp(real[0]["ts"]) if real else None,
        "last_alert": _stamp(real[-1]["ts"]) if real else None,
        "busiest_camera_top_rule": next(iter(busiest_rules), None),
        "by_camera_rule": [{"camera": c, "rule": r, "rule_type": rt, "n": n}
                           for (c, r, rt), n in sorted(cam_rule.items(), key=lambda kv: (-kv[1], kv[0]))],
        "no_helmet_violations": sum(helmet_cam.values()),
        "no_helmet_top_camera": next(iter(helmet_cam), None),
        "examples": _examples(real, cam_name),
    }


# ------------------------------------------------------------------ template

def _summary(s: dict, t: dict[str, str]) -> str:
    h = _num(s["hours"])
    if not s["total"]:
        return t["no_alerts"].format(h=h, cams=s["cameras_total"])
    split = t["split"].format(conf=s["confirmed"], unc=s["unchecked"])
    if s["not_applicable"]:
        split += t["split_na"].format(na=s["not_applicable"])
    v = {"h": h, "total": s["total"], "fa": s["false_alarms"], "viol": s["violations"], "split": split,
         "total_n": _en_count(s["total"], "alert") + (" was" if s["total"] == 1 else " were"),
         "fa_n": _en_count(s["false_alarms"], "alert"), "fa_were": "was" if s["false_alarms"] == 1 else "were",
         "fa_as": "a false alarm" if s["false_alarms"] == 1 else "false alarms",
         "viol_n": _en_count(s["violations"], "violation"), "viol_were": "is" if s["violations"] == 1 else "are"}
    parts = [t["alerts"].format(**v), (t["fa"] if s["false_alarms"] else t["no_fa"]).format(**v)]
    if s["violations"]:
        parts.append(t["where"].format(cam=s["busiest_camera"], cam_n=s["busiest_camera_count"],
                                       peak=s["peak_hour"], peak_n=s["peak_hour_count"], rule=s["top_rule"]))
    else:
        parts.append(t["clean"])
    return t["join"].join(parts)


def _cell(x: Any) -> str:
    return str(x).replace("|", "\\|")


def _findings(s: dict, t: dict[str, str]) -> list[str]:
    out: list[str] = []
    if s["by_camera_detail"]:
        out += [t["table"], "|---|---:|---:|---:|"]
        out += [f"| {_cell(d['camera'])} | {d['violations']} | {d['confirmed']} | {d['false_alarms']} |"
                for d in s["by_camera_detail"]]
        out.append("")
    if not s["violations"]:
        return out + [f"- {t['no_violations']}"]
    items = t["sep"].join(t["rule_item"].format(name=k, n=n) for k, n in list(s["by_rule"].items())[:4])
    out.append(f"- {t['rules'].format(items=items)}")
    out.append(f"- {t['severity'].format(**{k: s['by_severity'].get(k, 0) for k in SEVERITIES})}")
    out.append(f"- {t['timing'].format(peak=s['peak_hour'], peak_n=s['peak_hour_count'], first=s['first_alert'], last=s['last_alert'])}")
    out.append(f"- {t['coverage'].format(k=s['cameras_with_violations'], n=s['cameras_total'])}")
    if s["examples"]:
        out.append(f"- {t['examples']}")
        out += [f"  - {t['example'].format(time=e['time'], camera=e['camera'], message=e['message'], status=t[e['status']])}"
                for e in s["examples"]]
    return out


def _actions(s: dict, t: dict[str, str]) -> list[str]:
    acts: list[str] = []
    if s["violations"]:
        cam, peak = s["busiest_camera"], s["peak_hour"]
        acts.append(t["act_talk"].format(cam=cam, start=peak.split("-")[0], rule=s["busiest_camera_top_rule"]))
        helmet_cam = s["no_helmet_top_camera"]
        acts.append(t["act_helmet"].format(cam=helmet_cam, peak=peak) if helmet_cam
                    else t["act_patrol"].format(cam=cam, peak=peak))
    if s["unchecked"]:
        acts.append(t["act_review"].format(unc=s["unchecked"]))
    if s["false_alarms_by_camera"]:
        cam, n = next(iter(s["false_alarms_by_camera"].items()))
        acts.append(t["act_camera"].format(cam=cam, n=n))
    if len(s["by_camera"]) >= 2:
        cam, n = list(s["by_camera"].items())[1]
        acts.append(t["act_second"].format(cam=cam, n=n))
    acts += [t["act_keep"], t["act_coverage"], t["act_voice"]]
    return acts[:3]


def template_body(s: dict, lang: str) -> str:
    t = TEXT[_lang(lang)]
    lines = [f"## {t['h_summary']}", _summary(s, t), "", f"## {t['h_findings']}", *_findings(s, t), "",
             f"## {t['h_actions']}", *[f"- {a}" for a in _actions(s, t)]]
    return "\n".join(lines)


# ------------------------------------------------------------------ LLM

def llm_facts(s: dict) -> dict[str, Any]:
    """The stats renamed for a 1.5B model: explicit names, no 24-slot arrays."""
    return {
        "period": f"{s['period_start']} - {s['period_end']} ({_num(s['hours'])} h)",
        "cameras_monitored": s["cameras_total"],
        "cameras_with_violations": s["cameras_with_violations"],
        "alerts_total": s["total"],
        "false_alarms_filtered_by_double_check": s["false_alarms"],
        "false_alarm_share_pct": s["false_alarm_share_pct"],
        "violations_total": s["violations"],
        "violations_confirmed_by_double_check": s["confirmed"],
        "violations_not_yet_checked": s["unchecked"],
        "violations_not_subject_to_double_check": s["not_applicable"],
        "violations_by_camera": s["by_camera"],
        "violations_by_rule": s["by_rule"],
        "violations_by_severity": s["by_severity"],
        "violations_by_hour": {_hour_label(h): n for h, n in enumerate(s["hourly"]) if n},
        "peak_hour": s["peak_hour"],
        "busiest_camera": s["busiest_camera"],
        "busiest_camera_main_rule": s["busiest_camera_top_rule"],
        "most_frequent_rule": s["top_rule"],
        "camera_with_most_hard_hat_violations": s["no_helmet_top_camera"],
        "first_violation": s["first_alert"],
        "last_violation": s["last_alert"],
        "example_violations": s["examples"],
        "false_alarms_by_camera": s["false_alarms_by_camera"],
    }


def llm_prompt(facts: dict, lang: str) -> list[dict]:
    t = TEXT[_lang(lang)]
    system = (
        "You assist the safety officer of a construction site and write the daily safety report "
        f"in {t['lang_name']}.\n"
        "Rules:\n"
        "- Use ONLY numbers, names and times that appear in the JSON stats. Never invent numbers, causes or people.\n"
        "- false_alarms_filtered_by_double_check are alerts the on-device double-check (a second model that looks at each hard-hat alert crop) rejected. They are NOT "
        f"violations. Mention them once, as: {t['fa_phrase'].format(fa=facts['false_alarms_filtered_by_double_check'])}.\n"
        "- The double-check covers hard-hat alerts only. violations_not_subject_to_double_check come from other rules: they are "
        "violations, never 'awaiting review'.\n"
        "- Markdown with exactly these three sections and nothing else:\n"
        f"## {t['h_summary']}\n(2-3 sentences)\n"
        f"## {t['h_findings']}\n(3-4 bullets)\n"
        f"## {t['h_actions']}\n(exactly 3 bullets: concrete actions naming the camera and the peak hour, "
        "e.g. a short toolbox talk at the busiest camera before the peak hour)\n"
        f"- {t['length']} No title, no greeting, no sign-off."
    )
    user = t["ask"] + "\n\n" + json.dumps(facts, ensure_ascii=False, indent=1)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_CC: Any = None


def _to_traditional(text: str) -> str:
    """Qwen often answers in Simplified Chinese; the report is for a Taiwan site."""
    global _CC
    try:
        if _CC is None:
            from opencc import OpenCC
            _CC = OpenCC("s2twp")
        return _CC.convert(text)
    except Exception:
        return text


def _clean(raw: str, lang: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*\n|\n?```\s*$", "", text).strip()
    text = re.sub(r"(?m)^#\s+", "## ", text)
    text = re.sub(r"\n(#{2,4}\s)", r"\n\n\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _to_traditional(text) if lang == "zh" else text


def _well_formed(text: str) -> bool:
    """Three section headings and at least three bullets in the last one (catches truncation)."""
    heads = list(re.finditer(r"(?m)^#{2,4}\s+\S", text))
    if len(heads) < 3:
        return False
    return len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S", text[heads[-1].end():])) >= 3


_NUM = re.compile(r"\d+(?:\.\d+)?")
_DURATION = re.compile(r"\d+(?:\.\d+)?\s*-?\s*(?:分鐘|分钟|minutes?|mins?)", re.I)


def unsupported_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in the draft that are not in the stats (list numbering 1-4, 12-hour
    clock forms of listed times and durations like '5 minutes' are tolerated)."""
    src = json.dumps(facts, ensure_ascii=False)
    allowed = {float(x) for x in _NUM.findall(src)} | {1.0, 2.0, 3.0, 4.0}
    allowed |= {float(int(h) % 12 or 12) for h in re.findall(r"(\d{1,2}):\d{2}", src)}
    bad = [x for x in _NUM.findall(_DURATION.sub(" ", text)) if float(x) not in allowed]
    return list(dict.fromkeys(bad))


def _llm_body(llm: LLMClient, stats: dict, lang: str) -> tuple[str | None, str]:
    facts = llm_facts(stats)
    try:
        raw = llm.chat(llm_prompt(facts, lang), temperature=0.3, max_tokens=450)
    except Exception as e:  # offline, timeout, HTTP error -> template
        return None, f"LLM unavailable ({type(e).__name__}); used the template"
    text = _clean(raw, lang)
    if not text:
        return None, "LLM returned an empty draft; used the template"
    if not _well_formed(text):
        return None, "LLM draft missed the report sections; used the template"
    bad = unsupported_numbers(text, facts)
    if bad:
        return None, f"LLM draft used numbers not in the stats ({', '.join(bad[:5])}); used the template"
    return text, ""


MAX_ISSUES = 8
# what an action may talk about, by the kind of issue it answers; an action that brings in another
# kind's topic (a crowding issue answered with "wear hard hats") is replaced by the template action
TOPIC_WORDS = {
    "hard_hat": r"安全帽|頭盔|hard ?hats?|helmets?",
    "vehicle": r"挖土機|挖掘機|怪手|堆高機|叉車|托板車|油壓車|搬運車|機器人|excavators?|forklifts?|vehicles?|"
               r"pallet|trucks?|robots?|machines?|machinery",
    "crowding": r"人數|擁擠|聚集|人太多|人流|crowd|occupancy|headcount|too many people|number of people",
    "false_alarm": r"誤報|假警報|false alarms?",
}
KIND_OF_RULE = {"no_helmet": "hard_hat", "object_near_person": "vehicle", "object_present": "vehicle",
                "person_count_over": "crowding", "person_present": "zone"}
# Mainland words OpenCC leaves alone, in Taiwan usage
TW_TERMS = {"攝像頭": "攝影機", "攝像機": "攝影機", "假警報": "誤報", "視頻": "影片", "監控": "監視"}


def action_issues(s: dict, lang: str) -> list[dict]:
    """What actually happened, most frequent first: (camera, rule) pairs with violations and the
    cameras with filtered false alarms.  An action may only address one of these."""
    t = TEXT[_lang(lang)]
    out = [{"camera": d["camera"], "issue": d["rule"], "kind": KIND_OF_RULE.get(d.get("rule_type", ""), "zone")}
           for d in s.get("by_camera_rule", [])][:MAX_ISSUES - 2]
    out += [{"camera": c, "issue": t["issue_fa"], "kind": "false_alarm"} for c in list(s["false_alarms_by_camera"])[:2]]
    return out


def off_topic(text: str, kind: str) -> str | None:
    """The first other kind of problem an action talks about (false-alarm fixes may mention hard hats,
    since those are the alerts the double-check judges)."""
    allowed = {kind} | ({"hard_hat"} if kind == "false_alarm" else set())
    return next((k for k, pat in TOPIC_WORDS.items() if k not in allowed and re.search(pat, text, re.I)), None)


def actions_prompt(s: dict, lang: str) -> tuple[list[dict], dict, list[dict]]:
    """A small, checkable job for a 1.5B model: pick 3 of the listed issues by number and write
    one action for each.  The camera and the issue come from the list, so the model cannot
    attach a problem to the wrong camera; the list carries no counts, so there is no number to
    get wrong (the peak hour is the only one given)."""
    t = TEXT[_lang(lang)]
    issues = action_issues(s, lang)
    listing = "\n".join(f"{i}: camera {x['camera']} - {x['issue']}" for i, x in enumerate(issues))
    system = (f"You help a safety officer plan tomorrow. Choose 3 different issues from the numbered list (lower "
              f"numbers happened more often) and write one concrete action for each, in {t['lang_name']}, under 25 "
              "words. The system shows the camera and the issue next to your action, so do not repeat them and "
              "do not name any other camera. No numbers except the peak hour if it helps. Reply with JSON only.")
    facts = {"peak_hour": s["peak_hour"]}
    user = f"{t['ask_actions']}\n\nPeak hour: {s['peak_hour']}\nIssues:\n{listing}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], facts, issues


def _tw(text: str) -> str:
    text = _to_traditional(text)
    for a, b in TW_TERMS.items():
        text = text.replace(a, b)
    return text


def _llm_actions(llm: LLMClient, stats: dict, lang: str) -> tuple[list[str] | None, str]:
    if not stats["violations"]:
        return None, "no violations: template actions"
    msgs, facts, issues = actions_prompt(stats, lang)
    if len(issues) < 3:
        return None, "fewer than 3 issues: template actions"
    try:
        raw = llm.chat(msgs, fmt=actions_prompt_schema(issues), temperature=0.3, max_tokens=320)
        picks = json.loads(raw)["actions"]
    except Exception as e:  # offline, timeout, not JSON
        return None, f"LLM actions unavailable ({type(e).__name__})"
    t = TEXT[_lang(lang)]
    cams = {x["camera"] for x in issues}
    out, seen, replaced = [], set(), []
    sep, colon = ("・", "：") if lang == "zh" else (" · ", ": ")
    for p in picks[:3]:
        i, text = p.get("issue"), str(p.get("action") or "").strip()
        text = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", text)
        if lang == "zh":
            text = _tw(text)
        if not isinstance(i, int) or not 0 <= i < len(issues) or i in seen or not text or len(text) > 160:
            return None, "LLM actions did not pick 3 different listed issues"
        x = issues[i]
        seen.add(i)
        # one action that strays (another camera, a number, another kind of problem) is replaced by
        # the template action for its issue; the issue the model picked stays
        why = next((w for w, bad in (
            ("another camera", any(c != x["camera"] and c.lower() in text.lower() for c in cams)),
            ("a number not in the stats", bool(unsupported_numbers(text, facts))),
            ("another kind of problem", off_topic(text, x["kind"]) is not None)) if bad), None)
        if why:
            replaced.append(f"{x['camera']}: {why}")
            text = t[f"fix_{x['kind']}"]
        out.append(f"**{x['camera']}**{sep}{x['issue']}{colon}{text}")
    if len(out) < 3:
        return None, "LLM actions did not pick 3 different listed issues"
    if len(replaced) == len(out):
        return None, "every LLM action strayed (" + "; ".join(replaced) + ")"
    return out, ("template text for " + "; ".join(replaced)) if replaced else ""


def menu_prompt(s: dict, lang: str) -> tuple[list[dict], list[dict], dict]:
    """The default: the model plans, it does not write.  It picks the 3 issues that matter most from
    what happened and, for each, one of three vetted fixes for that kind of problem; the report text
    is the vetted text.  (On 20 synthetic days, two blind judges per action rated 45 % of freely written
    actions sensible and 20 % aimed at the wrong problem; the vetted fixes: 100 % and 0 %.
    docs/results/report_actions_judged.json.)"""
    t = TEXT[_lang(lang)]
    issues = action_issues(s, lang)
    peak = s["peak_hour"] or "-"
    lines = []
    for i, x in enumerate(issues):
        opts = " | ".join(f"{j}) {o.format(peak=peak)}" for j, o in enumerate(t["menu"][x["kind"]]))
        lines.append(f"{i}: camera {x['camera']} - {x['issue']}. Fixes: {opts}")
    system = ("You help a safety officer plan tomorrow. From the numbered issues (lower numbers happened more often) "
              "choose the 3 that matter most, all different, and for each the fix number that suits it best. "
              "Reply with JSON only.")
    schema = {"type": "object", "properties": {"actions": {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "object", "properties": {
            "issue": {"type": "integer", "enum": list(range(len(issues)))},
            "fix": {"type": "integer", "enum": [0, 1, 2]}}, "required": ["issue", "fix"]}}},
        "required": ["actions"]}
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"Peak hour: {peak}\nIssues:\n" + "\n".join(lines)}]
    return msgs, issues, schema


def _llm_menu_actions(llm: LLMClient, stats: dict, lang: str) -> tuple[list[str] | None, str]:
    if not stats["violations"]:
        return None, "no violations: template actions"
    msgs, issues, schema = menu_prompt(stats, lang)
    if len(issues) < 3:
        return None, "fewer than 3 issues: template actions"
    try:
        picks = json.loads(llm.chat(msgs, fmt=schema, temperature=0.2, max_tokens=120))["actions"]
    except Exception as e:  # offline, timeout, not JSON
        return None, f"LLM plan unavailable ({type(e).__name__})"
    t = TEXT[_lang(lang)]
    sep, colon = ("・", "：") if lang == "zh" else (" · ", ": ")
    out, seen = [], set()
    for p in picks[:3]:
        i, f = p.get("issue"), p.get("fix")
        if not isinstance(i, int) or not 0 <= i < len(issues) or i in seen or f not in (0, 1, 2):
            return None, "LLM plan did not pick 3 different listed issues"
        seen.add(i)
        x = issues[i]
        text = t["menu"][x["kind"]][f].format(peak=stats["peak_hour"] or "")
        out.append(f"**{x['camera']}**{sep}{x['issue']}{colon}{text}")
    return (out, "") if len(out) == 3 else (None, "LLM plan did not pick 3 different listed issues")


def actions_prompt_schema(issues: list[dict]) -> dict:
    return {"type": "object", "properties": {"actions": {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {"type": "object", "properties": {
            "issue": {"type": "integer", "enum": list(range(len(issues)))},
            "action": {"type": "string", "maxLength": 160}}, "required": ["issue", "action"]}}},
        "required": ["actions"]}


# ------------------------------------------------------------------ entry point

def build_report(store: EventStore, llm: LLMClient | None, cameras: list[dict], lang: str = "zh",
                 hours: float = 24.0, now: float | None = None, mode: str = "hybrid",
                 actions: str = "menu") -> dict[str, Any]:
    """-> {"markdown", "stats", "generator": "llm"|"llm+template"|"template", "ms", "note"}.
    actions="menu" (default): the LLM picks the issues and one vetted fix each; "free": it writes
    the action text itself (checked: no other camera, no invented number, no other kind of problem)."""
    t_start = time.perf_counter()
    lang = _lang(lang)
    t = TEXT[lang]
    now = time.time() if now is None else float(now)
    hours = min(MAX_HOURS, max(0.25, float(hours)))
    t0 = now - hours * 3600
    stats = compute_stats(store.between(t0, now + 1, COLUMNS), cameras, t0, now)
    # mode "hybrid" (default): every number, table and the summary come from the stats and the
    # LLM only drafts the actions.  A 1.5B model writing the whole report passed the numbers
    # check yet misattributed counts and invented terms in testing, so "llm" is opt-in.
    body, note, generator = None, "", "template"
    if llm is not None and mode == "llm":
        body, note = _llm_body(llm, stats, lang)
        if body:
            generator = "llm"
    if body is None and llm is not None and mode in ("llm", "hybrid"):
        acts, note2 = (_llm_actions if actions == "free" else _llm_menu_actions)(llm, stats, lang)
        if acts:
            body = "\n".join([f"## {t['h_summary']}", _summary(stats, t), "", f"## {t['h_findings']}",
                              *_findings(stats, t), "", f"## {t['h_actions']}", *[f"- {a}" for a in acts]])
            generator = "llm+template"
            note = f"{note}; full draft rejected, LLM wrote the actions only" if note else ""
            if note2:                                  # actions it wrote that strayed got the template text
                note = f"{note}; {note2}" if note else note2
        else:
            note = f"{note}; {note2}" if note else note2
    if body is None:
        body = template_body(stats, lang)
    h = _num(stats["hours"])
    title = t["title_day"] if abs(hours - 24) < 1e-6 else t["title_hours"].format(h=h)
    model = getattr(llm, "model", "LLM")
    gen = {"llm": t["gen_llm"].format(model=model), "llm+template": t["gen_hybrid"].format(model=model)}.get(
        generator, t["gen_template"])
    footer = t["footer"].format(start=stats["period_start"], end=stats["period_end"], h=h, gen=gen)
    markdown = f"# {title}\n\n{body}\n\n---\n*{footer}*\n"
    return {"markdown": markdown, "stats": stats, "generator": generator,
            "ms": int((time.perf_counter() - t_start) * 1000), "note": note}
