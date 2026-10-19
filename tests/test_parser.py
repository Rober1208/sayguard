"""Spoken-rule parsing against the demo site's cameras and zones.

Without an LLM the deterministic extractor must handle the phrasings a site
manager actually uses (Mandarin, simplified Chinese, English, Chinese numerals
and decimals, English number words and time ranges).  With an LLM, anything the
sentence states explicitly must override the model, and any LLM failure must
fall back to the extractor instead of losing the rule.  A draft whose scope
selects no camera is flagged as a problem so it cannot be activated silently.
"""
from __future__ import annotations

import json

import pytest
import yaml

from sayguard.config import DEFAULT_CONFIG
from sayguard.rules import parser
from sayguard.rules.parser import parse


@pytest.fixture(scope="module")
def site() -> tuple[list[dict], dict[str, dict]]:
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    return cfg["cameras"], cfg["zones"]


NIGHT = {"start": "22:00", "end": "06:00"}

CASES = [
    ("B區超過3個人就提醒我",
     dict(type="person_count_over", zones=["zone_b"], cameras=["*"], threshold=3, object=None,
          time_window=None, severity="medium")),
    ("倉庫裡有人靠近堆高機3公尺內要立刻警告",
     dict(type="object_near_person", object="forklift", threshold=3, zones=["warehouse"], severity="high")),
    ("晚上10點後入口不能有人",
     dict(type="person_present", zones=["entrance"], time_window=NIGHT, min_duration_s=1.0, severity="medium")),
    ("A區沒戴安全帽超過5秒就警告",
     dict(type="no_helmet", zones=["zone_a"], min_duration_s=5, severity="high", time_window=None)),
    ("Alert me if anyone in Zone A has no hard hat for more than 3 seconds",
     dict(type="no_helmet", zones=["zone_a"], cameras=["*"], min_duration_s=3)),
    ("Warn immediately if a person is within 2 meters of a forklift in the warehouse",
     dict(type="object_near_person", object="forklift", threshold=2, zones=["warehouse"], severity="high")),
    ("如果3號攝影機看到梯子就通知我",
     dict(type="object_present", object="ladder", cameras=["cam03"], zones=[], min_duration_s=1.0)),
    ("辦公室超過五個人要提醒",
     dict(type="person_count_over", threshold=5, zones=["office"])),
    ("22點到6點倉庫禁止進入",
     dict(type="person_present", zones=["warehouse"], time_window=NIGHT)),
    ("仓库里叉车3米内有人要立刻警告",
     dict(type="object_near_person", object="forklift", threshold=3, zones=["warehouse"], severity="high")),
    # time ranges where only one end carries the am/pm (or 早上/晚上) word
    ("早上8點到晚上6點A區要戴安全帽",
     dict(type="no_helmet", zones=["zone_a"], time_window={"start": "08:00", "end": "18:00"})),
    ("Alert if anyone enters the warehouse from 8am to 5pm",
     dict(type="person_present", zones=["warehouse"], time_window={"start": "08:00", "end": "17:00"})),
    ("下午1點到5點工作坊禁止進入",
     dict(type="person_present", zones=["workshop"], time_window={"start": "13:00", "end": "17:00"})),
    ("晚上12點後辦公室不能有人",
     dict(type="person_present", zones=["office"], time_window={"start": "00:00", "end": "06:00"})),
    ("camera 8 sees more than 4 people",
     dict(type="person_count_over", threshold=4, cameras=["cam08"])),
    ("Loading Dock 有人進入就記錄",
     dict(type="person_present", cameras=["cam08"], severity="low")),
    # Chinese decimals and "...and a half" distances
    ("倉庫裡有人靠近堆高機一點五公尺內要立刻警告",
     dict(type="object_near_person", object="forklift", threshold=1.5, zones=["warehouse"], severity="high")),
    ("堆高機1點5米內有人要警告",
     dict(type="object_near_person", object="forklift", threshold=1.5)),
    ("堆高機兩公尺半內有人",
     dict(type="object_near_person", object="forklift", threshold=2.5)),
    ("仓库叉车2米半内有人要立刻警告",
     dict(type="object_near_person", object="forklift", threshold=2.5, zones=["warehouse"])),
    ("晚上10點半後入口不能有人",            # 半 after a time is still half past
     dict(type="person_present", zones=["entrance"], time_window={"start": "22:30", "end": "06:00"})),
    # English time ranges
    ("Alert if anyone enters the warehouse between 10pm and 6am",
     dict(type="person_present", zones=["warehouse"], time_window=NIGHT)),
    ("Alert if anyone enters the warehouse between 22:00 and 06:00",
     dict(type="person_present", zones=["warehouse"], time_window=NIGHT)),
    ("Alert if anyone enters the warehouse from 10 p.m. to 6 a.m.",
     dict(type="person_present", zones=["warehouse"], time_window=NIGHT)),
    ("Log anyone in the office after 9 p.m.",
     dict(type="person_present", zones=["office"], time_window={"start": "21:00", "end": "06:00"})),
    # English number words
    ("Warn me if there are more than three people in Zone B",
     dict(type="person_count_over", threshold=3, zones=["zone_b"])),
    ("Warn immediately if anyone is within two metres of a forklift",
     dict(type="object_near_person", object="forklift", threshold=2, cameras=["*"], severity="high")),
    ("Alert if anyone in Zone A has no hard hat for five seconds",
     dict(type="no_helmet", zones=["zone_a"], min_duration_s=5)),
    # durations in minutes
    ("A區沒戴安全帽超過1分鐘就警告",
     dict(type="no_helmet", zones=["zone_a"], min_duration_s=60)),
    ("A區沒戴安全帽超過一分鐘就警告",
     dict(type="no_helmet", zones=["zone_a"], min_duration_s=60)),
    ("Alert if anyone in Zone A has no hard hat for 2 minutes",
     dict(type="no_helmet", zones=["zone_a"], min_duration_s=120)),
    ("Alert if anyone stays in the workshop for two minutes",
     dict(type="person_present", zones=["workshop"], min_duration_s=120)),
]


def assert_rule(rule, expected: dict) -> None:
    got = rule.model_dump()
    for key, want in expected.items():
        assert got[key] == want, f"{key}: got {got[key]!r}, want {want!r}"


@pytest.mark.parametrize("text, expected", CASES, ids=[c[0] for c in CASES])
def test_extractor_only(site, text, expected):
    cams, zones = site
    res = parse(text, cams, zones, llm=None)
    assert res.parser == "rules", res.problems
    assert res.rule is not None and res.problems == []
    assert res.rule.source_text == text and res.rule.name and res.rule.message
    assert_rule(res.rule, expected)


@pytest.mark.parametrize("text", ["今天天氣很好", "B區超過兩三個人就提醒我", ""])
def test_unparseable_text_fails_cleanly(site, text):
    res = parse(text, *site, llm=None)
    assert res.rule is None and res.parser == "failed" and res.problems


def test_chinese_numerals_are_normalised():
    assert parser._norm("超過十二個人") == "超過12個人"
    assert parser._norm("二十五秒") == "25秒"
    assert parser._norm("兩三個人") == "兩三個人"     # a range, not a number: left alone


@pytest.mark.parametrize("text, want", [
    ("一點五公尺", "1.5公尺"),
    ("1點5米", "1.5米"),
    ("兩點五米內", "2.5米內"),
    ("一點二五公尺", "1.25公尺"),
    ("十二點五公尺", "12.5公尺"),
    ("1点5 m", "1.5 m"),
    ("兩公尺半", "2.5公尺"),
    ("2公尺半", "2.5公尺"),
    ("一米半", "1.5米"),
    ("半公尺內", "0.5公尺內"),
    # times are not distances: 點 stays a time separator, 半 stays "half past"
    ("10點半", "10點半"),
    ("十點半", "10點半"),
    ("10點5分", "10點5分"),
    ("晚上十點五分後", "晚上10點5分後"),
])
def test_chinese_decimal_distances(text, want):
    assert parser._norm(text) == want


@pytest.mark.parametrize("text, want", [
    ("more than three people", "more than 3 people"),
    ("over Twelve workers", "over 12 workers"),
    ("within two metres", "within 2 metres"),
    ("within ten m of", "within 10 m of"),
    ("for five seconds", "for 5 seconds"),
    ("for twenty minutes", "for 20 minutes"),
    # number words not followed by a unit are left alone
    ("no one enters", "no one enters"),
    ("anyone or someone near one of the trucks", "anyone or someone near one of the trucks"),
    ("often tense", "often tense"),
])
def test_english_number_words_before_units(text, want):
    assert parser._norm(text) == want


@pytest.mark.parametrize("text, want", [
    ("between 10pm and 6am", NIGHT),
    ("between 10 PM and 6 AM", NIGHT),
    ("between 22:00 and 06:00", NIGHT),
    ("from 10 p.m. to 6 a.m.", NIGHT),
    ("from 10 p.m to 6 a.m", NIGHT),
    ("between 9 P.M. and 5 A.M.", {"start": "21:00", "end": "05:00"}),
    ("from 8am to 5pm", {"start": "08:00", "end": "17:00"}),
    ("between 1 and 5pm", {"start": "13:00", "end": "17:00"}),
    ("10pm到6am", NIGHT),
    ("after 10 p.m.", NIGHT),
    ("between 3 and 5 people", None),    # no clock word: a count, not a time
    ("10pm and 6am", None),              # "and" separates a range only after "between"
    ("3 amps", None),
])
def test_english_time_ranges(text, want):
    assert parser._time_window(parser._norm(text)) == want


def test_llm_decimal_distance_is_not_overridden(site):
    # the sentence and the LLM agree on 1.5 m; the extractor used to read 5 and override it
    text = "倉庫裡有人靠近堆高機一點五公尺內要立刻警告"
    llm = FakeLLM({"name": "堆高機距離", "type": "object_near_person", "cameras": ["*"], "zones": ["warehouse"],
                   "object": "forklift", "threshold": 1.5, "time_window": None, "min_duration_s": 1,
                   "severity": "high", "message": "有人在堆高機 1.5 公尺內"})
    res = parse(text, *site, llm=llm)
    assert res.parser == "llm+check" and res.notes == [] and res.problems == []
    assert res.rule.threshold == 1.5


# ---- scope that matches no camera ----------------------------------------------------
def test_scope_matching_no_camera_is_a_problem(site):
    res = parse("cam01 B區沒戴安全帽要警告", *site, llm=None)     # cam01 is at the entrance, not zone B
    assert res.rule is not None and res.parser == "rules"
    assert_rule(res.rule, dict(type="no_helmet", cameras=["cam01"], zones=["zone_b"]))
    assert res.problems == ["no camera matches this scope"]


def test_llm_scope_matching_no_camera_is_a_problem(site):
    llm = FakeLLM({"name": "Office crowding", "type": "person_count_over", "cameras": ["Site Office"],
                   "zones": ["warehouse"], "threshold": 5, "severity": "medium", "message": "Too many people"})
    res = parse("Tell me when it gets crowded", *site, llm=llm)
    assert res.parser == "llm+check"
    assert_rule(res.rule, dict(cameras=["cam02"], zones=["warehouse"]))
    assert res.problems == ["no camera matches this scope"]


def test_scope_check_needs_a_camera_list(site):
    _, zones = site
    res = parse("B區超過3個人就提醒我", [], zones, llm=None)      # no cameras configured: nothing to check against
    assert res.rule is not None and res.problems == []
    rule = res.rule.model_copy(update={"zones": ["zone_b"], "zones_exclude": ["zone_b"]})
    assert parser.scope_problems(rule, site[0]) == ["no camera matches this scope"]


# ---- LLM merge path ----------------------------------------------------------------
class FakeLLM:
    def __init__(self, reply: dict | str | Exception):
        self.reply = reply
        self.calls: list[dict] = []

    def chat(self, messages, fmt=None, temperature=0.2, max_tokens=512) -> str:
        self.calls.append({"messages": messages, "fmt": fmt, "temperature": temperature})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False)


def test_explicit_evidence_overrides_llm(site):
    text = "倉庫裡有人靠近堆高機3公尺內要立刻警告"
    llm = FakeLLM({"name": "堆高機警示", "type": "object_present", "cameras": ["*"], "zones": ["倉庫"],
                   "object": "堆高機", "threshold": 5, "time_window": None, "min_duration_s": 1,
                   "severity": "medium", "message": "小心堆高機"})
    res = parse(text, *site, llm=llm)
    assert res.parser == "llm+check" and res.problems == []
    assert_rule(res.rule, dict(type="object_near_person", object="forklift", threshold=3, zones=["warehouse"],
                               severity="high", min_duration_s=1, name="堆高機警示", message="有人在堆高機 3 公尺內"))
    for key in ("type", "threshold", "severity"):
        assert any(n.startswith(f"{key}:") for n in res.notes), res.notes
    assert not any(n.startswith("object:") for n in res.notes)   # LLM agreed after zh->en mapping
    call = llm.calls[0]
    assert call["temperature"] == 0.0 and call["fmt"]["type"] == "object"
    assert call["messages"][-1] == {"role": "user", "content": text}


def test_llm_fills_what_the_sentence_leaves_open(site):
    llm = FakeLLM({"name": "Dock crowding", "type": "person_count_over", "cameras": ["Loading Dock"],
                   "zones": [], "threshold": 5, "severity": "medium", "message": "Dock is crowded"})
    res = parse("Tell me when the Loading Dock gets crowded", *site, llm=llm)
    assert res.parser == "llm+check" and res.notes == [] and res.problems == []
    assert_rule(res.rule, dict(type="person_count_over", threshold=5, cameras=["cam08"], min_duration_s=2.0))


@pytest.mark.parametrize("reply", [ConnectionError("ollama down"), TimeoutError(), "not json {", '["a list"]'])
def test_llm_failure_falls_back_to_extractor(site, reply):
    res = parse("B區超過3個人就提醒我", *site, llm=FakeLLM(reply))
    assert res.parser == "rules"
    assert len(res.notes) == 1 and res.notes[0].startswith("LLM unavailable (")
    assert_rule(res.rule, dict(type="person_count_over", threshold=3, zones=["zone_b"]))


def test_invalid_llm_rule_falls_back_to_extractor(site):
    llm = FakeLLM({"name": "B區人數", "type": "person_count_over", "zones": ["zone_b"], "threshold": 3,
                   "severity": "critical", "message": "B區人數超過 3 人"})   # not low|medium|high
    res = parse("B區超過3個人就提醒我", *site, llm=llm)
    assert res.parser == "rules" and res.problems == []
    assert any("not a valid rule" in n for n in res.notes)
    assert_rule(res.rule, dict(type="person_count_over", threshold=3, zones=["zone_b"], severity="medium"))


def test_llm_and_extractor_both_clueless(site):
    res = parse("今天天氣很好", *site, llm=FakeLLM({"name": "x", "type": "weather", "message": "x"}))
    assert res.rule is None and res.parser == "failed"


class _FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def chat(self, *a, **k):
        return self.reply


def test_llm_inventions_are_removed_and_message_matches_rule():
    import json
    from sayguard.rules import parser as P
    cams = [{"id": "cam03", "name": "Scaffold North", "zone": "zone_a"}]
    zones = {"zone_a": {"en": "Zone A", "zh": "A區"}}
    reply = json.dumps({"name": "Zone A no helmet", "type": "no_helmet", "cameras": ["*"], "zones": ["zone_a"],
                        "object": "forklift", "threshold": 3, "time_window": {"start": "00:00", "end": "23:59"},
                        "min_duration_s": 3, "severity": "high", "message": "未戴安全帽"})
    res = P.parse("Alert me if anyone in Zone A has no hard hat for more than 3 seconds", cams, zones, _FakeLLM(reply))
    r = res.rule
    assert r.type == "no_helmet" and r.threshold is None and r.object is None and r.time_window is None
    assert r.message == "No hard hat" and r.name == "Zone A no helmet"


def test_llm_simplified_name_converted_and_wrong_message_replaced():
    import json
    from sayguard.rules import parser as P
    cams = [{"id": "cam07", "name": "Warehouse Aisle", "zone": "warehouse"}]
    zones = {"warehouse": {"en": "Warehouse", "zh": "倉庫"}}
    reply = json.dumps({"name": "仓库叉车距离", "type": "object_near_person", "cameras": ["*"], "zones": ["warehouse"],
                        "object": "forklift", "threshold": 1.5, "time_window": None, "min_duration_s": 1,
                        "severity": "medium", "message": "人數過多"})
    r = P.parse("倉庫有人靠近堆高機一點五公尺內就警告", cams, zones, _FakeLLM(reply)).rule
    assert r.threshold == 1.5 and r.object == "forklift"
    assert "堆高機" in r.message and "1.5" in r.message
    assert "仓" not in r.name and "倉" in r.name


# ---- generic object words and zones this site does not have ---------------------------
def test_generic_robot_word(site):
    res = parse("倉庫有機器人就提醒我", *site, llm=None)
    assert_rule(res.rule, dict(type="object_present", object="robot", zones=["warehouse"]))
    assert res.rule.message == "偵測到機器人" and res.problems == []
    # the specific robot classes still win over the generic word
    assert parse("倉庫出現人形機器人就提醒我", *site, llm=None).rule.object == "humanoid_robot"


@pytest.mark.parametrize("text, missing", [
    ("C區有人就提醒我", "C區"),
    ("9區超過3個人要警告", "9區"),
    ("Alert me if anyone enters zone D", "zone D"),
])
def test_unknown_zone_is_a_problem(site, text, missing):
    res = parse(text, *site, llm=None)
    assert res.rule is not None and len(res.problems) == 1 and missing in res.problems[0]


@pytest.mark.parametrize("text", [
    "B區有人就提醒我", "這一區有人就提醒我", "A區沒戴安全帽要警告", "整個區域都不能有人",
    # quantifiers, not zone labels
    "有人在任一區停留超過10秒", "任何一區超過5個人就通知我", "某一區有堆高機就警告", "其中一區有人就提醒我",
    "單一區超過3個人要警告",
    # a quantity or an article after a generic "area" / "zone"
    "Notify me if anyone lingers in the dock area 30 seconds or longer",
    "Alert if a worker is in the danger zone 2 m from a forklift",
    "Warn me if anyone enters the area a forklift is working in",
    # a configured zone written with a space or full-width letters
    "A 區沒戴安全帽要警告", "Ｂ區超過3個人就提醒我",
])
def test_known_zones_and_plain_words_are_not_flagged(site, text):
    assert parser.unknown_zones(text, *site) == []


def test_full_width_and_spaced_zone_names_match_the_configured_zone(site):
    assert parse("Ｂ區超過3個人就提醒我", *site, llm=None).rule.zones == ["zone_b"]
    assert parse("A 區沒戴安全帽要警告", *site, llm=None).rule.zones == ["zone_a"]
    assert "Ｃ區" not in parser.unknown_zones("Ｃ區有人就提醒我", *site)        # reported as the ASCII C區
    assert parser.unknown_zones("Ｃ區有人就提醒我", *site) == ["C區"]


# ---- camera names inside other camera names, and LLM titles that mangle them --------------
@pytest.fixture(scope="module")
def warehouse() -> tuple[list[dict], dict[str, dict]]:
    cfg = yaml.safe_load((DEFAULT_CONFIG.parent / "warehouse.yaml").read_text(encoding="utf-8"))
    return [dict(c, zones=list(cfg["zones"])) for c in cfg["cameras"]], cfg["zones"]


def test_longest_camera_name_wins(warehouse):
    # cam04 is "North Aisle", cam05 "North Aisle East": the sentence names cam05 only
    res = parse("North Aisle East出現人形機器人就提醒我", *warehouse, llm=None)
    assert res.rule.cameras == ["cam05"] and res.rule.name == "North Aisle East 偵測到人形機器人"
    assert parse("North Aisle出現人形機器人就提醒我", *warehouse, llm=None).rule.cameras == ["cam04"]


def test_llm_title_that_mangles_a_name_is_replaced(warehouse):
    llm = FakeLLM({"name": "北 aisle東出現人形機器人", "type": "object_present", "cameras": ["cam04", "cam05"],
                   "zones": [], "object": "humanoid_robot", "severity": "medium", "message": "x"})
    res = parse("North Aisle East出現人形機器人就提醒我", *warehouse, llm=llm)
    assert res.rule.cameras == ["cam05"] and res.rule.name == "North Aisle East 偵測到人形機器人"
