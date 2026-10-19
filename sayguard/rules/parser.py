"""Natural language (Mandarin or English) -> Rule.

The small on-device LLM (Qwen2.5-1.5B on Ollama / hailo-ollama) proposes a rule as
JSON; a deterministic extractor then checks it.  Anything the sentence states
explicitly (numbers, distances, times, the object word, the rule type keywords)
overrides the LLM when the two disagree, and the result is shown to the user for
confirmation before it is activated.  Without an LLM the extractor alone is used.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .model import RULE_TYPES, Rule

# zh/simplified/en object words -> English class name for YOLO-World
OBJECTS: list[tuple[str, str, str]] = [
    # (regex, english class, zh display); first match wins, so longer names come first
    # site-detector classes (site mode); before "pallet" and "truck"
    (r"托板車|托板车|拖板車|拖板车|油壓車|油压车|pallet ?(truck|jack)s?", "pallet_truck", "托板車"),
    (r"人形機器人|人形机器人|humanoid( ?robot)?s?", "humanoid_robot", "人形機器人"),
    (r"移動機器人|移动机器人|自主移動機器人|搬運機器人|搬运机器人|自走車|自走车|(?<![a-z])amrs?(?![a-z])|mobile ?robots?",
     "mobile_robot", "移動機器人"),
    (r"搬運車|搬运车|無人搬運車|无人搬运车|(?<![a-z])agvs?(?![a-z])|transporters?", "transporter", "搬運車"),
    # any robot at all: the engine matches every robot class (model.CLASS_GROUPS)
    (r"機器人|机器人|(?<![a-z])robots?(?![a-z])", "robot", "機器人"),
    (r"堆高機|堆高机|叉車|叉车|forklift", "forklift", "堆高機"),
    (r"手推車|手推车|推車|推车|hand ?cart|trolley", "hand cart", "推車"),
    (r"梯子|爬梯|ladder", "ladder", "梯子"),
    (r"滅火器|灭火器|fire extinguisher", "fire extinguisher", "滅火器"),
    (r"明火|火焰|火災|火灾|(?<![滅灭])火(?!車)|fire|flame", "fire", "火焰"),
    (r"煙霧|烟雾|濃煙|浓烟|smoke", "smoke", "煙霧"),
    (r"紙箱|纸箱|箱子|cardboard box|box", "cardboard box", "箱子"),
    (r"棧板|栈板|托盤|托盘|pallet", "pallet", "棧板"),
    (r"挖土機|挖土机|挖掘機|挖掘机|怪手|excavator", "excavator", "挖土機"),
    (r"吊車|吊车|起重機|起重机|crane", "crane", "吊車"),
    (r"卡車|卡车|貨車|货车|truck", "truck", "卡車"),
    (r"機車|机车|摩托車|摩托车|motorcycle|scooter", "motorcycle", "機車"),
    (r"腳踏車|脚踏车|自行車|自行车|bicycle|bike", "bicycle", "腳踏車"),
    (r"汽車|汽车|轎車|轿车|(?<!\w)car(?!\w)", "car", "汽車"),
    (r"反光背心|安全背心|safety vest|vest", "safety vest", "反光背心"),
    (r"手機|手机|cell ?phone|mobile phone|phone", "cell phone", "手機"),
    (r"背包|backpack", "backpack", "背包"),
    (r"香菸|香烟|菸|烟蒂|cigarette", "cigarette", "香菸"),
    (r"刀子|刀具|knife", "knife", "刀子"),
    (r"水桶|bucket", "bucket", "水桶"),
    (r"電鑽|电钻|drill", "power drill", "電鑽"),
    (r"狗|dog", "dog", "狗"),
]
ZH_OBJ = {en: zh for _, en, zh in OBJECTS}

CN_NUM = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "半": 0.5}

HELMET = r"安全帽|工程帽|工地帽|頭盔|头盔|帽子|hard ?hats?|helmets?"
PERSON_COUNT = r"(?:超過|超过|多於|多于|大於|大于|more than|over|exceeds?)\s*(\d+(?:\.\d+)?)\s*(?:個人|个人|位|名|人|people|persons?|workers?)"
# "N or more" / "at least N" / N人以上 -> more than N-1
PERSON_ATLEAST = (r"(?:(\d+)\s*(?:or more|and more|\+)\s*(?:people|persons?|workers?)|at least\s*(\d+)\s*(?:people|persons?|workers?)|"
                  r"(\d+)\s*(?:個人|个人|位|名|人)\s*(?:以上|或以上))")
DISTANCE = r"(\d+(?:\.\d+)?)\s*(?:公尺|米|m\b|meters?|metres?)"
NEAR = (r"內|内|以內|以内|之內|之内|範圍|范围|靠近|接近|附近|距離|距离|保持|不到|小於|小于|少於|少于|"
        r"within|near|close to|closer than|less than|away from|distance")
PRESENT_PERSON = r"有人|人員|人员|任何人|閒雜人等|闲杂人等|anyone|someone|any ?body|people|persons?|workers?"
ENTER = (r"進入|进入|闖入|闯入|出現|出现|逗留|停留|待|留在|不能有人|不准有人|禁止進入|禁止进入|禁止|不得進入|"
         r"有人(?=就|便|時|时|的話|的话)|enter|inside|present|in the|on the|at the|stays?|remains?|linger|loiter")
# prohibitions that imply people even when the sentence names nobody ("22點到6點倉庫禁止進入")
NO_ENTRY = r"禁止進入|禁止进入|不得進入|不得进入|不准進入|不准进入|禁止闖入|禁止闯入|no entry|keep out"
APPEAR = r"出現|出现|看到|偵測到|侦测到|發現|发现|有|appears?|detected|seen|visible|present"
HIGH = r"立刻|立即|馬上|马上|緊急|紧急|危險|危险|嚴重|严重|immediately|urgent|danger"
LOW = r"記錄|记录|留意|注意一下|log only|just log"


CN_INT = r"[零〇一二兩两三四五六七八九十]{1,3}"
# a length unit follows; used to tell 1點5公尺 (1.5 m) from 10點5分 (a time)
LENGTH_AHEAD = r"(?=\s*(?:公尺|米|m\b|meters?|metres?))"
EN_NUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
# English number words are converted only before a count/length/time unit, like the Chinese ones,
# so "no one" or "one of the trucks" are left alone
EN_UNIT_AHEAD = (r"(?=[\s-]*(?:people|persons?|workers?|metres?|meters?|m\b|seconds?|secs?\b|"
                 r"minutes?|mins?\b))")


def _cn_int(s: str) -> int | None:
    """一..九十九 -> int; None for 兩三 ("two or three"), which has no single value."""
    try:
        if s == "十":
            return 10
        if s.startswith("十"):
            return 10 + CN_NUM[s[1]]
        if s.endswith("十"):
            return CN_NUM[s[0]] * 10
        if "十" in s:
            return CN_NUM[s[0]] * 10 + CN_NUM[s[2]]
        return CN_NUM[s]
    except (KeyError, IndexError):
        return None


def _norm(text: str) -> str:
    # NFKC: full-width letters and digits from a Taiwanese IME (Ｂ區, ３公尺) read as ASCII
    t = unicodedata.normalize("NFKC", text).strip()
    t = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[區区])", "", t)          # "A 區" is the zone A區
    t = t.translate(str.maketrans("０１２３４５６７８９：．", "0123456789:."))

    # decimals before a length unit: 一點五公尺 / 1點5米 / 一點二五公尺 -> 1.5 / 1.5 / 1.25.
    # Without this 點 is read as a classifier and DISTANCE picks the 5 of 1點5.
    def dec(m: re.Match) -> str:
        whole = int(m[1]) if m[1].isascii() else _cn_int(m[1])
        frac = m[2] if m[2].isascii() else "".join(str(CN_NUM[c]) for c in m[2])
        return m[0] if whole is None else f"{whole}.{frac}"
    t = re.sub(rf"([0-9]+|{CN_INT})[點点]([0-9]+|[零〇一二三四五六七八九]+){LENGTH_AHEAD}", dec, t, flags=re.I)

    # Chinese numerals directly before a unit/classifier -> digits (covers 一..九十九)
    def cn(m: re.Match) -> str:
        v = _cn_int(m.group(1))
        return m.group(0) if v is None else str(v) + m.group(2)
    t = re.sub(rf"({CN_INT})(個|个|位|名|人|公尺|米|點|点|秒|分|號|号|支|台)", cn, t)

    # "and a half" after a length: 兩公尺半 / 2米半 -> 2.5公尺 / 2.5米; a bare 半公尺 is 0.5 m.
    # A unit is required before 半, so times such as 10點半 are untouched.
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(公尺|米)半", lambda m: f"{float(m[1]) + 0.5}{m[2]}", t)
    t = re.sub(r"(?<![0-9.])半(?=\s*(?:公尺|米))", "0.5", t)

    t = re.sub(rf"\b({'|'.join(EN_NUM)})\b{EN_UNIT_AHEAD}", lambda m: str(EN_NUM[m[1].lower()]), t, flags=re.I)
    return t


def _has(pat: str, text: str) -> bool:
    return re.search(pat, text, re.I) is not None


def _is_zh(text: str) -> bool:
    return re.search(r"[一-鿿]", text) is not None


@dataclass
class Evidence:
    """What the sentence says explicitly; used to check (and override) the LLM."""
    type: str | None = None
    object: str | None = None
    threshold: float | None = None
    time_window: dict | None = None
    min_duration_s: float | None = None
    severity: str | None = None
    cameras: list[str] = field(default_factory=list)
    zones: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _t2s():
    try:
        from opencc import OpenCC
        return OpenCC("t2s")
    except Exception:  # optional: without it only 區/区 is folded
        return None


@lru_cache(maxsize=1)
def _s2tw():
    try:
        from opencc import OpenCC
        return OpenCC("s2twp")
    except Exception:
        return None


def _to_traditional(s: str) -> str:
    """The 1.5B LLM sometimes answers in simplified characters; the UI is Taiwanese Mandarin."""
    cc = _s2tw()
    return cc.convert(s) if cc else s


@lru_cache(maxsize=512)
def _fold(s: str) -> str:
    """Traditional -> simplified, so "仓库" in a typed rule matches the configured zone "倉庫"."""
    cc = _t2s()
    return cc.convert(s) if cc else s


EN_WORD = {i: w for w, i in EN_NUM.items()}
CN_DIGIT = "零一二三四五六七八九十"


def _name_pattern(name: str) -> str:
    """Regex for a zone or camera display name as people say it: the name normalised like the
    sentence (七號走道 -> 7號走道), a number as digits, a Chinese numeral or an English word
    ("aisle seven"), and a trailing 區 optional when at least two characters remain (碼頭區 -> 碼頭)."""
    n = _fold(_norm(name))
    optional_qu = len(n) >= 3 and n[-1] in "區区"
    if optional_qu:
        n = n[:-1]
    out, i = "", 0
    for m in re.finditer(r"\d+", n):
        out += re.escape(n[i:m.start()])
        v = int(m.group())
        alts = [m.group()] + ([CN_DIGIT[v]] if v <= 10 else []) + ([EN_WORD[v]] if v in EN_WORD else [])
        out += "(?:" + "|".join(alts) + ")"
        i = m.end()
    out += re.escape(n[i:])
    out = out.replace(r"\ ", r"[\s-]*").replace("區", "[區区]")
    return out + ("(?:[區区])?" if optional_qu else "")


def _match_scope(text: str, cameras: list[dict], zones: dict[str, dict]) -> tuple[list[str], list[str]]:
    zs: list[str] = []
    ftext = _fold(text)
    for zid, names in zones.items():
        cands = {zid, names.get("en", ""), names.get("zh", "")}
        for c in cands:
            if not c:
                continue
            if re.search(_name_pattern(c), ftext, re.I):
                zs.append(zid)
                break
    # camera numbers are read with the zone names blanked out: 7號走道 names aisle 7, not camera 7
    for zid in zs:
        for c in (zones[zid].get("en", ""), zones[zid].get("zh", "")):
            if c:
                ftext = re.sub(_name_pattern(c), " ", ftext, flags=re.I)
    cams: list[str] = []
    ids = {c["id"] for c in cameras}
    for m in re.finditer(r"(?:cam(?:era)?\s*0?(\d{1,2}))|(?:(\d{1,2})\s*[號号]\s*(?:攝影機|摄影机|鏡頭|镜头|相機|相机)?)|"
                         r"(?:第\s*(\d{1,2})\s*(?:支|台|個|个)?\s*(?:攝影機|摄影机|鏡頭|镜头))", ftext, re.I):
        n = next(g for g in m.groups() if g)
        cid = f"cam{int(n):02d}"
        if cid in ids:
            cams.append(cid)
    for c in cameras:  # camera display names ("Loading Dock")
        if c["name"] and re.search(re.escape(c["name"]), text, re.I) and c["id"] not in cams:
            cams.append(c["id"])
    return cams, zs


PM = r"晚上|夜|下午|傍晚|pm|p\.m\."
# am/pm as typed or transcribed: pm, p.m., p.m, p m (but not the start of "amp" or "pmt")
AMPM = r"[ap]\.?\s?m\.?(?![a-z])"


def _ampm(s: str | None) -> str:
    """'p.m.' / 'P M' -> 'pm', so to24 sees one spelling."""
    return re.sub(r"[.\s]", "", s or "").lower()


def _time_window(text: str) -> dict | None:
    def to24(h: int, ctx: str) -> int:
        if re.search(r"晚上|夜", ctx) and h == 12:   # 晚上12點 is midnight, not noon
            return 0
        if re.search(PM, ctx, re.I) and h < 12:
            return h + 12
        if re.search(r"凌晨|早上|上午|am|a\.m\.", ctx, re.I) and h == 12:
            return 0
        return h
    tod = r"(?:晚上|夜間|夜间|下午|早上|上午|凌晨)?"
    # "and" separates the two ends only after "between" ("between 10pm and 6am", "between 22:00 and 06:00")
    m = re.search(rf"(?P<btw>\bbetween\s+)?(?P<p1>{tod})\s*(?P<h1>\d{{1,2}})(?:[:：點点時时](?P<m1>\d{{2}}|半)?)?\s*"
                  rf"(?P<a1>{AMPM})?\s*(?:到|至|~|-|–|until|to|(?(btw)and|(?!)))\s*"
                  rf"(?P<p2>{tod})\s*(?P<h2>\d{{1,2}})(?:[:：點点時时](?P<m2>\d{{2}}|半)?)?\s*(?P<a2>{AMPM})?",
                  text, re.I)
    if m and (m["a1"] or m["a2"] or re.search(r"[點点時时:]", m.group(0))):
        # each end uses its own am/pm word; a bare hour borrows the other end's afternoon/evening
        # word only when that keeps the range in order ("1 to 5pm", "下午1點到5點", not "10 to 6pm")
        h1, h2 = int(m["h1"]), int(m["h2"])
        c1, c2 = m["p1"] + _ampm(m["a1"]), m["p2"] + _ampm(m["a2"])
        if h1 < h2 and not c1 and re.search(PM, c2, re.I):
            c1 = c2
        if h1 < h2 and not c2 and re.search(PM, c1, re.I):
            c2 = c1
        h1, h2 = to24(h1, c1), to24(h2, c2)
        mm1 = 30 if m["m1"] == "半" else int(m["m1"] or 0)
        mm2 = 30 if m["m2"] == "半" else int(m["m2"] or 0)
        return {"start": f"{h1 % 24:02d}:{mm1:02d}", "end": f"{h2 % 24:02d}:{mm2:02d}"}
    m = re.search(r"((?:晚上|夜間|夜间|下午|早上|上午|凌晨)?)\s*(\d{1,2})\s*(?:[:：點点時时](\d{2}|半)?)\s*(?:以後|之後|以后|之后|後|后)", text)
    if not m:
        m2 = re.search(rf"after\s+(\d{{1,2}})(?::(\d{{2}}))?\s*({AMPM})?", text, re.I)
        if m2:
            h = to24(int(m2.group(1)), _ampm(m2.group(3)))
            return {"start": f"{h % 24:02d}:{int(m2.group(2) or 0):02d}", "end": "06:00"}
    if m:
        h = to24(int(m.group(2)), m.group(1))
        mm = 30 if m.group(3) == "半" else int(m.group(3) or 0)
        return {"start": f"{h % 24:02d}:{mm:02d}", "end": "06:00"}
    if _has(r"下班後|下班后|after hours|after work", text):
        return {"start": "18:00", "end": "07:00"}
    if _has(r"半夜|深夜|夜間|夜间|overnight|at night", text):
        return {"start": "22:00", "end": "06:00"}
    return None


def _without_scope_names(text: str, cameras: list[dict], zones: dict[str, dict]) -> str:
    """The text with zone and camera display names blanked out: "the forklift aisle" names a
    place, not an object, so it must not make the rule a forklift rule."""
    out = _fold(text)
    names = [n for z in zones.values() for n in (z.get("en", ""), z.get("zh", ""))] + [c.get("name", "") for c in cameras]
    for n in sorted((n for n in names if n), key=len, reverse=True):
        out = re.sub(_name_pattern(n), " ", out, flags=re.I)
    return out


# "B區", "3區", "zone B", "area 2": a zone label left over once the configured zone names are blanked
# out names a zone this site does not have.  Not labels: 這一區 / 任一區 / 其中一區 (quantifiers), and
# "the dock area 30 seconds" / "the danger zone 2 m" / "the area a forklift..." (a quantity or an article)
ZONE_LIKE_ZH = re.compile(r"(?<![A-Za-z0-9这這那每哪整同任何另某其中單单唯])"
                          r"([A-Za-z]|\d{1,2}|[一二三四五六七八九十]{1,3})[区區](?!域)")
ZONE_LIKE_EN = re.compile(r"(?<![A-Za-z])(?:[Zz]one|[Aa]rea|ZONE|AREA)\s+([A-Z]|[b-hj-z]|\d{1,2})(?![A-Za-z0-9])"
                          r"(?!\s*(?:m\b|metres?|meters?|seconds?|secs?\b|minutes?|mins?\b|hours?|people|persons?|workers?|%))")


def unknown_zones(text: str, cameras: list[dict], zones: dict[str, dict]) -> list[str]:
    """Zone labels the sentence names that match no configured zone, as the user wrote them."""
    rest = _without_scope_names(_norm(text), cameras, zones)
    found = [(m.start(), _to_traditional(m.group(0))) for m in ZONE_LIKE_ZH.finditer(rest)]
    found += [(m.start(), m.group(0).strip()) for m in ZONE_LIKE_EN.finditer(rest)]
    out: list[str] = []
    for _, name in sorted(found):
        if name not in out:
            out.append(name)
    return out


def unknown_zone_problems(text: str, cameras: list[dict], zones: dict[str, dict]) -> list[str]:
    """A rule must not quietly drop a place the officer named and watch the whole site instead."""
    missing = unknown_zones(text, cameras, zones)
    if not missing:
        return []
    zh = _is_zh(text)
    names = [z.get("zh" if zh else "en") or zid for zid, z in zones.items()]
    if zh:
        return [f"這個場地沒有「{'、'.join(missing)}」；可用的區域：{'、'.join(names)}"]
    return [f"This site has no zone called {', '.join(repr(m) for m in missing)}; zones here: {', '.join(names)}"]


def extract(text: str, cameras: list[dict], zones: dict[str, dict]) -> Evidence:
    t = _norm(text)
    ev = Evidence()
    ev.cameras, ev.zones = _match_scope(t, cameras, zones)
    t_obj = _without_scope_names(t, cameras, zones)
    obj = next(((en, pat) for pat, en, _ in OBJECTS if _has(pat, t_obj)), None)
    dist = re.search(DISTANCE, t, re.I)
    count = re.search(PERSON_COUNT, t, re.I)
    atleast = None if count else re.search(PERSON_ATLEAST, t, re.I)
    if obj and dist and _has(NEAR, t):
        ev.type, ev.object, ev.threshold = "object_near_person", obj[0], float(dist.group(1))
    elif count:
        ev.type, ev.threshold = "person_count_over", float(count.group(1))
    elif atleast:
        ev.type, ev.threshold = "person_count_over", float(next(g for g in atleast.groups() if g)) - 1
    elif _has(HELMET, t):
        ev.type = "no_helmet"
    elif obj and _has(APPEAR, t):
        ev.type, ev.object = "object_present", obj[0]
    elif _has(NO_ENTRY, t) or (_has(PRESENT_PERSON, t) and _has(ENTER, t)):
        ev.type = "person_present"
    elif obj:
        ev.type, ev.object = "object_present", obj[0]
    if obj and ev.type in ("object_present", "object_near_person"):
        ev.object = obj[0]
    ev.time_window = _time_window(t)
    d = re.search(r"(?:超過|超过|持續|持续|連續|连续|for|over|more than|longer than)\s*(\d+(?:\.\d+)?)\s*"
                  r"(?:(?P<min>分鐘|分钟|minutes?|mins?\b)|秒|s\b|sec|seconds?)", t, re.I)
    if d is None:
        d = re.search(r"(?:at least\s*)?(\d+(?:\.\d+)?)\s*(?:(?P<min>分鐘|分钟|minutes?|mins?\b)|秒|seconds?)\s*(?:以上|或以上|or (?:more|longer))",
                      t, re.I) or re.search(r"at least\s*(\d+(?:\.\d+)?)\s*(?:(?P<min>minutes?|mins?\b)|seconds?|secs?\b)", t, re.I)
    if d:
        ev.min_duration_s = float(d.group(1)) * (60 if d["min"] else 1)
    if _has(HIGH, t):
        ev.severity = "high"
    elif _has(LOW, t):
        ev.severity = "low"
    return ev


def _default_text(ev: Evidence, zh: bool, zones: dict[str, dict]) -> tuple[str, str]:
    where_zh = "、".join(zones.get(z, {}).get("zh", z) for z in ev.zones) or ""
    where_en = ", ".join(zones.get(z, {}).get("en", z) for z in ev.zones) or ""
    objzh = ZH_OBJ.get(ev.object or "", ev.object or "")
    n = int(ev.threshold) if ev.threshold is not None and ev.threshold == int(ev.threshold) else ev.threshold
    if zh:
        msg = {
            "no_helmet": "未戴安全帽",
            "person_present": f"{where_zh or '管制區'}有人進入",
            "person_count_over": f"人數超過 {n} 人",
            "object_present": f"偵測到{objzh}",
            "object_near_person": f"有人在{objzh} {n} 公尺內",
        }.get(ev.type or "", "安全規則觸發")
        name = (f"{where_zh} " if where_zh else "") + msg
    else:
        msg = {
            "no_helmet": "No hard hat",
            "person_present": f"Person in {where_en or 'restricted area'}",
            "person_count_over": f"More than {n} people",
            "object_present": f"{(ev.object or 'object').replace('_', ' ').capitalize()} detected",
            "object_near_person": f"Person within {n} m of {(ev.object or 'object').replace('_', ' ')}",
        }.get(ev.type or "", "Safety rule triggered")
        name = (f"{where_en}: " if where_en else "") + msg
    return name.strip(), msg


def rule_from_evidence(text: str, ev: Evidence, zones: dict[str, dict]) -> dict[str, Any]:
    name, msg = _default_text(ev, _is_zh(text), zones)
    d: dict[str, Any] = {
        "name": name, "type": ev.type, "cameras": ev.cameras or ["*"], "zones": ev.zones,
        "object": ev.object, "threshold": ev.threshold, "time_window": ev.time_window,
        "severity": ev.severity or ("high" if ev.type in ("no_helmet", "object_near_person") else "medium"),
        "message": msg, "source_text": text,
    }
    if ev.min_duration_s is not None:
        d["min_duration_s"] = ev.min_duration_s
    elif ev.type in ("person_present", "object_present"):
        d["min_duration_s"] = 1.0
    return d


# ---------------------------------------------------------------- LLM side
LLM_FIELDS = ["name", "type", "cameras", "zones", "object", "threshold", "time_window",
              "min_duration_s", "severity", "message"]


def llm_schema() -> dict:
    schema = Rule.model_json_schema()
    props = {k: v for k, v in schema["properties"].items() if k in LLM_FIELDS}
    out = {"type": "object", "properties": props, "required": ["name", "type", "message"]}
    if "$defs" in schema:
        out["$defs"] = schema["$defs"]
    return out


def llm_prompt(text: str, cameras: list[dict], zones: dict[str, dict]) -> list[dict]:
    cams = "\n".join(f'- {c["id"]}: "{c["name"]}" (zone {c["zone"]})' for c in cameras)
    zs = "\n".join(f'- {z}: {n.get("en", "")} / {n.get("zh", "")}' for z, n in zones.items())
    types = "\n".join(f"- {k}: alert when {v}" for k, v in RULE_TYPES.items())
    sys = (
        "You turn a safety rule spoken by a site manager into one JSON object.\n"
        f"Rule types:\n{types}\n\nCameras:\n{cams}\n\nZones:\n{zs}\n\n"
        "Fields: name (short title), type, cameras (ids or [\"*\"]), zones (zone ids, may be empty), "
        "object (English noun for object_* types, e.g. forklift), threshold (people count or metres), "
        "time_window ({start,end} HH:MM 24h, or null), min_duration_s, severity (low|medium|high), "
        "message (alert text in the user's language).\nReply with JSON only."
    )
    ex_user = "B區超過3個人就提醒我"
    ex_out = {"name": "B區人數上限", "type": "person_count_over", "cameras": ["*"], "zones": ["zone_b"],
              "object": None, "threshold": 3, "time_window": None, "min_duration_s": 2,
              "severity": "medium", "message": "B區人數超過 3 人"}
    ex2_user = "Warn me immediately if anyone is within 3 metres of a forklift in the warehouse"
    ex2_out = {"name": "Forklift proximity", "type": "object_near_person", "cameras": ["*"], "zones": ["warehouse"],
               "object": "forklift", "threshold": 3, "time_window": None, "min_duration_s": 1,
               "severity": "high", "message": "Person within 3 m of a forklift"}
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": ex_user},
        {"role": "assistant", "content": json.dumps(ex_out, ensure_ascii=False)},
        {"role": "user", "content": ex2_user},
        {"role": "assistant", "content": json.dumps(ex2_out, ensure_ascii=False)},
        {"role": "user", "content": text},
    ]


def _clean_llm(d: dict, cameras: list[dict], zones: dict[str, dict]) -> dict:
    ids = {c["id"] for c in cameras}
    names = {c["name"].lower(): c["id"] for c in cameras}
    cams = []
    for c in d.get("cameras") or []:
        c = str(c)
        if c == "*" or c in ids:
            cams.append(c)
        elif c.lower() in names:
            cams.append(names[c.lower()])
    d["cameras"] = cams or ["*"]
    zrev = {}
    for z, n in zones.items():
        for k in (z, n.get("en", ""), n.get("zh", "")):
            if k:
                zrev[k.lower()] = z
    d["zones"] = [zrev[str(z).lower()] for z in d.get("zones") or [] if str(z).lower() in zrev]
    obj = d.get("object")
    if obj and _is_zh(str(obj)):
        hit = next((en for pat, en, _ in OBJECTS if _has(pat, str(obj))), None)
        d["object"] = hit
    if d.get("type") not in RULE_TYPES:
        d["type"] = None
    return {k: v for k, v in d.items() if k in LLM_FIELDS}


@dataclass
class ParseResult:
    rule: Rule | None
    parser: str                      # "llm+check" | "rules" | "failed"
    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def parse(text: str, cameras: list[dict], zones: dict[str, dict], llm=None) -> ParseResult:
    ev = extract(text, cameras, zones)
    base = rule_from_evidence(text, ev, zones)
    notes: list[str] = []
    data, parser = None, "rules"
    if llm is not None:
        try:
            raw = llm.chat(llm_prompt(text, cameras, zones), fmt=llm_schema(), temperature=0.0, max_tokens=300)
            data = _clean_llm(json.loads(raw), cameras, zones)
            parser = "llm+check"
        except Exception as e:  # LLM down, timeout, bad JSON -> deterministic path
            notes.append(f"LLM unavailable ({type(e).__name__}); used the rule extractor")
            data = None
    if data is None:
        data = dict(base)
    else:
        # explicit evidence wins over the LLM where they disagree
        for key in ("type", "object", "threshold", "time_window", "min_duration_s", "severity"):
            v = getattr(ev, key)
            if v is not None and data.get(key) != v:
                if data.get(key) is not None:
                    notes.append(f"{key}: sentence says {v!r}, LLM said {data.get(key)!r} -> using {v!r}")
                data[key] = v
        if ev.cameras:
            data["cameras"] = ev.cameras
        elif ev.zones and data.get("cameras") not in (None, [], ["*"]):
            # a zone was named and no camera: the zone is the scope, not a camera the LLM guessed
            notes.append(f"cameras: sentence names a zone and no camera, LLM said {data['cameras']!r} -> using every camera")
            data["cameras"] = ["*"]
        if ev.zones:
            data["zones"] = ev.zones
        for key in ("name", "message"):
            if not data.get(key):
                data[key] = base[key]
        if data.get("type") != ev.type and ev.type is None and data.get("type") is None:
            data["type"] = base["type"]
        data["source_text"] = text
        if "min_duration_s" not in data or data["min_duration_s"] is None:
            data["min_duration_s"] = base.get("min_duration_s", 2.0)
        _sanitise(data, ev, text, zones, notes)
    if not data.get("type"):
        return ParseResult(None, "failed", notes, ["could not tell what to watch for"])
    data = {k: v for k, v in data.items() if v is not None or k in ("object", "threshold", "time_window")}
    try:
        rule = Rule(**data)
    except Exception as e:
        if parser == "llm+check" and base.get("type"):
            # a malformed LLM field (e.g. time_window "22:00-06:00") must not sink a sentence the extractor understood
            res = parse(text, cameras, zones, llm=None)
            res.notes[:0] = notes + [f"LLM output was not a valid rule ({type(e).__name__}); used the rule extractor"]
            return res
        return ParseResult(None, "failed", notes, [f"invalid rule: {e}"])
    return ParseResult(rule, parser, notes, rule.check_complete() + scope_problems(rule, cameras)
                       + unknown_zone_problems(text, cameras, zones))


USES_THRESHOLD = {"person_count_over", "object_near_person"}
USES_OBJECT = {"object_present", "object_near_person"}


def _sanitise(data: dict, ev: Evidence, text: str, zones: dict[str, dict], notes: list[str]) -> None:
    """Remove what the LLM invented and make the alert text match the final rule.

    Observed with Qwen2.5-1.5B: an all-day 00:00-23:59 window and a threshold on a
    hard-hat rule for sentences that mention neither, alert messages describing a
    different rule type, simplified characters and English/Chinese mix-ups.  The alert
    message is therefore generated from the final fields; the LLM's title is kept only
    when it is in the user's language."""
    t = data.get("type")
    if t not in USES_THRESHOLD:
        data["threshold"] = None
    if t not in USES_OBJECT:
        data["object"] = None
    if ev.time_window is None and data.get("time_window"):
        notes.append("time_window: the sentence names no time -> dropped the LLM's window")
        data["time_window"] = None
    zh = _is_zh(text)
    final = Evidence(type=t, object=data.get("object"), threshold=data.get("threshold"), zones=data.get("zones") or [])
    name, msg = _default_text(final, zh, zones)
    data["message"] = msg
    llm_name = str(data.get("name") or "").strip()
    data["name"] = (_to_traditional(llm_name) if zh else llm_name) if llm_name and _is_zh(llm_name) == zh \
        and len(llm_name) <= 40 else name


def scope_problems(rule: Rule, cameras: list[dict]) -> list[str]:
    """A draft whose camera/zone filters select no configured camera could be activated but never fire
    ("cam01 B區..." when cam01 is at the entrance), so it is reported like any other problem."""
    if cameras and not any(rule.covers_any(c["id"], c.get("zones") or [c.get("zone", "")]) for c in cameras):
        return ["no camera matches this scope"]
    return []
