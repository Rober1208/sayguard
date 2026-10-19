"""Rule schema shared by the engine, the natural-language parser and the API."""
from __future__ import annotations

import re
import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

RuleType = Literal["no_helmet", "person_present", "person_count_over", "object_present", "object_near_person"]

RULE_TYPES: dict[str, str] = {
    "no_helmet": "a person is not wearing a hard hat",
    "person_present": "any person is present (restricted area)",
    "person_count_over": "more than `threshold` people are present at once",
    "object_present": "an object of class `object` is visible",
    "object_near_person": "an object of class `object` is within `threshold` metres of a person",
}


# site-detector class names; rule objects are matched against detector labels after canon_label
SITE_CLASSES = {"person", "forklift", "pallet_truck", "transporter", "mobile_robot", "humanoid_robot"}
_SYNONYMS = {"pallet_jack": "pallet_truck", "amr": "mobile_robot", "agv": "transporter", "humanoid": "humanoid_robot",
             "fork_lift": "forklift"}
# a generic word in a rule covers every detector class it names: "any robot" means humanoids,
# mobile robots and transporters alike, not a YOLO-World search for the word "robot"
CLASS_GROUPS = {"robot": frozenset({"humanoid_robot", "mobile_robot", "transporter"})}


def canon_label(name: str | None) -> str:
    """'Pallet trucks' / 'pallet-truck' / 'AMR' -> 'pallet_truck' / 'pallet_truck' / 'mobile_robot'."""
    n = re.sub(r"[\s\-]+", "_", (name or "").strip().lower())
    n = _SYNONYMS.get(n, n)
    if n.endswith("s") and (n[:-1] in SITE_CLASSES or n[:-1] in CLASS_GROUPS
                            or _SYNONYMS.get(n[:-1], "") in SITE_CLASSES):
        n = _SYNONYMS.get(n[:-1], n[:-1])
    return n


def label_set(name: str | None) -> frozenset[str]:
    """The canonical detector labels a rule's object stands for: itself, or every member of its group."""
    n = canon_label(name)
    return CLASS_GROUPS.get(n, frozenset()) | {n}


def detector_labels(name: str | None) -> frozenset[str]:
    """The classes a detector must look for to serve the rule object: a group's members, never the
    group word itself (so "robot" in site mode costs no open-vocabulary search)."""
    n = canon_label(name)
    return CLASS_GROUPS.get(n) or frozenset({n})


def label_matches(label: str | None, obj: str | None) -> bool:
    """Does a detection labelled `label` count as the rule object `obj`?"""
    return canon_label(label) in label_set(obj)


class TimeWindow(BaseModel):
    start: str = Field(description="HH:MM, 24-hour")
    end: str = Field(description="HH:MM, 24-hour; may be earlier than start for overnight windows")

    @field_validator("start", "end")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        h, m = (v.split(":") + ["0"])[:2]
        h, m = int(h), int(m)
        if not (0 <= h <= 24 and 0 <= m < 60):
            raise ValueError(f"bad time {v}")
        return f"{h % 24:02d}:{m:02d}"

    def contains(self, t: time.struct_time) -> bool:
        now = t.tm_hour * 60 + t.tm_min
        s = int(self.start[:2]) * 60 + int(self.start[3:])
        e = int(self.end[:2]) * 60 + int(self.end[3:])
        # start == end (e.g. 00:00-24:00, stored as 00:00-00:00) means all day, not never
        return s <= now < e if s < e else (now >= s or now < e)


class Rule(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    type: RuleType
    cameras: list[str] = Field(default_factory=lambda: ["*"], description='camera ids, or ["*"] for all')
    zones: list[str] = Field(default_factory=list, description="zone ids; empty = no zone filter")
    zones_exclude: list[str] = Field(default_factory=list)
    object: str | None = Field(default=None, description="English class name for object_* rules, e.g. forklift")
    threshold: float | None = Field(default=None, description="people count or metres")
    time_window: TimeWindow | None = None
    min_duration_s: float = 2.0
    cooldown_s: float = 30.0
    severity: Literal["low", "medium", "high"] = "medium"
    message: str = ""
    enabled: bool = True
    source_text: str | None = None
    created_at: float = Field(default_factory=time.time)

    @field_validator("object")
    @classmethod
    def _obj(cls, v: str | None) -> str | None:
        return v.strip().lower() if v else v

    def covers(self, cam_id: str, zone: str) -> bool:
        """Camera/zone scope only (not enabled or time_window).

        Defined once here so the engine (which fires rules) and the parser (which
        warns about a draft that matches no camera) cannot disagree.
        """
        if "*" not in self.cameras and cam_id not in self.cameras:
            return False
        if self.zones and zone not in self.zones:
            return False
        return zone not in self.zones_exclude

    def covers_any(self, cam_id: str, zones: list[str]) -> bool:
        """covers() for a camera that sees several zones (site mode: floor-plan zones)."""
        if "*" not in self.cameras and cam_id not in self.cameras:
            return False
        if self.zones and not any(z in self.zones for z in zones):
            return False
        return not zones or not all(z in self.zones_exclude for z in zones)

    def check_complete(self) -> list[str]:
        """Semantic problems that the JSON schema alone cannot catch."""
        problems = []
        if self.type in ("object_present", "object_near_person") and not self.object:
            problems.append("this rule type needs an object class")
        if self.type == "person_count_over" and (self.threshold is None or self.threshold < 0):
            problems.append("person_count_over needs a people-count threshold")
        if self.type == "object_near_person" and (self.threshold is None or self.threshold <= 0):
            problems.append("object_near_person needs a distance in metres")
        return problems
