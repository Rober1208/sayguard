"""Rule schema shared by the engine, the natural-language parser and the API."""
from __future__ import annotations

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
