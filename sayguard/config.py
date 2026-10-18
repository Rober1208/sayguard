"""Configuration loading.  Heavy artifacts (datasets, weights, clips, the event
database) live in a work directory outside the repo; SAYGUARD_WORK overrides it."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "site.yaml"


def work_dir() -> Path:
    return Path(os.environ.get("SAYGUARD_WORK", "D:/ugen-work"))


def resolve(path: str | os.PathLike, base: Path | None = None) -> Path:
    """Absolute paths stay; '@work/x' -> work_dir()/x; others are repo-relative."""
    s = str(path)
    if s.startswith("@work/"):
        return work_dir() / s[len("@work/"):]
    p = Path(s)
    return p if p.is_absolute() else (base or ROOT) / p


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(cfg_path)
    return cfg
