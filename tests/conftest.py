"""Make the repo importable without installing it, whatever directory pytest runs from.

Tests here exercise pure-Python code only; none may import torch/ultralytics/transformers
or load a model, so they are safe to run while the GPU is busy with training.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
