"""
Intelligent Image Element Extractor
------------------------------------
Source package root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config. Falls back to project default."""
    p = Path(path) if path else _CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
