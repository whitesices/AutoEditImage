from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .natural_language_command import LLMParserSettings


def default_settings_path() -> Path:
    base_dir = os.environ.get("LOCALAPPDATA")
    if base_dir:
        return Path(base_dir) / "DeepClaudeImageExtractor" / "llm_settings.json"
    return Path.home() / ".deep_claude_image_extractor" / "llm_settings.json"


def load_llm_settings(path: str | Path | None = None) -> LLMParserSettings:
    settings_path = Path(path) if path is not None else default_settings_path()
    if not settings_path.exists():
        return LLMParserSettings()
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return LLMParserSettings()
    if not isinstance(data, dict):
        return LLMParserSettings()
    return LLMParserSettings(
        enabled=bool(data.get("enabled", False)),
        api_url=str(data.get("api_url", "")),
        api_key=str(data.get("api_key", "")),
        model=str(data.get("model", "deepseek-chat")),
        timeout_seconds=float(data.get("timeout_seconds", 30.0)),
    )


def save_llm_settings(
    settings: LLMParserSettings,
    path: str | Path | None = None,
) -> Path:
    settings_path = Path(path) if path is not None else default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = asdict(settings)
    settings_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return settings_path
