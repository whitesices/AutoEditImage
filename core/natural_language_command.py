from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOCATION_HINTS = {
    "left",
    "right",
    "top",
    "bottom",
    "center",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
}

_EXPORT_PATTERNS = (
    r"(?:save|export|output|write)\s+(?:to|into|as)\s+(?P<path1>[^\s,;]+)",
    r"(?:保存|导出|输出|写入)\s*(?:到|至|为)?\s*(?P<path2>[^\s，,；;]+)",
)

_TARGET_PREFIX_PATTERNS = (
    r"^(?:please\s+)?(?:cut\s+out|cut|extract|segment|isolate|crop|remove\s+background\s+for)\s+",
    r"^(?:请)?(?:切割|切出|抠出|提取|分割|裁剪|抠图|去背景)\s*",
    r"^(?:把|将)\s*",
)

_TARGET_SUFFIX_PATTERNS = (
    r"\s+(?:from|out\s+of)\s+(?:the\s+)?(?:image|picture|photo).*$",
    r"\s*(?:从|在)?(?:图片|图像|照片|画面)(?:中|里)?(?:切出来|抠出来|提取出来|分割出来)?.*$",
    r"\s*(?:并|然后)?\s*(?:save|export|output|write|保存|导出|输出|写入).*$",
)

_LOCATION_RULES = (
    ("top_left", ("top left", "upper left", "左上", "左上角")),
    ("top_right", ("top right", "upper right", "右上", "右上角")),
    ("bottom_left", ("bottom left", "lower left", "左下", "左下角")),
    ("bottom_right", ("bottom right", "lower right", "右下", "右下角")),
    ("left", ("left", "左边", "左侧", "左面", "左方")),
    ("right", ("right", "右边", "右侧", "右面", "右方")),
    ("top", ("top", "upper", "上方", "上面", "顶部")),
    ("bottom", ("bottom", "lower", "下方", "下面", "底部")),
    ("center", ("center", "middle", "中央", "中间", "正中")),
)

_LOCATION_REMOVALS = (
    r"\b(?:the\s+)?(?:left|right|top|bottom|center|middle)\s+(?=(?:man|person|woman|character|object|icon|car|cup|sword)\b)",
    r"\b(?:in|at|on)\s+(?:the\s+)?(?:left|right|top|bottom|center|middle)\b",
    r"\b(?:top|bottom)\s+(?:left|right)\b",
    r"\b(?:upper|lower)\s+(?:left|right)\b",
    r"(?:左上角|右上角|左下角|右下角|左边|右边|左侧|右侧|上方|下方|中间|中央|顶部|底部|位置|这个位置|当前位置)",
)

_LOCAL_TRANSLATIONS = {
    "人": "person",
    "人物": "person",
    "男人": "man",
    "男性": "man",
    "女人": "woman",
    "女性": "woman",
    "角色": "character",
    "小人": "character",
    "法师": "wizard",
    "战士": "warrior",
    "剑": "sword",
    "图标": "icon",
    "按钮": "button",
    "车": "car",
    "汽车": "car",
}


@dataclass(frozen=True, slots=True)
class NaturalLanguageCutCommand:
    """Parsed natural-language image cutting command."""

    raw: str
    target: str
    output_dir: Path | None = None
    auto_export: bool = False
    location_hint: str | None = None
    max_results: int | None = None
    parser: str = "local"

    @property
    def display_output_dir(self) -> str:
        return str(self.output_dir) if self.output_dir is not None else ""


@dataclass(frozen=True, slots=True)
class LLMParserSettings:
    """OpenAI-compatible LLM parser settings."""

    enabled: bool = False
    api_url: str = ""
    api_key: str = ""
    model: str = "deepseek-chat"
    timeout_seconds: float = 30.0

    @property
    def is_ready(self) -> bool:
        return self.enabled and bool(self.api_url.strip()) and bool(self.model.strip())


def parse_cut_command(
    command: str,
    fallback_output_dir: str | Path | None = None,
) -> NaturalLanguageCutCommand:
    """Parse a lightweight natural-language image cutting command locally."""
    raw = command.strip()
    if not raw:
        raise ValueError("Natural language command cannot be empty.")

    output_dir = _parse_output_dir(raw)
    auto_export = output_dir is not None or _contains_export_intent(raw)
    if output_dir is None and auto_export and fallback_output_dir is not None:
        output_dir = Path(fallback_output_dir)

    location_hint = _parse_location_hint(raw)
    target = _parse_target(raw)
    if not target:
        raise ValueError("Could not find an extraction target in the command.")

    return NaturalLanguageCutCommand(
        raw=raw,
        target=target,
        output_dir=output_dir,
        auto_export=auto_export,
        location_hint=location_hint,
        max_results=1 if location_hint else None,
    )


def resolve_cut_command(
    command: str,
    fallback_output_dir: str | Path | None = None,
    llm_settings: LLMParserSettings | None = None,
) -> NaturalLanguageCutCommand:
    """Use an OpenAI-compatible LLM parser when configured, otherwise local rules."""
    if llm_settings is not None and llm_settings.is_ready:
        try:
            return parse_cut_command_with_llm(
                command,
                llm_settings,
                fallback_output_dir=fallback_output_dir,
            )
        except Exception:
            return parse_cut_command(command, fallback_output_dir=fallback_output_dir)
    return parse_cut_command(command, fallback_output_dir=fallback_output_dir)


def parse_cut_command_with_llm(
    command: str,
    settings: LLMParserSettings,
    fallback_output_dir: str | Path | None = None,
) -> NaturalLanguageCutCommand:
    """Parse a command through an OpenAI-compatible chat/completions endpoint."""
    raw = command.strip()
    if not raw:
        raise ValueError("Natural language command cannot be empty.")

    payload = _build_chat_payload(raw, settings.model)
    request = urllib.request.Request(
        normalize_chat_completions_url(settings.api_url),
        data=json.dumps(payload).encode("utf-8"),
        headers=_build_headers(settings.api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    content = _extract_chat_content(data)
    parsed = parse_llm_command_json(content)
    output_dir = _path_or_none(parsed.get("output_dir")) or _parse_output_dir(raw)
    auto_export = bool(parsed.get("auto_export")) or output_dir is not None or _contains_export_intent(raw)
    if output_dir is None and auto_export and fallback_output_dir is not None:
        output_dir = Path(fallback_output_dir)

    target = str(parsed.get("target_prompt") or "").strip()
    if not target:
        target = _parse_target(raw)
    location_hint = _normalize_location_hint(parsed.get("location_hint")) or _parse_location_hint(raw)
    max_results = _normalize_max_results(parsed.get("max_results"))
    if max_results is None and location_hint:
        max_results = 1

    if not target:
        raise ValueError("LLM response did not provide an extraction target.")

    return NaturalLanguageCutCommand(
        raw=raw,
        target=target,
        output_dir=output_dir,
        auto_export=auto_export,
        location_hint=location_hint,
        max_results=max_results,
        parser="llm",
    )


def normalize_chat_completions_url(api_url: str) -> str:
    value = api_url.strip().rstrip("/")
    if not value:
        raise ValueError("LLM API URL is required.")
    if value.endswith("/chat/completions"):
        return value
    return f"{value}/chat/completions"


def parse_llm_command_json(content: str) -> dict[str, Any]:
    """Parse JSON from model content, including fenced code blocks."""
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object.")
    return data


def _build_chat_payload(command: str, model: str) -> dict[str, Any]:
    system_prompt = (
        "You parse image element extraction commands. Return only JSON. "
        "Translate Chinese object descriptions to short English detection prompts. "
        "Use location_hint only when the user specifies position. "
        "Allowed location_hint values: left, right, top, bottom, center, "
        "top_left, top_right, bottom_left, bottom_right, null. "
        "Schema: {\"target_prompt\": string, \"location_hint\": string|null, "
        "\"auto_export\": boolean, \"output_dir\": string|null, "
        "\"max_results\": integer|null}."
    )
    user_prompt = f"Command: {command}"
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }


def _build_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def _extract_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response missing message.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response message content is empty.")
    return content


def _parse_output_dir(command: str) -> Path | None:
    for pattern in _EXPORT_PATTERNS:
        match = re.search(pattern, command, flags=re.IGNORECASE)
        if match:
            value = next((part for part in match.groupdict().values() if part), "")
            value = value.strip().strip("\"'")
            if value:
                return Path(value)
    return None


def _contains_export_intent(command: str) -> bool:
    lowered = command.lower()
    return any(token in lowered for token in ("save", "export", "output", "保存", "导出", "输出"))


def _parse_location_hint(command: str) -> str | None:
    lowered = command.lower()
    for hint, tokens in _LOCATION_RULES:
        if any(token in lowered for token in tokens):
            return hint
    return None


def _parse_target(command: str) -> str:
    target = command.strip().strip("\"'")

    for pattern in _EXPORT_PATTERNS:
        target = re.sub(pattern, "", target, flags=re.IGNORECASE).strip()

    for pattern in _TARGET_PREFIX_PATTERNS:
        target = re.sub(pattern, "", target, flags=re.IGNORECASE).strip()

    for pattern in _TARGET_SUFFIX_PATTERNS:
        target = re.sub(pattern, "", target, flags=re.IGNORECASE).strip()

    for pattern in _LOCATION_REMOVALS:
        target = re.sub(pattern, "", target, flags=re.IGNORECASE).strip()

    target = re.sub(r"^(?:目标|对象|target|object)\s*[:：]\s*", "", target, flags=re.IGNORECASE)
    target = re.sub(r"(?:\s+(?:and|then)|(?:并|然后))\s*$", "", target, flags=re.IGNORECASE)
    target = re.sub(r"^的", "", target)
    target = target.strip(" ，,。.;；:\"'")
    return _translate_local_target(target)


def _translate_local_target(target: str) -> str:
    normalized = target.strip()
    if normalized in _LOCAL_TRANSLATIONS:
        return _LOCAL_TRANSLATIONS[normalized]
    return normalized


def _normalize_location_hint(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in LOCATION_HINTS else None


def _normalize_max_results(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip().strip("\"'")
    return Path(text) if text else None
