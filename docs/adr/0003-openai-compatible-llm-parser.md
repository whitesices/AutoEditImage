# ADR 0003: OpenAI-Compatible LLM Command Parser

Status: Accepted

## Context

Local deterministic parsing is not enough for natural Chinese commands such as
"抠出右边的人" or "把中间的角色导出到 my_outputs". The downstream detection
models work best with short English prompts, while users need to express intent
and location naturally in Chinese.

## Decision

Add an optional OpenAI-compatible LLM parsing layer:

- UI exposes `LLM Settings` for API URL, model, API key, enable flag, and
  timeout.
- API calls use the Chat Completions shape:
  `POST {base_url}/chat/completions`.
- The default model name is `deepseek-chat`.
- The LLM must return JSON with `target_prompt`, `location_hint`,
  `auto_export`, `output_dir`, and `max_results`.
- Local deterministic parsing remains the fallback when the LLM is disabled or
  fails.
- Parsed `location_hint` is applied to detections before SAM2 segmentation so
  commands such as "right person" or "右边的人" select the intended instance.

## Consequences

- Chinese command quality can be improved without adding an LLM SDK dependency.
- API keys stay in local user settings, not repository files.
- The core pipeline remains testable without network access.
- External LLM behavior is still bounded by a strict JSON schema and local
  fallback.

## Verification

Run:

```powershell
.venv\Scripts\python.exe scripts\verify.py
.venv\Scripts\python.exe main.py cut --help
```
