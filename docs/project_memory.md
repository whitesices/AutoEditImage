# Project Memory

Last updated: 2026-05-07

This file is the durable memory for future AI agents working in this repository.
Keep it concise, factual, and safe to share. Do not store secrets here.

## One-Sentence Mission

Build an intelligent image element extraction tool: natural language or point
selection drives Grounding DINO detection, SAM2 segmentation, mask
postprocessing, and transparent PNG export.

## Current Product State

- Phase: Phase 1 MVP.
- Main user interface: Click CLI in `main.py`.
- Desktop UI: `python main.py ui` launches a PySide6 layer extraction UI.
- Working modes:
  - `extract-text`: text query -> Grounding DINO bbox -> SAM2 mask -> PNG.
  - `extract-point`: point prompt -> SAM2 mask -> PNG.
  - `cut`: natural-language command -> parsed target/export dir -> text
    extraction -> PNG.
  - `info`: image dimensions and dtype.
- Default output directory: `outputs/`.
- Sample/local output directory: `my_outputs/`.
- Heavy model checks are manual and opt-in; normal verification must not
  download model weights.
- UI manual rectangle extraction does not require model weights; UI text
  extraction does use the existing Grounding DINO + SAM2 pipeline.
- UI and CLI natural-language cutting commands are parsed by
  `core/natural_language_command.py`.
- UI can enable an OpenAI-compatible LLM parser from `LLM Settings`; tested
  default model name is `deepseek-chat`.

## Environment Snapshot

- OS/workspace: Windows, `C:\ML\Deep_Claude_project`.
- Python: 3.13.1.
- Virtualenv: `.venv/`, activate with `.venv\Scripts\activate`.
- GPU target: NVIDIA RTX 4090, CUDA 12.8 available in the project notes.
- Important installed versions from project notes:
  - `torch` 2.11.0+cu128
  - `transformers` 5.8.0
  - `sam2` 1.1.0
  - `opencv-python` 4.13.0
- As of this memory entry, the folder is not initialized as a git repository.

## Source Map

- `main.py`: CLI entry point and command wiring.
- `configs/config.yaml`: model IDs, thresholds, device preferences, output
  options, and postprocess parameters.
- `src/__init__.py`: `load_config(path)`.
- `src/detection/grounded_dino.py`: Grounding DINO wrapper and detection
  dataclasses.
- `src/segmentation/sam_engine.py`: SAM2 backend abstraction, image embedding
  cache, and prompt inference.
- `src/segmentation/sam_prompts.py`: fluent prompt builders.
- `src/postprocess/morph.py`: mask thresholding, morphology, connected
  component filtering, trimap helper.
- `src/pipeline/extraction_pipeline.py`: orchestration, result dataclasses,
  mask application, and result saving.
- `src/utils/image_utils.py`: image loading, resizing, RGBA PNG saving.
- `src/utils/visualization.py`: overlays and detection drawing.
- `app/`: PySide6 canvas, layer panel, and main desktop window.
- `core/`: UI layer contracts, mask utilities, project state, and exporter.
- `core/natural_language_command.py`: deterministic command parser shared by
  CLI and UI, plus OpenAI-compatible LLM parser helpers.
- `core/llm_settings_store.py`: local UI settings persistence for LLM parser
  URL, model, key, enable flag, and timeout.
- `app/llm_settings_dialog.py`: UI dialog for filling LLM API URL/API Key/model.
- `segmenters/`: UI segmentation backends, currently OpenCV/rectangle fallback.
- `tests/`: pure unit tests that avoid model downloads.
- `scripts/verify.py`: canonical verification entry point for agents.
- `docs/agentic_engineering.md`: agent workflow and verification tiers.
- `docs/ui_integration.md`: UI usage, architecture, and verification notes.
- `docs/adr/`: architecture decision records.
- `docs/tasks/`: task brief templates.

## Core Contracts

- Internal images are RGB `numpy.ndarray` values shaped `(H, W, 3)` with
  dtype `uint8`.
- OpenCV mask operations expect 2D masks shaped `(H, W)`.
- Model classes do not load weights in `__init__`; call `load()` and
  `unload()` explicitly.
- `SAM2Engine.set_image()` should be called once per image so embeddings can be
  reused across prompts.
- `SAM2Engine.predict()` returns `(masks, scores, logits)`.
- The pipeline chooses the best SAM2 mask with
  `ExtractionPipeline._select_best_mask()`, which handles both transformers
  shape `(B, N, H, W)` and native shape `(N, H, W)`.
- Grounding DINO text queries should end in `.`, and the wrapper auto-appends
  the trailing period.
- Runtime behavior belongs in `configs/config.yaml`; avoid hardcoding model IDs,
  thresholds, or output defaults.
- UI layers store full-canvas `uint8` masks and crop only during export.
- UI export contract is `Export/layers`, `Export/masks`, `project.json`, and
  `preview.png`.
- Location hints from command parsing are applied before SAM2 segmentation:
  `left`, `right`, `top`, `bottom`, `center`, and the four corners.

## Verification Memory

Use these commands before handoff:

```powershell
.venv\Scripts\python.exe scripts\verify.py --quick
.venv\Scripts\python.exe scripts\verify.py
.venv\Scripts\python.exe main.py --help
.venv\Scripts\python.exe main.py ui --help
.venv\Scripts\python.exe main.py cut --help
```

`scripts/verify.py` uses Python stdlib `unittest` by default, so verification
does not require installing `pytest`. `requirements-dev.txt` keeps `pytest` as
an optional convenience dependency.

Current baseline when this file was created:

- `python scripts/verify.py --quick`: passes.
- `python scripts/verify.py`: passes, 19 unit tests.
- `python main.py --help`: passes.
- `python main.py ui --help`: passes.
- `python main.py cut -i GAS2.png -n "cut out a man with sword and save to ./my_outputs"`:
  passes with model access; found 1 element and saved element/overlay PNG.
- `python scripts/verify.py`: now covers LLM settings persistence, LLM JSON
  parsing, and location-based detection selection.

## Agent Workflow Memory

- Start with `AGENTS.md`, `docs/project_memory.md`, and
  `docs/agentic_engineering.md`.
- For non-trivial work, create a task brief from `docs/tasks/TEMPLATE.md`.
- For architectural changes, add an ADR under `docs/adr/`.
- Keep changes small and verifiable.
- Do not rely on hidden chat context; update this file when durable facts
  change.
- Prefer pure tests for mask utilities, prompt contracts, config loading, and
  pipeline helpers.
- Only run model-backed extraction checks when the task touches model calls,
  output quality, prompt conversion, or backend compatibility.

## Security And Local State

- Do not commit API keys or local tool auth configuration.
- Local-only ignored files include:
  - `.codex/config.toml`
  - `.codex/settings.local.json`
  - `.claude/settings.local.json`
  - `.env` and `.env.*`
- Ignored generated/heavy state includes:
  - `outputs/`
  - `my_outputs/`
  - `models/`
  - `.test_tmp/`
  - `.venv/`
  - `__pycache__/`
- `rg --files` is filtered by `.ignore` because the project currently has no
  git repo and `.gitignore` alone is not enough for local search hygiene.

## Known Risks And Sharp Edges

- Model-backed behavior is not covered by the fast test suite.
- Transformers 5.x APIs differ from older examples. Check real signatures with
  `inspect.signature()` before changing Grounding DINO or SAM2 calls.
- `README.md`, `AGENTS.md`, and `CLAUDE.md` contain Chinese project context;
  preserve UTF-8 text carefully when editing.
- Windows sandbox permissions can make `tempfile.TemporaryDirectory()` produce
  inaccessible directories. Tests should write deterministic files under
  `.test_tmp/` instead of relying on the default temp directory.
- Two inaccessible ignored `.test_tmp/tmp*` directories may remain from earlier
  permission probing; they are ignored and not part of the project state.
- The file `InstallPrompt.txt` appears to contain a pending dependency split
  request about CPU/GPU PyTorch installation; verify intent with the user before
  treating it as active work.

## Useful Next Tasks

- Initialize git and make the current agentic engineering baseline the first
  tracked checkpoint.
- Add a `requirements-cu128.txt` or clear GPU install guide if dependency
  installation becomes a priority.
- Add CLI box-prompt extraction mode.
- Add UI mask brush editing and project save/open if interactive editing grows.
- Add one deterministic model-backed smoke fixture once local model cache
  policy is settled.
- Add output contract tests for filenames, overlays, optional masks, and
  metadata if export behavior expands.
