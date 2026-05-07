# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Intelligent image element extraction — natural language → Grounding DINO (detect) → SAM2 (segment) → postprocess → transparent PNG.

Design doc: `docs/头脑风暴_智能图片元素提取工具.md`
Full guide: `README.md`
Environment summary: `汇总.md`

## Environment

- Python 3.13.1, venv at `.venv/`, activate with `.venv\Scripts\activate`
- GPU: NVIDIA RTX 4090 (24GB), CUDA 12.8
- Key installed versions: torch 2.11.0+cu128, transformers 5.8.0, sam2 1.1.0, opencv-python 4.13.0
- No git repo initialized yet; `.claude/settings.local.json` is gitignored (contains API key)

See `汇总.md` for the full pip list.

## Common Commands

```bash
# Activate venv
.venv\Scripts\activate

# CLI help
python main.py --help

# Text-driven extraction
python main.py extract-text -i photo.jpg -t "a red car"

# Point-click extraction
python main.py extract-point -i photo.jpg -x 500 -y 375

# Image info
python main.py info -i photo.jpg

# Custom config
python main.py -c configs/my_config.yaml extract-text -i photo.jpg -t "a cat"

# Skip postprocessing (see raw SAM2 output)
python main.py extract-text -i photo.jpg -t "a cat" --no-postprocess

# Run with verbose logging
python main.py -v extract-text -i photo.jpg -t "a cat"

# Verify imports (no models needed)
python -c "from src import load_config; print(list(load_config().keys()))"
python -c "from src.segmentation.sam_prompts import PromptSet; ps = PromptSet().add_point(100,200,1); print(repr(ps))"
python -c "from src.postprocess.morph import refine_mask; import numpy as np; print(refine_mask(np.random.rand(100,100)).shape)"
python -c "from src.pipeline.extraction_pipeline import ExtractionPipeline; print('OK')"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## Agentic Engineering Workflow

- Read `docs/project_memory.md` and `docs/agentic_engineering.md` before
  non-trivial changes.
- Use `docs/tasks/TEMPLATE.md` for larger work that needs explicit scope,
  acceptance checks, and handoff notes.
- Run `python scripts/verify.py --quick` for every Python edit.
- Run `python scripts/verify.py` when behavior changes or tests are added.
- Add ADRs under `docs/adr/` when changing model contracts, output formats,
  architecture boundaries, or agent workflow.

Hooks are configured in `.claude/settings.local.json`: a `PostToolUse` hook runs `py_compile` on modified `.py` files after every `Write`/`Edit` to catch syntax errors immediately.

## Architecture Principles

- **Internal image representation**: numpy `(H, W, 3)` RGB uint8 throughout
- **Explicit model lifecycle**: `load()` / `unload()` on every model class; pipeline uses try/finally
- **SAM2 embedding cache**: `set_image()` precomputes once per image; subsequent `predict()` calls reuse the cached embeddings
- **Config-driven**: all model paths, thresholds, params in `configs/config.yaml` — no hardcoded values
- **Dual SAM2 backend**: `transformers` (HuggingFace Sam2Model API) and `native` (Meta sam2 package), switchable in config

## Key Design Decisions

- **SAM2Engine.predict()** returns `(masks, scores, logits)` — caller picks best mask via `argmax(scores)`. Mask shape is `(B, N, H, W)` where B=1 (batch), N=1 or 3 (multimask). Always `.squeeze()` the selected mask to get 2D `(H, W)` before OpenCV ops.
- **Grounding DINO text format**: must end with `.`; multiple objects separated by `". "`. The `detect()` method auto-appends the trailing period. Query in English for best results.
- **PromptSet** supports chaining: `PromptSet().add_box(...).add_point(...)`
- **Postprocess pipeline** order: threshold → morphological close (fill holes) → morphological open (remove noise) → connected component filter (drop tiny islands)
- Models not loaded in `__init__` — must call `load()` explicitly for light import-time footprint.
- `outputs/` and `__pycache__/` are gitignored.

## transformers 5.x API Changes

This project uses **transformers 5.8.0**. Several APIs differ from the 4.x versions the code was originally written for:

| Area | 4.x API (old) | 5.x API (current) |
|------|--------------|-------------------|
| GroundingDINO post_process | `box_threshold=value` | `threshold=value` |
| GroundingDINO results | `results["labels"]` (strings) | `results["text_labels"]` (strings) |
| Sam2Model get_image_embeddings | Returns object with `.image_embeddings` | Returns `list[Tensor]` directly |
| Sam2Processor predict call | `images=None` accepted | Must pass `original_sizes=[...]` |
| Mask shape from post_process | Varies | `(B, N, H, W)` — always includes batch dim |

If adding new model calls or upgrading transformers, check the actual parameter signatures with `inspect.signature()`.

## File Responsibilities

```
configs/config.yaml     — All tunable params (device, model IDs, thresholds, postprocess kernels, output settings)
main.py                 — Click CLI: extract-text, extract-point, info
src/__init__.py         — load_config(path) -> dict
src/detection/grounded_dino.py   — GroundingDINO class: text → List[DetectedObject]
src/segmentation/sam_engine.py   — SAM2Engine: set_image() cache, predict() with point/box/mask prompts
src/segmentation/sam_prompts.py  — PromptSet builder: add_point(), add_box(), set_mask()
src/postprocess/morph.py         — refine_mask(), mask_to_trimap()
src/pipeline/extraction_pipeline.py — ExtractionPipeline orchestrator + ExtractedElement/PipelineResult dataclasses
src/utils/image_utils.py         — load_image(), save_extracted_element(), resize_with_aspect_ratio()
src/utils/visualization.py       — draw_mask_overlay(), draw_detections(), create_debug_grid()
```
