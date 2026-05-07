# ADR 0002: Desktop UI Layer Workflow

Status: Accepted

## Context

The original project provided a CLI for text/point-driven extraction using
Grounding DINO and SAM2. A related project at
`C:\ML\EditImage\ai_image_layer_extractor` already explored a desktop layer
extraction workflow with PySide6, OpenCV rectangle segmentation, layer metadata,
and transparent PNG export.

## Decision

Add a desktop UI to this repository while preserving the existing CLI/model
pipeline:

- `python main.py ui` launches the UI.
- `app/` owns PySide6 widgets and orchestration.
- `core/` owns UI layer contracts, mask utilities, project state, and export.
- `segmenters/` owns UI segmentation backends.
- Manual rectangle extraction uses `OpenCVSegmenter` with deterministic fallback.
- Toolbar text extraction calls the existing `src.pipeline.ExtractionPipeline`
  and imports AI masks as UI layers.

## Consequences

- The project is usable without loading heavy models: manual rectangle layers
  work locally through OpenCV/fallback.
- Model-backed text extraction remains available from the UI, but it is explicit
  and may download weights on first use.
- Exported UI projects follow the stable `Export/layers`, `Export/masks`,
  `project.json`, and `preview.png` contract.
- `PySide6==6.10.3` is required for the UI because the active Python is 3.13.

## Verification

Run:

```powershell
.venv\Scripts\python.exe scripts\verify.py
.venv\Scripts\python.exe main.py ui --help
```
