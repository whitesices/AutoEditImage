#!/usr/bin/env python
"""Local verification entry point for humans and AI agents."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str]) -> None:
    print(f"\n== {name} ==", flush=True)
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_python(name: str, code: str) -> None:
    run_step(name, [sys.executable, "-c", code])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run repository verification checks."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run compile and import smoke checks only.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unit tests even outside quick mode.",
    )
    args = parser.parse_args()

    run_step(
        "Compile Python files",
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "app",
            "core",
            "main.py",
            "scripts",
            "segmenters",
            "src",
            "tests",
        ],
    )

    run_python(
        "Config smoke",
        "from src import load_config; cfg = load_config(); "
        "assert 'models' in cfg and 'output' in cfg; print(sorted(cfg.keys()))",
    )
    run_python(
        "PromptSet smoke",
        "from src.segmentation.sam_prompts import PromptSet; "
        "ps = PromptSet().add_point(100, 200, 1).add_box(1, 2, 3, 4); "
        "assert len(ps) == 2; print(repr(ps))",
    )
    run_python(
        "Morph smoke",
        "import numpy as np; from src.postprocess.morph import refine_mask; "
        "mask = refine_mask(np.random.rand(32, 32)); "
        "assert mask.shape == (32, 32); print(mask.dtype)",
    )
    run_python(
        "Pipeline import smoke",
        "from src.pipeline.extraction_pipeline import ExtractionPipeline; "
        "print(ExtractionPipeline.__name__)",
    )

    if not args.quick and not args.skip_tests:
        run_step(
            "Unit tests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        )

    print("\nVerification complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
