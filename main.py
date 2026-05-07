#!/usr/bin/env python
"""
Image Element Extractor - CLI
==============================
Phase 1 CLI entry point for testing the extraction pipeline.

Usage:
    python main.py extract-text --image path/to/image.jpg --text "a red car"
    python main.py extract-point --image path/to/image.jpg --x 500 --y 375
    python main.py info --image path/to/image.jpg
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from core.natural_language_command import parse_cut_command
from src import load_config
from src.pipeline.extraction_pipeline import ExtractionPipeline

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML config file.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable DEBUG-level logging.",
)
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """Intelligent Image Element Extraction Tool (Phase 1)."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    cfg = load_config(config)
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg


@cli.command()
@click.option(
    "--image",
    "-i",
    type=click.Path(exists=True),
    default=None,
    help="Optional image to open when the UI starts.",
)
def ui(image: str | None) -> None:
    """Launch the desktop UI."""
    from app.main_window import run_ui

    raise SystemExit(run_ui(image))


@cli.command()
@click.option(
    "--image", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Path to input image.",
)
@click.option(
    "--text", "-t",
    required=True,
    help="Text query for object detection (e.g., 'a red car').",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output directory (default: ./outputs).",
)
@click.option(
    "--no-postprocess",
    is_flag=True,
    help="Disable mask post-processing.",
)
@click.pass_context
def extract_text(
    ctx: click.Context,
    image: str,
    text: str,
    output: str | None,
    no_postprocess: bool,
) -> None:
    """Extract element from image using a text query."""
    config = ctx.obj["config"]

    pipeline = ExtractionPipeline(config)
    try:
        pipeline.load_models()

        result = pipeline.extract_text(
            image_path=image,
            text_query=text,
            apply_postprocess=not no_postprocess,
        )

        out_dir = output or config.get("output", {}).get("default_dir", "./outputs")
        saved = pipeline.save_results(result, output_dir=out_dir)

        click.echo(f"\nExtraction complete. Found {len(result)} element(s).")
        for i, elem in enumerate(result.elements):
            click.echo(f"  [{i}] {elem.label} (score: {elem.score:.3f})")
        click.echo(f"\nSaved {len(saved)} file(s) to: {Path(out_dir).resolve()}")
        if saved:
            click.echo(f"  Output: {saved[0]}")

    except Exception as e:
        logger.exception("Extraction failed")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        pipeline.unload_models()


@cli.command("cut")
@click.option(
    "--image", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Path to input image.",
)
@click.option(
    "--command", "-n",
    required=True,
    help=(
        "Natural-language cutting command, e.g. "
        "'cut out a man with sword and save to ./my_outputs'."
    ),
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Fallback output directory if the command does not include one.",
)
@click.option(
    "--no-postprocess",
    is_flag=True,
    help="Disable mask post-processing.",
)
@click.pass_context
def cut(
    ctx: click.Context,
    image: str,
    command: str,
    output: str | None,
    no_postprocess: bool,
) -> None:
    """Cut image elements using a natural-language command."""
    config = ctx.obj["config"]
    default_output = output or config.get("output", {}).get("default_dir", "./outputs")

    try:
        parsed = parse_cut_command(command, fallback_output_dir=default_output)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    out_dir = parsed.output_dir or Path(default_output)
    click.echo(f"Command: {parsed.raw}")
    click.echo(f"Target: {parsed.target}")
    click.echo(f"Output: {Path(out_dir).resolve()}")

    pipeline = ExtractionPipeline(config)
    try:
        pipeline.load_models()
        result = pipeline.extract_text(
            image_path=image,
            text_query=parsed.target,
            apply_postprocess=not no_postprocess,
            location_hint=parsed.location_hint,
            max_results=parsed.max_results,
        )
        saved = pipeline.save_results(result, output_dir=out_dir)

        click.echo(f"\nCut complete. Found {len(result)} element(s).")
        for i, elem in enumerate(result.elements):
            click.echo(f"  [{i}] {elem.label} (score: {elem.score:.3f})")
        click.echo(f"\nSaved {len(saved)} file(s) to: {Path(out_dir).resolve()}")
        for path in saved:
            click.echo(f"  {path}")
    except Exception as e:
        logger.exception("Natural-language cut failed")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        pipeline.unload_models()


@cli.command()
@click.option(
    "--image", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Path to input image.",
)
@click.option("--x", type=float, required=True, help="X coordinate of click point.")
@click.option("--y", type=float, required=True, help="Y coordinate of click point.")
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output directory (default: ./outputs).",
)
@click.pass_context
def extract_point(
    ctx: click.Context,
    image: str,
    x: float,
    y: float,
    output: str | None,
) -> None:
    """Extract element from image using a point click."""
    config = ctx.obj["config"]

    pipeline = ExtractionPipeline(config)
    try:
        pipeline.load_models()

        result = pipeline.extract_point(
            image_path=image,
            points=[(x, y)],
            labels=[1],
        )

        out_dir = output or config.get("output", {}).get("default_dir", "./outputs")
        saved = pipeline.save_results(result, output_dir=out_dir)

        click.echo(f"\nExtraction complete. Score: {result[0].score:.3f}")
        click.echo(f"Saved {len(saved)} file(s) to: {Path(out_dir).resolve()}")

    except Exception as e:
        logger.exception("Extraction failed")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        pipeline.unload_models()


@cli.command()
@click.option(
    "--image", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Path to input image.",
)
@click.pass_context
def info(ctx: click.Context, image: str) -> None:
    """Display image info (debug utility)."""
    from src.utils.image_utils import load_image

    img = load_image(image)
    click.echo(f"Image: {image}")
    click.echo(f"  Size: {img.shape[1]} x {img.shape[0]} (W x H)")
    click.echo(f"  Channels: {img.shape[2]}")
    click.echo(f"  Dtype: {img.dtype}")


if __name__ == "__main__":
    cli()
