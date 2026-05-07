from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .image_utils import sanitize_filename
from .layer import LayerItem
from .mask_utils import apply_mask_to_rgba, normalize_mask
from .project import ProjectData


class LayerExporter:
    """Export layer PNGs, masks, preview, and project metadata."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.layers_dir = self.output_dir / "layers"
        self.masks_dir = self.output_dir / "masks"

    def _ensure_dirs(self) -> None:
        self.layers_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def layer_file_stem(layer: LayerItem) -> str:
        return f"{layer.id}_{sanitize_filename(layer.name)}"

    def export_layer_png(self, project: ProjectData, layer: LayerItem) -> Path:
        if project.image is None:
            raise RuntimeError("No source image is loaded")
        self._ensure_dirs()
        filename = f"{self.layer_file_stem(layer)}.png"
        output_path = self.layers_dir / filename
        layer_image = apply_mask_to_rgba(project.image, layer.mask, layer.bbox)
        if layer.opacity < 1.0:
            alpha = layer_image.getchannel("A").point(
                lambda value: int(value * layer.opacity)
            )
            layer_image.putalpha(alpha)
        layer_image.save(output_path)
        layer.file = f"layers/{filename}"
        return output_path

    def export_mask_png(self, project: ProjectData, layer: LayerItem) -> Path:
        if project.image is None:
            raise RuntimeError("No source image is loaded")
        self._ensure_dirs()
        filename = f"{self.layer_file_stem(layer)}_mask.png"
        output_path = self.masks_dir / filename
        x, y, width, height = layer.bbox
        mask_crop = normalize_mask(layer.mask)[y : y + height, x : x + width]
        Image.fromarray(mask_crop, mode="L").save(output_path)
        layer.mask_file = f"masks/{filename}"
        return output_path

    def export_project_json(self, project: ProjectData) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "project.json"
        output_path.write_text(
            json.dumps(project.to_project_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    def export_preview(self, project: ProjectData) -> Path:
        if project.image is None:
            raise RuntimeError("No source image is loaded")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "preview.png"

        preview = Image.new("RGBA", project.image.size, (0, 0, 0, 0))
        visible_layers = [layer for layer in project.layers if layer.visible]
        if visible_layers:
            for layer in visible_layers:
                layer_image = apply_mask_to_rgba(project.image, layer.mask, layer.bbox)
                preview.alpha_composite(layer_image, dest=(layer.x, layer.y))
        else:
            preview = project.image.convert("RGBA")
        preview.save(output_path)
        return output_path

    def export_all_layers(self, project: ProjectData) -> dict[str, Path | list[Path]]:
        if project.image is None:
            raise RuntimeError("No source image is loaded")

        self._ensure_dirs()
        layer_paths: list[Path] = []
        mask_paths: list[Path] = []
        for layer in project.layers:
            layer_paths.append(self.export_layer_png(project, layer))
            mask_paths.append(self.export_mask_png(project, layer))

        project_json = self.export_project_json(project)
        preview = self.export_preview(project)
        return {
            "layers": layer_paths,
            "masks": mask_paths,
            "project": project_json,
            "preview": preview,
        }

