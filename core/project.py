from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .layer import LayerItem, MaskResult


@dataclass(slots=True)
class ProjectData:
    """In-memory UI project state for one source image."""

    source_image_path: Path | None = None
    image: Image.Image | None = None
    layers: list[LayerItem] = field(default_factory=list)
    _next_layer_index: int = 1

    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def canvas_size(self) -> tuple[int, int]:
        if self.image is None:
            return (0, 0)
        return self.image.size

    @property
    def source_image_name(self) -> str:
        if self.source_image_path is None:
            return ""
        return self.source_image_path.name

    def set_source_image(self, path: str | Path, image: Image.Image) -> None:
        self.source_image_path = Path(path)
        self.image = image.copy()
        self.layers.clear()
        self._next_layer_index = 1

    def next_layer_id(self) -> str:
        layer_id = f"{self._next_layer_index:03d}"
        self._next_layer_index += 1
        return layer_id

    def add_layer_from_mask(self, name: str, result: MaskResult) -> LayerItem:
        if self.image is None:
            raise RuntimeError("Cannot create a layer before opening an image")
        if result.is_empty or result.bbox is None:
            raise ValueError("Cannot create a layer from an empty mask")
        if result.mask.shape[:2] != (self.image.height, self.image.width):
            raise ValueError("Mask size does not match the current image")

        layer = LayerItem(
            id=self.next_layer_id(),
            name=name,
            mask=result.mask,
            bbox=result.bbox,
            metadata={"segmenter": result.source, "score": result.score},
        )
        self.layers.append(layer)
        return layer

    def remove_layer(self, layer_id: str) -> bool:
        before = len(self.layers)
        self.layers = [layer for layer in self.layers if layer.id != layer_id]
        return len(self.layers) != before

    def get_layer(self, layer_id: str) -> LayerItem | None:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def to_project_dict(self) -> dict[str, Any]:
        width, height = self.canvas_size
        return {
            "source_image": self.source_image_name,
            "canvas": {"width": width, "height": height},
            "layers": [layer.to_project_dict() for layer in self.layers],
        }

