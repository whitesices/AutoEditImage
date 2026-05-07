from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

from core.layer import LayerItem
from core.mask_utils import normalize_mask


BBox = tuple[int, int, int, int]


class CanvasWidget(QWidget):
    """Interactive image canvas with zoom, pan, selection, and mask preview."""

    selectionChanged = Signal(object)
    selectionCompleted = Signal(object)
    layerClicked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(520, 360)

        self._image: Image.Image | None = None
        self._qimage: QImage | None = None
        self._mask_preview: QImage | None = None
        self._layers: list[LayerItem] = []
        self._selected_layer_id: str | None = None

        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._is_selecting = False
        self._is_panning = False
        self._selection_start = QPointF()
        self._selection_end = QPointF()
        self._last_mouse_pos = QPointF()
        self._has_custom_view = False

    def set_image(self, image: Image.Image | None) -> None:
        self._image = image.copy() if image is not None else None
        self._qimage = self._pil_to_qimage(self._image) if self._image else None
        self._mask_preview = None
        self._layers = []
        self._selected_layer_id = None
        self.clear_selection()
        self.fit_to_view()

    def set_layers(
        self,
        layers: Iterable[LayerItem],
        selected_layer_id: str | None = None,
    ) -> None:
        self._layers = list(layers)
        self._selected_layer_id = selected_layer_id
        self.update()

    def set_preview_mask(self, mask: np.ndarray | None) -> None:
        self._mask_preview = self._mask_to_overlay_qimage(mask) if mask is not None else None
        self.update()

    def clear_selection(self) -> None:
        self._is_selecting = False
        self._selection_start = QPointF()
        self._selection_end = QPointF()
        self.selectionChanged.emit(None)
        self.update()

    def fit_to_view(self) -> None:
        if self._image is None or self.width() <= 0 or self.height() <= 0:
            self._zoom = 1.0
            self._pan = QPointF(0.0, 0.0)
            self.update()
            return

        margin = 32
        available_width = max(1, self.width() - margin * 2)
        available_height = max(1, self.height() - margin * 2)
        self._zoom = min(
            available_width / self._image.width,
            available_height / self._image.height,
        )
        self._zoom = max(0.05, min(8.0, self._zoom))
        displayed_width = self._image.width * self._zoom
        displayed_height = self._image.height * self._zoom
        self._pan = QPointF(
            (self.width() - displayed_width) / 2,
            (self.height() - displayed_height) / 2,
        )
        self._has_custom_view = False
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#202124"))

        if self._qimage is None or self._image is None:
            painter.setPen(QColor("#9aa0a6"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open Image")
            return

        painter.save()
        painter.translate(self._pan)
        painter.scale(self._zoom, self._zoom)
        painter.drawImage(0, 0, self._qimage)

        if self._mask_preview is not None:
            painter.drawImage(0, 0, self._mask_preview)

        self._draw_layers(painter)
        self._draw_selection(painter)
        painter.restore()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._image is not None and not self._has_custom_view:
            self.fit_to_view()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._image is None:
            return

        self._last_mouse_pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            clicked_layer = self._layer_at(event.position())
            if clicked_layer is not None:
                self.layerClicked.emit(clicked_layer.id)
            self._is_selecting = True
            image_pos = self._clamp_image_point(self._widget_to_image(event.position()))
            self._selection_start = image_pos
            self._selection_end = image_pos
            self._mask_preview = None
            self.selectionChanged.emit(self._selection_rect_image())
            self.update()
        elif event.button() in {Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton}:
            self._is_panning = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._image is None:
            return

        if self._is_selecting:
            self._selection_end = self._clamp_image_point(
                self._widget_to_image(event.position())
            )
            self.selectionChanged.emit(self._selection_rect_image())
            self.update()
        elif self._is_panning:
            delta = event.position() - self._last_mouse_pos
            self._pan += delta
            self._last_mouse_pos = event.position()
            self._has_custom_view = True
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._image is None:
            return

        if event.button() == Qt.MouseButton.LeftButton and self._is_selecting:
            self._is_selecting = False
            self._selection_end = self._clamp_image_point(
                self._widget_to_image(event.position())
            )
            rect = self._selection_rect_image()
            if rect is None or rect[2] < 2 or rect[3] < 2:
                self.clear_selection()
                self.selectionCompleted.emit(None)
                return
            self.selectionChanged.emit(rect)
            self.selectionCompleted.emit(rect)
            self.update()
        elif event.button() in {Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton}:
            self._is_panning = False
            self.unsetCursor()

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if self._image is None:
            return

        cursor_pos = event.position()
        image_pos = self._widget_to_image(cursor_pos)
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom = max(0.05, min(16.0, self._zoom * zoom_factor))
        self._pan = QPointF(
            cursor_pos.x() - image_pos.x() * self._zoom,
            cursor_pos.y() - image_pos.y() * self._zoom,
        )
        self._has_custom_view = True
        self.update()

    def _draw_layers(self, painter: QPainter) -> None:
        for layer in self._layers:
            if not layer.visible and layer.id != self._selected_layer_id:
                continue

            rect = QRectF(layer.x, layer.y, layer.width, layer.height)
            selected = layer.id == self._selected_layer_id
            pen = QPen(QColor("#00e5ff") if selected else QColor("#fbbc04"))
            pen.setWidthF((2.5 if selected else 1.5) / self._zoom)
            if not layer.visible:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

    def _draw_selection(self, painter: QPainter) -> None:
        rect = self._selection_rect_qrect()
        if rect is None:
            return
        pen = QPen(QColor("#ffffff"))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidthF(1.5 / self._zoom)
        painter.setPen(pen)
        painter.setBrush(QColor(255, 255, 255, 28))
        painter.drawRect(rect)

    def _selection_rect_qrect(self) -> QRectF | None:
        rect = self._selection_rect_image()
        if rect is None:
            return None
        x, y, width, height = rect
        return QRectF(float(x), float(y), float(width), float(height))

    def _selection_rect_image(self) -> BBox | None:
        if self._image is None:
            return None
        x1 = int(round(min(self._selection_start.x(), self._selection_end.x())))
        y1 = int(round(min(self._selection_start.y(), self._selection_end.y())))
        x2 = int(round(max(self._selection_start.x(), self._selection_end.x())))
        y2 = int(round(max(self._selection_start.y(), self._selection_end.y())))
        x1 = max(0, min(self._image.width, x1))
        y1 = max(0, min(self._image.height, y1))
        x2 = max(0, min(self._image.width, x2))
        y2 = max(0, min(self._image.height, y2))
        if x2 == x1 or y2 == y1:
            return None
        return (x1, y1, x2 - x1, y2 - y1)

    def _layer_at(self, widget_pos: QPointF) -> LayerItem | None:
        image_pos = self._widget_to_image(widget_pos)
        x = image_pos.x()
        y = image_pos.y()
        for layer in reversed(self._layers):
            if layer.x <= x <= layer.x + layer.width and layer.y <= y <= layer.y + layer.height:
                return layer
        return None

    def _widget_to_image(self, point: QPointF) -> QPointF:
        return QPointF(
            (point.x() - self._pan.x()) / self._zoom,
            (point.y() - self._pan.y()) / self._zoom,
        )

    def _clamp_image_point(self, point: QPointF) -> QPointF:
        if self._image is None:
            return point
        return QPointF(
            max(0.0, min(float(self._image.width), point.x())),
            max(0.0, min(float(self._image.height), point.y())),
        )

    @staticmethod
    def _pil_to_qimage(image: Image.Image) -> QImage:
        rgba = image.convert("RGBA")
        data = rgba.tobytes("raw", "RGBA")
        qimage = QImage(
            data,
            rgba.width,
            rgba.height,
            rgba.width * 4,
            QImage.Format.Format_RGBA8888,
        )
        return qimage.copy()

    @staticmethod
    def _mask_to_overlay_qimage(mask: np.ndarray) -> QImage:
        normalized = normalize_mask(mask)
        overlay = np.zeros((normalized.shape[0], normalized.shape[1], 4), dtype=np.uint8)
        overlay[..., 1] = 229
        overlay[..., 2] = 255
        overlay[..., 3] = (normalized.astype(np.float32) * 0.35).astype(np.uint8)
        qimage = QImage(
            overlay.tobytes(),
            normalized.shape[1],
            normalized.shape[0],
            normalized.shape[1] * 4,
            QImage.Format.Format_RGBA8888,
        )
        return qimage.copy()

