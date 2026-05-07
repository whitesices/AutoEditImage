from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QToolBar,
    QWidget,
)

from app.canvas_widget import CanvasWidget
from app.layer_panel import LayerPanel
from app.llm_settings_dialog import LLMSettingsDialog
from core.exporter import LayerExporter
from core.image_utils import load_source_image
from core.layer import MaskResult
from core.llm_settings_store import load_llm_settings, save_llm_settings
from core.natural_language_command import resolve_cut_command
from core.project import ProjectData
from segmenters.opencv_segmenter import OpenCVSegmenter


BBox = tuple[int, int, int, int]


class MainWindow(QMainWindow):
    """Desktop UI that combines manual layers with the AI extraction pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Intelligent Image Element Extractor")
        self.resize(1280, 820)

        self.project = ProjectData()
        self.segmenter = OpenCVSegmenter()
        self.current_selection: BBox | None = None
        self.current_mask_result: MaskResult | None = None
        self.selected_layer_id: str | None = None
        self.llm_settings = load_llm_settings()

        self.canvas = CanvasWidget()
        self.layer_panel = LayerPanel()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.layer_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)

        self.text_query = QLineEdit()
        self.text_query.setPlaceholderText(
            "Natural command, e.g. cut out a man with sword export to ./my_outputs"
        )
        self.text_query.setMinimumWidth(420)

        self._build_toolbar()
        self._connect_signals()
        self._apply_style()
        self.statusBar().showMessage("Ready")

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Open Image", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

        create_action = QAction("Create Layer", self)
        create_action.setShortcut(QKeySequence("Ctrl+L"))
        create_action.triggered.connect(self.create_layer_from_selection)
        toolbar.addAction(create_action)

        export_action = QAction("Export All", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export_all_layers)
        toolbar.addAction(export_action)

        fit_action = QAction("Fit View", self)
        fit_action.setShortcut(QKeySequence("Ctrl+0"))
        fit_action.triggered.connect(self.canvas.fit_to_view)
        toolbar.addAction(fit_action)

        spacer = QWidget()
        spacer.setFixedWidth(12)
        toolbar.addWidget(spacer)
        toolbar.addWidget(self.text_query)

        text_button = QPushButton("Run NL Cut")
        text_button.clicked.connect(self.extract_layers_from_text)
        toolbar.addWidget(text_button)

        settings_button = QPushButton("LLM Settings")
        settings_button.clicked.connect(self.open_llm_settings)
        toolbar.addWidget(settings_button)

    def _connect_signals(self) -> None:
        self.canvas.selectionChanged.connect(self._on_selection_changed)
        self.canvas.selectionCompleted.connect(self._on_selection_completed)
        self.canvas.layerClicked.connect(self._select_layer)

        self.layer_panel.createLayerRequested.connect(self.create_layer_from_selection)
        self.layer_panel.exportAllRequested.connect(self.export_all_layers)
        self.layer_panel.exportLayerRequested.connect(self.export_single_layer)
        self.layer_panel.deleteLayerRequested.connect(self.delete_layer)
        self.layer_panel.layerRenamed.connect(self.rename_layer)
        self.layer_panel.layerVisibilityChanged.connect(self.set_layer_visible)
        self.layer_panel.layerSelected.connect(self._select_layer)

    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if file_path:
            self.open_image_path(file_path)

    def open_image_path(self, file_path: str | Path) -> None:
        try:
            image = load_source_image(file_path)
        except Exception as exc:
            self._show_error("Open Image Failed", str(exc))
            return

        self.project.set_source_image(file_path, image)
        self.current_selection = None
        self.current_mask_result = None
        self.selected_layer_id = None
        self.canvas.set_image(image)
        self._refresh_layers()
        self.statusBar().showMessage(
            f"Opened {Path(file_path).name} ({image.width}x{image.height})"
        )

    def create_layer_from_selection(self) -> None:
        if self.project.image is None:
            self._show_info("Open an image first.")
            return

        if self.current_mask_result is None:
            if self.current_selection is None:
                self._show_info("Select an area on the image first.")
                return
            if not self._generate_preview_mask(self.current_selection):
                return

        try:
            layer_name = f"layer_{len(self.project.layers) + 1:03d}"
            layer = self.project.add_layer_from_mask(layer_name, self.current_mask_result)
        except Exception as exc:
            self._show_error("Create Layer Failed", str(exc))
            return

        self.selected_layer_id = layer.id
        self.current_mask_result = None
        self.current_selection = None
        self.canvas.clear_selection()
        self.canvas.set_preview_mask(None)
        self._refresh_layers()
        self.statusBar().showMessage(
            f"Created {layer.id} {layer.name}: x={layer.x}, y={layer.y}, "
            f"{layer.width}x{layer.height}"
        )

    def extract_layers_from_text(self) -> None:
        if self.project.image is None or self.project.source_image_path is None:
            self._show_info("Open an image first.")
            return

        command = self.text_query.text().strip()
        if not command:
            self._show_info("Enter a natural-language cut command first.")
            return

        try:
            parsed = resolve_cut_command(
                command,
                fallback_output_dir=self._default_export_dir(),
                llm_settings=self.llm_settings,
            )
        except ValueError as exc:
            self._show_error("Command Parse Failed", str(exc))
            return

        self.statusBar().showMessage(
            f"Parsed by {parsed.parser}: target={parsed.target}, location={parsed.location_hint or 'any'}"
        )
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            from src import load_config
            from src.pipeline.extraction_pipeline import ExtractionPipeline

            pipeline = ExtractionPipeline(load_config())
            try:
                pipeline.load_models()
                result = pipeline.extract_text(
                    self.project.source_image_path,
                    parsed.target,
                    location_hint=parsed.location_hint,
                    max_results=parsed.max_results,
                )
            finally:
                pipeline.unload_models()

            if len(result) == 0:
                self._show_info(f"No elements detected for: {parsed.target}")
                return

            created = 0
            for index, element in enumerate(result.elements, start=1):
                mask = np.asarray(element.mask)
                mask_result = MaskResult(
                    mask=mask,
                    score=element.score,
                    source="grounding_dino_sam2",
                )
                name = element.label or f"{parsed.target}_{index:03d}"
                layer = self.project.add_layer_from_mask(name, mask_result)
                self.selected_layer_id = layer.id
                created += 1

            self.current_mask_result = None
            self.current_selection = None
            self.canvas.clear_selection()
            self.canvas.set_preview_mask(None)
            self._refresh_layers()

            message = f"Created {created} AI layer(s) for: {parsed.target}"
            if parsed.auto_export:
                output_dir = parsed.output_dir or self._default_export_dir()
                export_result = LayerExporter(output_dir).export_all_layers(self.project)
                exported_count = (
                    len(export_result["layers"])
                    if isinstance(export_result["layers"], list)
                    else 0
                )
                message += f"; exported {exported_count} layer(s) to {output_dir}"
            self.statusBar().showMessage(message)
        except Exception as exc:
            self._show_error("Natural-Language Cut Failed", str(exc))
        finally:
            QGuiApplication.restoreOverrideCursor()

    def open_llm_settings(self) -> None:
        dialog = LLMSettingsDialog(self.llm_settings, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.llm_settings = dialog.settings()
        save_llm_settings(self.llm_settings)
        mode = "enabled" if self.llm_settings.enabled else "disabled"
        self.statusBar().showMessage(f"LLM parser {mode}: {self.llm_settings.model}")

    def export_all_layers(self) -> None:
        if self.project.image is None:
            self._show_info("Open an image first.")
            return
        if not self.project.layers:
            self._show_info("Create at least one layer before exporting.")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Export Layers",
            str(self._default_export_dir()),
        )
        if not output_dir:
            return

        try:
            result = LayerExporter(output_dir).export_all_layers(self.project)
        except Exception as exc:
            self._show_error("Export Failed", str(exc))
            return

        self._refresh_layers()
        exported_count = len(result["layers"]) if isinstance(result["layers"], list) else 0
        self.statusBar().showMessage(f"Exported {exported_count} layer(s) to {output_dir}")
        QMessageBox.information(self, "Export Complete", f"Exported {exported_count} layer(s).")

    def export_single_layer(self, layer_id: str) -> None:
        if self.project.image is None:
            self._show_info("Open an image first.")
            return

        layer = self.project.get_layer(layer_id)
        if layer is None:
            self._show_info("Layer does not exist.")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Export Layer",
            str(self._default_export_dir()),
        )
        if not output_dir:
            return

        try:
            exporter = LayerExporter(output_dir)
            exporter.export_layer_png(self.project, layer)
            exporter.export_mask_png(self.project, layer)
        except Exception as exc:
            self._show_error("Export Layer Failed", str(exc))
            return

        self._refresh_layers()
        self.statusBar().showMessage(f"Exported layer {layer.id} to {output_dir}")

    def delete_layer(self, layer_id: str) -> None:
        removed = self.project.remove_layer(layer_id)
        if removed:
            if self.selected_layer_id == layer_id:
                self.selected_layer_id = self.project.layers[-1].id if self.project.layers else None
            self._refresh_layers()
            self.statusBar().showMessage(f"Deleted layer {layer_id}")

    def rename_layer(self, layer_id: str, name: str) -> None:
        layer = self.project.get_layer(layer_id)
        if layer is None:
            return
        try:
            layer.rename(name)
        except ValueError as exc:
            self._show_error("Rename Failed", str(exc))
            self._refresh_layers()
            return
        self._refresh_layers()

    def set_layer_visible(self, layer_id: str, visible: bool) -> None:
        layer = self.project.get_layer(layer_id)
        if layer is None:
            return
        layer.set_visible(visible)
        self._refresh_layers()

    def _on_selection_changed(self, rect: BBox | None) -> None:
        self.current_selection = rect
        if rect is not None:
            x, y, width, height = rect
            self.statusBar().showMessage(f"Selection: x={x}, y={y}, {width}x{height}")

    def _on_selection_completed(self, rect: BBox | None) -> None:
        self.current_selection = rect
        self.current_mask_result = None
        self.canvas.set_preview_mask(None)
        if rect is not None:
            self._generate_preview_mask(rect)

    def _generate_preview_mask(self, rect: BBox) -> bool:
        if self.project.image is None:
            return False
        try:
            result = self.segmenter.segment(self.project.image, rect)
        except Exception as exc:
            self._show_error("Mask Generation Failed", str(exc))
            return False
        if result.is_empty:
            self._show_info("The selected area did not produce a usable mask.")
            return False
        self.current_mask_result = result
        self.canvas.set_preview_mask(result.mask)
        if result.bbox is not None:
            x, y, width, height = result.bbox
            self.statusBar().showMessage(f"Mask ready: x={x}, y={y}, {width}x{height}")
        return True

    def _select_layer(self, layer_id: str) -> None:
        if self.project.get_layer(layer_id) is None:
            return
        self.selected_layer_id = layer_id
        self._refresh_layers()

    def _refresh_layers(self) -> None:
        self.layer_panel.set_layers(self.project.layers, self.selected_layer_id)
        self.canvas.set_layers(self.project.layers, self.selected_layer_id)

    def _default_export_dir(self) -> Path:
        if self.project.source_image_path is None:
            return Path.cwd() / "Export"
        return self.project.source_image_path.parent / "Export"

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _show_info(self, message: str) -> None:
        QMessageBox.information(self, "Image Element Extractor", message)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #2b2c30;
                color: #f1f3f4;
                font-size: 13px;
            }
            QToolBar {
                background: #25262a;
                border: none;
                spacing: 6px;
                padding: 6px;
            }
            QPushButton, QToolButton {
                background: #3c4043;
                border: 1px solid #5f6368;
                border-radius: 4px;
                padding: 6px 10px;
            }
            QPushButton:hover, QToolButton:hover {
                background: #4b4f54;
            }
            QPushButton:disabled {
                color: #8a8d91;
                background: #313236;
            }
            QListWidget {
                background: #202124;
                border: 1px solid #4b4f54;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 4px 6px;
                min-height: 34px;
            }
            QListWidget::item:selected {
                background: #145a66;
            }
            QListWidget QLineEdit {
                min-height: 28px;
                padding: 2px 6px;
                selection-background-color: #1976d2;
                selection-color: #ffffff;
            }
            QLineEdit {
                background: #202124;
                border: 1px solid #5f6368;
                border-radius: 4px;
                padding: 6px;
            }
            QStatusBar {
                background: #25262a;
            }
            """
        )


def run_ui(initial_image: str | Path | None = None) -> int:
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    if initial_image is not None:
        window.open_image_path(initial_image)
    window.show()
    return app.exec()
