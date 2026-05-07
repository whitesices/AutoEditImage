from __future__ import annotations

import os
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class UISmokeTests(unittest.TestCase):
    def test_main_window_instantiates_offscreen(self) -> None:
        from PySide6.QtWidgets import QApplication

        from app.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        try:
            self.assertEqual(window.windowTitle(), "Intelligent Image Element Extractor")
            self.assertFalse(window.project.has_image)
            self.assertIsNotNone(app)
        finally:
            window.close()
