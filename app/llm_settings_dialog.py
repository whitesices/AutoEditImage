from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from core.natural_language_command import LLMParserSettings


class LLMSettingsDialog(QDialog):
    """Edit OpenAI-compatible LLM parser settings."""

    def __init__(self, settings: LLMParserSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LLM Parser Settings")
        self.setMinimumWidth(560)

        self.enabled = QCheckBox("Enable LLM parsing")
        self.enabled.setChecked(settings.enabled)

        self.api_url = QLineEdit(settings.api_url)
        self.api_url.setPlaceholderText("https://api.deepseek.com/v1")

        self.model = QLineEdit(settings.model)
        self.model.setPlaceholderText("deepseek-chat")

        self.api_key = QLineEdit(settings.api_key)
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("sk-...")

        self.timeout = QSpinBox()
        self.timeout.setRange(5, 180)
        self.timeout.setValue(int(settings.timeout_seconds))
        self.timeout.setSuffix(" sec")

        form = QFormLayout()
        form.addRow(self.enabled)
        form.addRow("API URL", self.api_url)
        form.addRow("Model", self.model)
        form.addRow("API Key", self.api_key)
        form.addRow("Timeout", self.timeout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def settings(self) -> LLMParserSettings:
        return LLMParserSettings(
            enabled=self.enabled.isChecked(),
            api_url=self.api_url.text().strip(),
            api_key=self.api_key.text().strip(),
            model=self.model.text().strip() or "deepseek-chat",
            timeout_seconds=float(self.timeout.value()),
        )
