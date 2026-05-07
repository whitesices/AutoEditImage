from __future__ import annotations

import unittest
from pathlib import Path

from core.llm_settings_store import load_llm_settings, save_llm_settings
from core.natural_language_command import LLMParserSettings


class LLMSettingsStoreTests(unittest.TestCase):
    def test_save_and_load_llm_settings(self) -> None:
        path = Path.cwd() / ".test_tmp" / "llm_settings" / "settings.json"
        settings = LLMParserSettings(
            enabled=True,
            api_url="https://api.deepseek.com/v1",
            api_key="test-key",
            model="deepseek-chat",
            timeout_seconds=15.0,
        )

        saved_path = save_llm_settings(settings, path)
        loaded = load_llm_settings(saved_path)

        self.assertEqual(saved_path, path)
        self.assertTrue(loaded.enabled)
        self.assertEqual(loaded.api_url, "https://api.deepseek.com/v1")
        self.assertEqual(loaded.api_key, "test-key")
        self.assertEqual(loaded.model, "deepseek-chat")
        self.assertEqual(loaded.timeout_seconds, 15.0)
