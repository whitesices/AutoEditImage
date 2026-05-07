from __future__ import annotations

import unittest
from pathlib import Path

from core.natural_language_command import (
    LLMParserSettings,
    normalize_chat_completions_url,
    parse_cut_command,
    parse_llm_command_json,
    resolve_cut_command,
)


class NaturalLanguageCommandTests(unittest.TestCase):
    def test_parse_english_cut_and_export_command(self) -> None:
        command = parse_cut_command(
            "cut out the right man with sword and save to ./my_outputs",
            fallback_output_dir="./outputs",
        )

        self.assertEqual(command.target, "man with sword")
        self.assertEqual(command.output_dir, Path("./my_outputs"))
        self.assertEqual(command.location_hint, "right")
        self.assertEqual(command.max_results, 1)
        self.assertTrue(command.auto_export)

    def test_parse_chinese_cut_and_export_command(self) -> None:
        command = parse_cut_command("抠出右边的人 导出到 ./my_outputs")

        self.assertEqual(command.target, "person")
        self.assertEqual(command.output_dir, Path("./my_outputs"))
        self.assertEqual(command.location_hint, "right")
        self.assertTrue(command.auto_export)

    def test_parse_target_only_command(self) -> None:
        command = parse_cut_command("extract target: a red car")

        self.assertEqual(command.target, "a red car")
        self.assertIsNone(command.output_dir)
        self.assertIsNone(command.location_hint)
        self.assertFalse(command.auto_export)

    def test_export_intent_uses_fallback_output_dir(self) -> None:
        command = parse_cut_command(
            "extract a red car and export",
            fallback_output_dir="./outputs",
        )

        self.assertEqual(command.target, "a red car")
        self.assertEqual(command.output_dir, Path("./outputs"))
        self.assertTrue(command.auto_export)

    def test_empty_command_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            parse_cut_command("   ")

    def test_normalize_openai_compatible_chat_url(self) -> None:
        self.assertEqual(
            normalize_chat_completions_url("https://api.deepseek.com/v1"),
            "https://api.deepseek.com/v1/chat/completions",
        )
        self.assertEqual(
            normalize_chat_completions_url("https://example.com/v1/chat/completions"),
            "https://example.com/v1/chat/completions",
        )

    def test_parse_llm_json_from_fenced_content(self) -> None:
        data = parse_llm_command_json(
            """```json
            {"target_prompt":"a man","location_hint":"left","auto_export":true,"output_dir":"./out","max_results":1}
            ```"""
        )

        self.assertEqual(data["target_prompt"], "a man")
        self.assertEqual(data["location_hint"], "left")

    def test_resolve_command_falls_back_when_llm_not_ready(self) -> None:
        command = resolve_cut_command(
            "抠出左边的人",
            llm_settings=LLMParserSettings(enabled=True, api_url="", model="deepseek-chat"),
        )

        self.assertEqual(command.target, "person")
        self.assertEqual(command.location_hint, "left")
        self.assertEqual(command.parser, "local")
