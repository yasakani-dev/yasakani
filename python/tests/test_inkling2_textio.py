# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from yasakani.chat import clean_messages
from yasakani.runtime import ModelPack
from yasakani.textio import ChatFormatter, Inkling2OutputStream

INK2_DIR = Path("/home/au/models/Inkling-2-Mini")
LOCAL_INK2_DIR = Path("/Volumes/w/models/Inkling-2-Mini")


def _model_dir() -> Path | None:
    for p in (INK2_DIR, LOCAL_INK2_DIR):
        if (p / "tokenizer.json").is_file() and (p / "chat_template.jinja").is_file():
            return p
    return None


class Inkling2TextIoTest(unittest.TestCase):
    def test_clean_messages_preserves_tool_ids_and_names(self) -> None:
        msgs = [
            {"role": "assistant", "name": "get_weather", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Taipei"}'}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp":24}'},
        ]
        cleaned = clean_messages(msgs, tool_ids=True)
        self.assertEqual(cleaned[0]["name"], "get_weather")
        self.assertEqual(cleaned[0]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(cleaned[1]["tool_call_id"], "call_1")

    def test_formatter_and_stream_against_model_files(self) -> None:
        mdir = _model_dir()
        if mdir is None:
            self.skipTest("Inkling-2-Mini files not present")
        pack = ModelPack.load(mdir)
        self.assertTrue(pack.is_inkling2)
        self.assertEqual(pack.vocab_size, 200058)
        self.assertEqual(pack.eos_ids, (200006, 199999))

        fmt = ChatFormatter(mdir)
        self.assertTrue(fmt.is_inkling2)

        r1 = fmt.render([{"role": "user", "content": "Hello!"}], add_generation_prompt=False, reasoning_effort="high")
        self.assertEqual(
            r1,
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
            "<|message_user|><|content_text|>Hello!<|end_message|>",
        )

        msgs2 = [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Reply with one word: 2+2="},
        ]
        rendered2 = fmt.render(msgs2, add_generation_prompt=False, reasoning_effort="none")
        self.assertEqual(
            rendered2,
            "<|message_system|><|content_text|>You are terse.<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
            "<|message_user|><|content_text|>Reply with one word: 2+2=<|end_message|>",
        )
        ids, in_reasoning = fmt.prompt(msgs2, enable_thinking=False)
        self.assertFalse(in_reasoning)
        self.assertEqual(ids.tolist(), fmt.encode(rendered2).tolist())

        stream = Inkling2OutputStream(fmt, hidden_ids=pack.eos_ids)
        sample_toks = fmt.encode(
            "<|message_model|><|content_thinking|>Need weather.<|end_message|>"
            "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Taipei\"}}<|end_message|>"
            "<|message_model|><|content_text|> 台灣<|end_message|><|content_model_end_sampling|>"
        )
        events = []
        for i in range(len(sample_toks)):
            events.extend(stream.feed(sample_toks[: i + 1], final=(i + 1 == len(sample_toks))))
        reasoning = "".join(v for k, v in events if k == "reasoning")
        content = "".join(v for k, v in events if k == "content")
        calls = [v for k, v in events if k == "tool_call"]
        self.assertEqual(reasoning, "Need weather.")
        self.assertEqual(content, " 台灣")
        self.assertEqual(calls, [{"name": "get_weather", "arguments": {"city": "Taipei"}}])
        self.assertGreater(stream.reasoning_tokens, 0)


if __name__ == "__main__":
    unittest.main()
