# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The image parts and messages of a request on a published Inkling-2 Mini folder (the Linux engine) follow the rules of the
converted folder: image_url, input_image and image parts in user and tool messages; video, images in system or assistant
messages, and (without image input) any image are refused with the same messages."""
from __future__ import annotations

import unittest

from yasakani.chat import RequestError, clean_messages
from yasakani.vision import split_images

URI = "data:image/png;base64,iVBORw0KGgo="


def user(*parts, role="user"):
    return [{"role": role, "content": list(parts)}]


class PartTypesTests(unittest.TestCase):
    def test_three_part_types_are_read(self):
        for part in ({"type": "image_url", "image_url": {"url": URI}}, {"type": "image_url", "image_url": URI},
                     {"type": "input_image", "image_url": URI}, {"type": "input_image", "image_url": {"url": URI}},
                     {"type": "image", "image": URI}, {"type": "image", "image_url": {"url": URI}}):
            with self.subTest(part=part):
                msgs = clean_messages(user({"type": "text", "text": "what is it?"}, part), images=True, tool_ids=True,
                                      vision_parts=True)
                out, sources = split_images(msgs, is_inkling2=True)
                self.assertEqual(sources, [URI])
                self.assertEqual([p["type"] for p in out[0]["content"]], ["text", "image"])

    def test_tool_message_image(self):
        msgs = clean_messages(user({"type": "input_image", "image_url": URI}, role="tool"), images=True, tool_ids=True,
                              vision_parts=True)
        self.assertEqual(split_images(msgs, is_inkling2=True)[1], [URI])

    def test_part_without_image(self):
        msgs = clean_messages(user({"type": "image"}), images=True, tool_ids=True, vision_parts=True)
        with self.assertRaises(RequestError) as cm:
            split_images(msgs, is_inkling2=True)
        self.assertIn("an image part needs its image", str(cm.exception))

    def test_refused_kinds(self):
        cases = [(user({"type": "video_url", "video_url": {"url": URI}}), "messages[0]: video is not supported (only still images)"),
                 (user({"type": "image_url", "image_url": URI}, role="system"),
                  "messages[0]: images are supported only in user and tool messages"),
                 (user({"type": "image", "image": URI}, role="assistant"),
                  "messages[0]: images are supported only in user and tool messages")]
        for msgs, text in cases:
            with self.subTest(text=text):
                with self.assertRaises(RequestError) as cm:
                    clean_messages(msgs, images=True, tool_ids=True, vision_parts=True)
                self.assertEqual(str(cm.exception), text)

    def test_without_image_input_the_reason(self):
        with self.assertRaises(RequestError) as cm:
            clean_messages(user({"type": "input_image", "image_url": URI}), images=False, tool_ids=True, vision_parts=True,
                           vision_problem="image input is off (--vision off)")
        self.assertEqual(str(cm.exception), "image input is not available: image input is off (--vision off)")

    def test_qwen_reads_image_url_only(self):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": URI}}, {"type": "text", "text": "x"}]}]
        out, sources = split_images(msgs)
        self.assertEqual(sources, [URI])
        self.assertEqual(out[0]["content"][0]["type"], "text")       # the Qwen3.8 marker text


if __name__ == "__main__":
    unittest.main()
