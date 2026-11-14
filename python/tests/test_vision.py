"""Image input: message splitting, placeholder expansion and positions, feature checks, image reading.  No GPU, no stub library."""
import base64
import io
import unittest

import numpy as np

from yasakani.chat import ContextLengthError, RequestError
from yasakani.vision import MARKER, MAX_IMAGES, build_visual_prompt, read_image, split_images, validate_features

IMG = 99   # stands in for the <|image_pad|> token id


def fake_image(h, w, fill=1):
    n = h * w
    return dict(features=np.full((n, 5120), fill, np.uint16),
                positions=np.stack([np.zeros(n, np.int32), np.arange(n, dtype=np.int32) // w, np.arange(n, dtype=np.int32) % w]),
                grid=np.array([h, w], np.int32))


def image_part(url="data:image/png;base64,AAAA"):
    return {"type": "image_url", "image_url": {"url": url}}


class SplitImagesTests(unittest.TestCase):
    def test_text_only_is_returned_unchanged(self):
        msgs = [{"role": "user", "content": "hello"}]
        out, sources = split_images(msgs)
        self.assertIs(out, msgs)
        self.assertEqual(sources, [])

    def test_images_become_markers_in_order(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "a"}, image_part("data:1"), image_part("data:2")]}]
        out, sources = split_images(msgs)
        self.assertEqual(sources, ["data:1", "data:2"])
        self.assertEqual([p["text"] for p in out[0]["content"]], ["a", MARKER, MARKER])
        self.assertEqual(msgs[0]["content"][1]["type"], "image_url")      # the caller's messages are not changed

    def test_rejections(self):
        with self.assertRaises(RequestError):
            split_images([{"role": "assistant", "content": [image_part()]}])
        with self.assertRaises(RequestError):
            split_images([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": ""}}]}])
        with self.assertRaises(RequestError):
            split_images([{"role": "user", "content": [image_part()] * (MAX_IMAGES + 1)}])


class VisualPromptTests(unittest.TestCase):
    def test_expansion_slots_and_positions(self):
        ids, v = build_visual_prompt([1, IMG, 2], [fake_image(2, 3)], IMG)
        self.assertEqual(ids.tolist(), [1] + [IMG] * 6 + [2])
        self.assertEqual(v["slots"].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(v["positions"].shape, (3, 8))
        # text before: position 0 on all three axes; the image: [time, row, column] from 1; text after: 1 + max(2, 3)
        self.assertEqual(v["positions"][:, 0].tolist(), [0, 0, 0])
        self.assertEqual(v["positions"][:, 1:7].T.tolist(), [[1, 1 + r, 1 + c] for r in range(2) for c in range(3)])
        self.assertEqual(v["positions"][:, 7].tolist(), [4, 4, 4])
        self.assertEqual(v["delta"], 5 - 8)                                # next position minus the number of tokens
        self.assertEqual(v["features"].shape, (6, 5120))

    def test_two_images(self):
        ids, v = build_visual_prompt([IMG, 7, IMG], [fake_image(1, 2, 1), fake_image(2, 2, 2)], IMG)
        self.assertEqual(len(ids), 2 + 1 + 4)
        self.assertEqual(v["slots"].tolist(), [0, 1, 3, 4, 5, 6])
        self.assertEqual(v["features"][:2, 0].tolist(), [1, 1])
        self.assertEqual(v["features"][2:, 0].tolist(), [2, 2, 2, 2])

    def test_count_mismatch_and_context(self):
        with self.assertRaises(ValueError):
            build_visual_prompt([1, 2], [fake_image(1, 1)], IMG)
        with self.assertRaises(ContextLengthError):
            build_visual_prompt([1, IMG], [fake_image(4, 4)], IMG, max_tokens=10)


class FeatureCheckTests(unittest.TestCase):
    def test_bad_features(self):
        item = fake_image(2, 2)
        item["features"] = item["features"][:3]
        with self.assertRaises(ValueError):
            validate_features(item)
        item = fake_image(2, 2)
        item["features"][0, 0] = 0x7F80                                    # bfloat16 infinity
        with self.assertRaises(ValueError):
            validate_features(item)
        item = fake_image(2, 2)
        item["positions"] = item["positions"][:, ::-1].copy()
        with self.assertRaises(ValueError):
            validate_features(item)


class ReadImageTests(unittest.TestCase):
    def test_sources(self):
        with self.assertRaises(ValueError):
            read_image("https://example.com/a.png")
        with self.assertRaises(ValueError):
            read_image("C:/a.png")                                         # local files need the server option
        with self.assertRaises(ValueError):
            read_image("data:text/plain;base64,AAAA")

    def test_data_uri(self):
        try:
            from PIL import Image
        except ImportError:
            raise unittest.SkipTest("Pillow is not installed")
        buf = io.BytesIO()
        Image.new("RGBA", (5, 3), (255, 0, 0, 128)).save(buf, format="PNG")
        im = read_image("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
        self.assertEqual((im.mode, im.size), ("RGB", (5, 3)))


if __name__ == "__main__":
    unittest.main()
