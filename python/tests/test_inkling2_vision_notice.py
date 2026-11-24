# SPDX-License-Identifier: Apache-2.0
"""tools/convert/inkling2_vision_convert.py: the NOTICE bullet about the image tower file and the manifest entry.

The bullet is wrapped like the other bullets of NOTICE, names no document section, and stands before the acceptable-use
bullet, so that bullet stays the last one and covers the image tower file too (at the end only when there is no such bullet).
A NOTICE that already has the bullet (the one-line form of an earlier version of the tool, or the wrapped form at the end)
comes out the same as a fresh one, and a second run adds nothing (same bytes).  No GPU and no model file.
"""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests import support  # noqa: F401  (puts the package on the path)

CONVERT = Path(__file__).resolve().parents[2] / "tools" / "convert" / "inkling2_vision_convert.py"

# The end of the published NOTICE: the weights bullet, two more bullets, and the acceptable-use bullet as the last one.
NOTICE = ("Model\n\n"
          "* MODIFIED WEIGHTS: w.yskn was made from the published model-00001-of-00001.safetensors.\n"
          "  The outputs differ slightly. The vision tower is not included.\n"
          "* The configuration, the tokenizer, the chat template and the model card are the published files, unchanged.\n"
          "* selfcheck.json and yasakani.json were written by the converter. The expected tokens in selfcheck.json were made\n"
          "  from these converted weights by a reference implementation of the model.\n")
POLICY = "* The acceptable-use policy above applies to the use of these converted weights as well.\n"

BULLET = ("* inkling_2_mini_vision.yskn holds the image tower of the model: the published bfloat16 tensors model.visual.* of\n"
          "  model-00001-of-00001.safetensors, unchanged, in the YASAKANI container format.\n")


def fixed(text: str) -> str:
    return text.replace("The vision tower is not included.", "The vision tower is not in this weights file.")


# What every case with the acceptable-use bullet must give: the bullet right before it, so that it stays the last one.
EXPECTED = fixed(NOTICE) + BULLET + POLICY


def load():
    spec = importlib.util.spec_from_file_location("ink2_vision_convert_notice_test", CONVERT)
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    return C


@unittest.skipUnless(CONVERT.is_file(), "tools/convert/inkling2_vision_convert.py not found")
class NoticeBullet(unittest.TestCase):
    def setUp(self):
        self.C = load()

    def run_update(self, dst: Path, notice: str):
        (dst / "NOTICE").write_bytes(notice.encode("utf-8"))                 # "\n" as in the published file, on every OS
        manifest = {"sha256": {"NOTICE": "0" * 64}, "sizes": {"NOTICE": 1}}
        (dst / "yasakani.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        r = dict(file="inkling_2_mini_vision.yskn", size=8, sha256="1" * 64)
        for _ in range(2):
            p = dict(manifest=json.loads((dst / "yasakani.json").read_text(encoding="utf-8")),
                     weights="model-00001-of-00001.safetensors")
            with contextlib.redirect_stdout(io.StringIO()):
                self.C.update_folder(dst, p, r)
            yield (dst / "NOTICE").read_bytes(), (dst / "yasakani.json").read_bytes()

    def check(self, start: str, expected: str) -> str:
        with tempfile.TemporaryDirectory() as d:
            (first, man1), (second, man2) = list(self.run_update(Path(d), start))
            self.assertEqual((first, man1), (second, man2))                  # a second run changes no byte
            text = first.decode("utf-8")
            self.assertEqual(text, expected)
            self.assertEqual(text.count("inkling_2_mini_vision.yskn"), 1)
            self.assertNotIn("section", text)
            for line in text.splitlines():
                self.assertLessEqual(len(line), 120, line)
            m = json.loads(man1)
            self.assertEqual(m["sizes"]["NOTICE"], len(first))
            self.assertEqual(m["vision"]["file"], "inkling_2_mini_vision.yskn")
            return text

    def test_bullet_text(self):
        self.assertEqual([len(s) for s in BULLET.splitlines()], [113, 80])
        self.assertEqual(self.C.NOTICE_LINE % ("inkling_2_mini_vision.yskn", "model-00001-of-00001.safetensors") + "\n",
                         BULLET)

    def test_fresh(self):
        text = self.check(NOTICE + POLICY, EXPECTED)
        self.assertTrue(text.endswith("\n" + POLICY))                         # the acceptable-use bullet stays the last one

    def test_already_at_the_old_end_position(self):
        self.check(fixed(NOTICE) + POLICY + BULLET, EXPECTED)

    def test_already_in_the_one_line_form(self):
        old = self.C.NOTICE_LINE_V1 % ("inkling_2_mini_vision.yskn", "model-00001-of-00001.safetensors")
        self.check(fixed(NOTICE) + POLICY + old + "\n", EXPECTED)

    def test_already_in_the_new_place(self):
        self.check(EXPECTED, EXPECTED)

    def test_no_policy_bullet(self):
        self.check(NOTICE, fixed(NOTICE) + BULLET)


if __name__ == "__main__":
    unittest.main()
