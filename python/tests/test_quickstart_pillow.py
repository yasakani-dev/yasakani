# SPDX-License-Identifier: Apache-2.0
"""The quickstart and Pillow, which image input of Inkling-2 Mini needs (the extra ENGINE_EXTRAS of the package).

Step 4 installs the engine folder with its extras and huggingface_hub in one pip call; when that fails, the command of 0.3.4
(without the extras) runs, the install goes on with text only and one line says how to add Pillow later; when both fail, the
old failure (exit code 4).  An environment in place gets no pip call.  An Inkling-2 Mini folder with the image file: when the
image file is asked for (the yes answer, or START.cmd --image-file) Pillow is added once when it is missing; at a start without
such a request one line says that images need Pillow.  The Qwen models are never checked.

No GPU and no network: main() runs with the stand-ins of test_quickstart_web.LauncherTests, and every subprocess.call of the
launcher goes to a stand-in that answers the pip and Python commands.
"""
import json
import os
import sys
import unittest
from unittest import mock

from tests import test_quickstart_ink2_vision as IV
from tests import test_quickstart_web as W

QS = W.QS
NONE_LINE = "Pillow (needed for image input) could not be installed. Text works as before."
COMMAND = 'python.exe" -m pip install Pillow'
NEED_LINE = "Image input will not work until Pillow is added to the Python environment (text works)."
ADD_LINE = "Adding Pillow (needed for image input) to the Python environment."
PIP = ["-m", "pip", "install", "--disable-pip-version-check"]
REAL_PILLOW_OK = getattr(QS, "pillow_ok", None)


class Say:
    def __init__(self):
        self.said = []

    def say(self, key, **kw):
        self.said.append(key)


class CommandTests(unittest.TestCase):
    """install_package() with a stand-in for subprocess.call."""

    def setUp(self):
        if QS is None:
            self.skipTest("quickstart/quickstart.py not found (set YASAKANI_QUICKSTART)")

    def run_install(self, *rcs):
        calls, ui = [], Say()
        got = QS.install_package(ui, "py", r"C:\a b\engine", call=lambda c, **k: calls.append(c) or rcs[len(calls) - 1])
        return got, calls, ui.said

    def test_the_extras_are_read_from_one_plain_line(self):
        with open(QS.__file__, encoding="utf-8") as f:
            self.assertIn('\nENGINE_EXTRAS = "vision"\n', f.read())

    def test_one_pip_call_with_the_extras(self):
        got, calls, said = self.run_install(0)
        self.assertEqual(got, True)
        self.assertEqual(calls, [["py"] + PIP + [r"C:\a b\engine[vision]", "huggingface_hub"]])
        self.assertEqual(said, [])

    def test_the_command_of_034_when_the_extras_fail(self):
        got, calls, said = self.run_install(1, 0)
        self.assertEqual(got, False)
        self.assertEqual(calls[1], ["py"] + PIP + [r"C:\a b\engine", "huggingface_hub"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(said, ["pillow_none"])

    def test_both_fail(self):
        got, calls, said = self.run_install(1, 1)
        self.assertIsNone(got)
        self.assertEqual(len(calls), 2)
        self.assertEqual(said, [])

    def test_the_lines_in_both_languages_name_the_command(self):
        vpy = r"C:\a b\venv\Scripts\python.exe"
        for key in ("pillow_none", "pillow_need"):
            for lang in ("ja", "en"):
                self.assertIn('"%s" -m pip install Pillow' % vpy, QS.Ui(lang).t(key, p=vpy))

    def test_the_check_imports_what_image_input_imports(self):
        calls = []
        self.assertTrue(QS.pillow_ok("py", call=lambda c, **k: calls.append(c) or 0))
        self.assertFalse(QS.pillow_ok("py", call=lambda c, **k: 1))
        self.assertEqual(calls, [["py", "-c", "from PIL import Image, ImageOps"]])


class PipStandIn:
    """subprocess.call of the launcher: the pip commands and the Python check, with the exit codes given."""

    def __init__(self, full=0, plain=0, pil=0, add=0):
        self.rc, self.calls = dict(full=full, plain=plain, pil=pil, add=add), []

    def kind(self, c):
        c = [str(x) for x in c]
        if c[1:3] == ["-c", getattr(QS, "PILLOW_CHECK", "")]:
            return "pil"
        if c[1:5] == PIP:
            if c[5:] == [getattr(QS, "PILLOW_REQ", "Pillow>=10")]:
                return "add"
            return "full" if c[5].endswith("[%s]" % getattr(QS, "ENGINE_EXTRAS", "vision")) else "plain"
        raise AssertionError("an unexpected command: %r" % (c,))

    def __call__(self, c, **kw):
        k = self.kind(c)
        self.calls.append((k, list(c)))
        return self.rc[k]

    def kinds(self):
        return [k for k, _ in self.calls]


@unittest.skipUnless(sys.platform == "win32", "the launcher itself runs on Windows only")
class MainTests(IV.OfferTests):
    """main() from step 4 to the end, the environment and Pillow."""

    def setUp(self):
        super().setUp()
        p = mock.patch.object(QS, "pillow_ok", REAL_PILLOW_OK)                 # OfferTests puts a stand-in in its place
        p.start()
        self.addCleanup(p.stop)

    def pip(self, **rc):
        s = PipStandIn(**rc)
        p = mock.patch.object(QS.subprocess, "call", s)
        p.start()
        self.addCleanup(p.stop)
        return s

    def fresh_env(self, h):
        st = self.state(h)
        st["venv_ok"] = False
        (h / "quickstart_state.json").write_text(json.dumps(st), encoding="utf-8")

    def image_home(self):
        """An Inkling-2 Mini folder that has the image file."""
        h, m = self.ink_home()
        for n, b in self.files.items():
            (m / n).write_bytes(b)
        self.assertTrue(QS.ink2_has_image(str(m)))
        return h, m

    def test_fresh_environment_installs_the_extras(self):
        h = self.home(checked=("quality",))
        self.fresh_env(h)
        s = self.pip()
        self.assertEqual(self.run_main(h, "--action", "none"), 0)
        edir = os.path.join(str(h), "engine")
        self.assertEqual(s.calls, [("full", [os.path.join(str(h), "venv", "Scripts", "python.exe")] + PIP
                                    + [edir + "[vision]", "huggingface_hub"])])
        self.assertTrue(self.state(h)["venv_ok"])
        self.assertNotIn("Pillow", self.out.getvalue())

    def test_the_extras_fail_then_the_command_of_034_and_one_line(self):
        h, m = self.image_home()
        self.fresh_env(h)
        s = self.pip(full=1, pil=1)
        self.assertEqual(self.run_on(h), 0)
        self.assertEqual(s.kinds(), ["full", "plain", "pil"])
        self.assertEqual(s.calls[1][1][5:], [os.path.join(str(h), "engine"), "huggingface_hub"])
        out = self.out.getvalue()
        self.assertEqual(out.count(NONE_LINE), 1)                          # at step 4 ...
        self.assertEqual(out.count(NEED_LINE), 1)                          # ... and once more at the end, after the downloads
        self.assertLess(out.index(NONE_LINE), out.index(NEED_LINE))
        self.assertEqual(out.count(COMMAND), 2)                            # both name the command with the environment's Python
        self.assertTrue(self.state(h)["venv_ok"])
        self.assertEqual(self.state(h)["models_ok"], {"inkling2": True})

    def test_full_install_then_no_check(self):
        h, m = self.image_home()
        self.fresh_env(h)
        s = self.pip()
        self.assertEqual(self.run_on(h), 0)
        self.assertEqual(s.kinds(), ["full"])                              # step 4 installed Pillow just now: no run to check it
        self.assertNotIn("Pillow", self.out.getvalue())

    def test_both_fail_is_the_old_failure(self):
        h = self.home(checked=("quality",))
        self.fresh_env(h)
        s = self.pip(full=1, plain=1)
        self.assertEqual(self.run_main(h, "--action", "none"), 4)
        self.assertEqual(s.kinds(), ["full", "plain"])
        self.assertIn("Installing the package failed. See the messages above.", self.out.getvalue())
        self.assertNotIn(NONE_LINE, self.out.getvalue())
        self.assertFalse(self.state(h)["venv_ok"])

    def test_in_place_no_pip_and_the_notice_without_pillow(self):
        h, m = self.image_home()
        s = self.pip(pil=1)
        self.assertEqual(self.run_on(h), 0)
        self.assertEqual(s.kinds(), ["pil"])                               # one short run, no pip call
        self.assertIn("The Python environment is already in place.", self.out.getvalue())
        self.assertEqual(self.out.getvalue().count(NEED_LINE), 1)

    def test_in_place_with_pillow_says_nothing(self):
        h, m = self.image_home()
        s = self.pip(pil=0)
        self.assertEqual(self.run_on(h), 0)
        self.assertEqual(s.kinds(), ["pil"])
        self.assertNotIn("Pillow", self.out.getvalue())

    def test_image_file_option_adds_pillow_once(self):
        h, m = self.image_home()
        for add in (0, 1):
            s = self.pip(pil=1, add=add)
            self.out.seek(0)
            self.out.truncate()
            self.assertEqual(self.run_on(h, "--image-file"), 0)
            self.assertEqual(s.kinds(), ["pil", "add"])
            self.assertEqual(s.calls[1][1], [os.path.join(str(h), "venv", "Scripts", "python.exe")] + PIP + ["Pillow>=10"])
            out = self.out.getvalue()
            self.assertIn(ADD_LINE, out)
            self.assertNotIn(NONE_LINE, out)
            self.assertEqual(out.count(NEED_LINE), add)                    # only when the add failed

    def test_yes_answer_gets_the_image_file_and_adds_pillow(self):
        h, m = self.ink_home()
        self.fresh_env(h)
        s = self.pip(full=1, pil=1)                                        # step 4 without the extras, then the yes answer
        self.answers = ["y"]
        self.assertEqual(self.run_on(h), 0)
        self.assertTrue(QS.ink2_has_image(str(m)))
        self.assertEqual(s.kinds(), ["full", "plain", "pil", "add"])
        self.assertIn(ADD_LINE, self.out.getvalue())

    def test_no_answer_is_not_a_request_then_the_image_file_option(self):
        h, m = self.ink_home()
        s = self.pip(pil=1)
        self.answers = ["n"]
        self.assertEqual(self.run_on(h), 0)
        self.assertEqual(s.kinds(), [])                                    # no image file: nothing to check
        self.assertNotIn("Pillow", self.out.getvalue())
        self.assertEqual(self.run_on(h, "--image-file"), 0)               # the file is fetched, then Pillow is added
        self.assertTrue(QS.ink2_has_image(str(m)))
        self.assertEqual(s.kinds(), ["pil", "add"])
        self.assertIn(ADD_LINE, self.out.getvalue())

    def test_qwen_models_are_never_checked(self):
        h = self.home(checked=("quality",))
        self.fresh_env(h)
        s = self.pip(full=1, pil=1)
        self.assertEqual(self.run_main(h, "--action", "none"), 0)
        self.assertEqual(s.kinds(), ["full", "plain"])                     # the fallback, and no check of Pillow
        self.assertEqual(self.out.getvalue().count(NONE_LINE), 1)
        for more in ([], ["--image-file"]):                                # environment in place: no run at all
            s2 = self.pip(pil=1)
            self.assertEqual(self.run_on(h, *more), 0)
            self.assertEqual(s2.calls, [])
        self.assertNotIn(NEED_LINE, self.out.getvalue())


def load_tests(loader, tests, pattern):
    """Only the tests of this file (MainTests inherits those of OfferTests and LauncherTests, which their files run)."""
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(CommandTests))
    for n in sorted(n for n in vars(MainTests) if n.startswith("test_")):
        suite.addTest(MainTests(n))
    return suite


if __name__ == "__main__":
    unittest.main()
