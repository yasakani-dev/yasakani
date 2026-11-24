# SPDX-License-Identifier: Apache-2.0
"""The quickstart and the image file of Inkling-2 Mini (INK2_IMAGE, the one switch of quickstart/quickstart.py).

Switch off (None, as shipped before the upload): the launcher is that of 0.3.4.  Switch on (a stand-in entry): a new download gets
the folder of the new revision with the image file; a folder of 0.3.2 to 0.3.4 is offered the extra files once (yes: they are
fetched, checked and put in place, the text weights are not fetched or read; no: text only and one line per start); a wrong file
is refused and the folder stays as it was.  The context note counts the image part (the rule of yasakani/gpu.py).

No GPU and no network: the launcher runs with the stand-ins of test_quickstart_web.LauncherTests, and the files come from a folder
(YASAKANI_QS_TEST_MODEL_SRC_INKLING2).
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import types
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from tests import test_quickstart_web as W
from yasakani import gpu as G
from yasakani import inkling2 as ink
from yasakani import inkling2_vision as V

QS = W.QS
SRC_VAR = "YASAKANI_QS_TEST_MODEL_SRC_INKLING2"
REV = "0123456789abcdef0123456789abcdef01234567"
IMG = b"\x01image-part\x02" * 50
OLD_W = {"file": "inkling_2_mini_quality.yskn", "size": 11, "sha256": "aa" * 32}
OLD_D = {"file": "mtp_d4.yskn", "size": 7, "sha256": "bb" * 32}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def new_files(img=IMG, weights=None):
    """The files of the stand-in revision: the image file, the new yasakani.json and NOTICE.  -> {name: bytes}."""
    man = {"architecture": "inkling2", "weights": dict(weights or OLD_W), "drafter": dict(OLD_D),
           "vision": {"file": QS.INK2_IMAGE_FILE, "format": "yskn", "size": len(IMG), "sha256": sha(IMG)}}
    return {QS.INK2_IMAGE_FILE: img, "yasakani.json": json.dumps(man, indent=1).encode("utf-8"), "NOTICE": b"new notice\n"}


def entry(files=None):
    """The INK2_IMAGE value for the stand-in revision (the pinned values of the right files)."""
    files = files or new_files()
    return {"revision": REV, "files": {n: (len(b), sha(b)) for n, b in files.items()}}


def switched_module(image, engine=True):
    """quickstart.py with the edit of the switch (the line ``INK2_IMAGE = None``) and of the engine it needs (``ENGINE_IMAGE =
    False``: a zip with image input), loaded as its own module."""
    text = Path(QS.__file__).read_text(encoding="utf-8")
    assert text.count("\nINK2_IMAGE = None\n") == 1 and text.count("\nENGINE_IMAGE = False\n") == 1
    tmp = Path(tempfile.mkdtemp(prefix="yskn_qs_sw_"))
    p = tmp / "quickstart_switched.py"
    text = text.replace("\nINK2_IMAGE = None\n", "\nINK2_IMAGE = %r\n" % (image,))
    p.write_text(text.replace("\nENGINE_IMAGE = False\n", "\nENGINE_IMAGE = %r\n" % (engine,)), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("quickstart_switched", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil.rmtree(str(tmp), True)
    return mod


class SwitchTests(unittest.TestCase):

    def test_off_is_0_3_4(self):
        self.assertIsNone(QS.INK2_IMAGE)
        self.assertFalse(QS.ENGINE_IMAGE)                                          # the pinned 0.3.4 engine zip reads no images
        self.assertEqual(QS.ENGINE_SHA256, "831576fe908c0dc52494aae6534d2a45e27f310b3ceaba9bbc0b1e65e4a8a7fc")
        e = QS.MODELS["inkling2"]
        self.assertEqual((e["revision"], e["get_gb"]), ("8de37e9980397da63b050fe008493a35f1c483e9", "15.3"))
        self.assertEqual(QS.ink2_download_bytes(), 15288396146)                    # the 14 files of the published folder
        self.assertEqual(QS.INK2_FILES_BYTES, (15052423172, 209688040))
        for lang in ("ja", "en"):
            text = QS.Ui(lang).t("pick", d="4")
            self.assertNotIn("画像", text)
            self.assertNotIn("image", text)
        self.assertFalse(QS.parse_args([]).image_file)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            QS.parse_args(["--image-file"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            QS.parse_args(["--help"])
        self.assertNotIn("image", out.getvalue())

    def test_one_edit_turns_it_on(self):
        img_entry = {"revision": REV, "files": {QS.INK2_IMAGE_FILE: (637136896, "85" * 32), "yasakani.json": (3000, "11" * 32),
                                                "NOTICE": (2300, "22" * 32)}}
        on = switched_module(img_entry)
        e = on.MODELS["inkling2"]
        self.assertEqual((e["revision"], e["get_gb"], e["need_gb"]), (REV, "15.9", 17))
        self.assertGreaterEqual(e["need_gb"], on.ink2_download_bytes(img_entry) / 1e9 + 1)
        self.assertEqual(on.download_command("py", "inkling2", "D:/m"), ["py", "-c", on.DOWNLOAD_CODE, e["repo"], "D:/m", REV])
        self.assertIn("  4) Inkling-2 Mini〈試験的〉 文章と画像。依頼は 1 件ずつ。GPU 24 GB 以上。ダウンロード約 15.9 GB\n",
                      on.Ui("ja").t("pick", d="4"))
        self.assertIn("  4) Inkling-2 Mini (experimental) text and images, one request at a time, a GPU with 24 GB or more;\n"
                      "                                  about 15.9 GB download\n", on.Ui("en").t("pick", d="4"))
        self.assertTrue(on.parse_args(["--image-file"]).image_file)
        tmp = Path(tempfile.mkdtemp(prefix="yskn_qs_fetch_"))                     # fetch_model creates the folder it is given
        self.addCleanup(shutil.rmtree, str(tmp), True)
        mdir = str(tmp / "m")
        calls, out = [], io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(on.fetch_model(on.Ui("en"), "py", {}, "inkling2", mdir, call=lambda c, env=None: calls.append(c) or 0,
                                            environ={}), 0)
        self.assertEqual(calls, [on.download_command("py", "inkling2", mdir)])
        self.assertIn("(about 15.9 GB", out.getvalue())

    def test_new_download_brings_the_image_file(self):
        on = switched_module(entry())
        tmp = Path(tempfile.mkdtemp(prefix="yskn_qs_new_"))
        self.addCleanup(shutil.rmtree, str(tmp), True)
        src = tmp / "src"                                                         # the folder of the new revision
        src.mkdir()
        for n, b in new_files().items():
            (src / n).write_bytes(b)
        (src / OLD_W["file"]).write_bytes(b"w" * OLD_W["size"])
        dst = tmp / "models" / "inkling2"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(on.fetch_model(on.Ui("en"), "py", {}, "inkling2", str(dst), call=lambda *a, **k: self.fail("net"),
                                            environ={SRC_VAR: str(src)}), 0)
        self.assertTrue(on.ink2_has_image(str(dst)))
        self.assertTrue(on.ink2_vision_planned(str(tmp / "not_yet"), on.INK2_IMAGE, environ={}))    # planned before the download
        self.assertFalse(QS.ink2_vision_planned(str(tmp / "not_yet"), QS.INK2_IMAGE, environ={}))   # switch off: not
        self.assertFalse(on.ink2_vision_planned(str(tmp / "not_yet"), on.INK2_IMAGE, environ={}, engine=False))  # no image engine

    def test_switch_needs_an_engine_with_image_input(self):
        """INK2_IMAGE on with the engine pin of a zip without image input would download a file nothing reads."""
        if QS.INK2_IMAGE:
            self.assertTrue(QS.ENGINE_IMAGE)
        self.assertTrue(switched_module(entry()).ENGINE_IMAGE)


    def test_entry_shape(self):
        """The switch's value, once filled in (and the stand-in): what the launcher needs of it."""
        import re
        for e in [x for x in (QS.INK2_IMAGE,) if x] + [entry()]:
            self.assertRegex(e["revision"], r"^[0-9a-f]{40}$")
            self.assertNotEqual(e["revision"], "8de37e9980397da63b050fe008493a35f1c483e9")
            self.assertIn(QS.INK2_IMAGE_FILE, e["files"])
            self.assertIn("yasakani.json", e["files"])
            for name, (size, digest) in e["files"].items():
                self.assertEqual(os.path.basename(name), name)
                self.assertTrue(isinstance(size, int) and size > 0, name)
                self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", digest), name)
        if QS.INK2_IMAGE:                                                          # the file of tools/convert/inkling2_vision_convert.py
            self.assertEqual(QS.INK2_IMAGE["files"][QS.INK2_IMAGE_FILE],
                             (637136896, "858a5a5cacf28c1fdae0dcee23350565ac3eba77d86bc02057ce6c3bb249eabb"))

    def test_image_code_fetches_only_the_listed_files_at_the_revision(self):
        """The code that runs in the private environment, with a stand-in huggingface_hub."""
        import subprocess
        import sys
        tmp = Path(tempfile.mkdtemp(prefix="yskn_qs_ic_"))
        self.addCleanup(shutil.rmtree, str(tmp), True)
        (tmp / "huggingface_hub.py").write_text(
            "import json, sys\n"
            "def hf_hub_download(**kw):\n"
            "    open(sys.argv[2] + '.log', 'a').write(json.dumps(kw) + '\\n')\n", encoding="utf-8")
        cmd = QS.image_files_command(sys.executable, entry(), str(tmp / "d"))
        r = subprocess.run(cmd, cwd=str(tmp), capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.assertEqual(r.returncode, 0, r.stderr)
        got = [json.loads(ln) for ln in (tmp / "d.log").read_text().splitlines()]
        self.assertEqual(got, [{"repo_id": QS.MODELS["inkling2"]["repo"], "filename": n, "revision": REV, "local_dir": str(tmp / "d")}
                               for n in sorted(new_files())])


class MemoryRuleTests(unittest.TestCase):
    """The launcher's copy of the image part's memory (gpu.py ink2_vision_mib with the defaults of chat and serve)."""

    def test_same_as_package(self):
        self.assertEqual((QS.INK2_VISION_WEIGHTS_BYTES, QS.INK2_VISION_SCRATCH_BYTES, QS.INK2_VISION_SCRATCH_BYTES_PER_TOKEN,
                          QS.INK2_VISION_FEATURE_BYTES_PER_ROW),
                         (G.INK2_VISION_WEIGHTS_BYTES, G.INK2_VISION_SCRATCH_BYTES, G.INK2_VISION_SCRATCH_BYTES_PER_TOKEN,
                          G.INK2_VISION_FEATURE_BYTES_PER_ROW))
        cap, req = ink.vision_limits()
        self.assertEqual((cap, req), (QS.INK2_VISION_IMAGE_TOKENS, QS.INK2_VISION_REQUEST_FACTOR * QS.INK2_VISION_IMAGE_TOKENS))
        self.assertEqual((V.DEFAULT_IMAGE_TOKENS, V.REQUEST_IMAGE_TOKENS_FACTOR),
                         (QS.INK2_VISION_IMAGE_TOKENS, QS.INK2_VISION_REQUEST_FACTOR))
        vmib = G.ink2_vision_mib(cap, req, weights=True)                         # a separate tower file: its weights count
        self.assertEqual(QS.ink2_vision_mib(), vmib)
        self.assertEqual(QS.INK2_VISION_ENV, ink.VISION_ENV)
        files = sum(QS.INK2_FILES_BYTES)
        for fast in (False, True):
            per = G.ink2_bytes_per_position(ink.bytes_per_position({"text_config": {}}, mxfp8=fast), True, fast=fast)
            for free in range(0, 40001, 13):
                want = G.ink2_context(free, files, per, 32768, 512, fast=fast, vision_mib=vmib)
                self.assertEqual(QS.estimate_ink2_context(free, fast=fast, vision=True), want, (fast, free))
                self.assertEqual(QS.estimate_ink2_context(free, fast=fast),
                                 G.ink2_context(free, files, per, 32768, 512, fast=fast), (fast, free))

    def test_planned_follows_the_folder_and_the_mode(self):
        tmp = Path(tempfile.mkdtemp(prefix="yskn_qs_pl_"))
        self.addCleanup(shutil.rmtree, str(tmp), True)
        (tmp / "yasakani.json").write_text(json.dumps({"architecture": "inkling2"}), encoding="utf-8")
        self.assertFalse(QS.ink2_vision_planned(str(tmp), entry(), environ={}))   # a folder without it: not counted
        (tmp / "yasakani.json").write_bytes(new_files()["yasakani.json"])
        self.assertFalse(QS.ink2_vision_planned(str(tmp), None, environ={}))     # named, but the file is not there
        (tmp / QS.INK2_IMAGE_FILE).write_bytes(IMG)
        for env, want in (({}, True), ({"YASAKANI_INK2_VISION": "auto"}, True), ({"YASAKANI_INK2_VISION": "on"}, True),
                          ({"YASAKANI_INK2_VISION": " OFF "}, False)):
            self.assertEqual(QS.ink2_vision_planned(str(tmp), None, environ=env, engine=True), want, env)
            self.assertEqual(ink.vision_mode_of(None, env) != "off", want, env)

    def test_planned_asks_the_engine_as_the_package_does(self):
        """The folder has the image file (made by hand, e.g. with tools/convert/inkling2_vision_convert.py): counted only when the
        library reads images, as vision_plan_mib(); the launcher's own engine (ENGINE_IMAGE, the 0.3.4 zip) does not."""
        tmp = Path(tempfile.mkdtemp(prefix="yskn_qs_eng_"))
        self.addCleanup(shutil.rmtree, str(tmp), True)
        (tmp / "yasakani.json").write_bytes(new_files()["yasakani.json"])
        (tmp / QS.INK2_IMAGE_FILE).write_bytes(IMG)
        self.assertTrue(QS.ink2_has_image(str(tmp)))
        pack = types.SimpleNamespace(has_vision=True, vision_entry={"file": QS.INK2_IMAGE_FILE})
        for mode in ("", "auto", "on", "off"):
            env = {"YASAKANI_INK2_VISION": mode}
            for engine in (False, True):
                want = ink.vision_plan_mib(pack, ink.vision_mode_of(None, env), has_library=engine) > 0
                self.assertEqual(QS.ink2_vision_planned(str(tmp), None, environ=env, engine=engine), want, (mode, engine))
                self.assertEqual(QS.ink2_vision_planned(str(tmp), entry(), environ=env, engine=engine), want, (mode, engine))
        self.assertFalse(QS.ink2_vision_planned(str(tmp), None, environ={}))      # default: the launcher's engine pin
        self.assertFalse(QS.ink2_vision_planned(str(tmp), QS.INK2_IMAGE, environ={}))
        with mock.patch.object(QS, "ENGINE_IMAGE", True):
            self.assertTrue(QS.ink2_vision_planned(str(tmp), None, environ={}))
        (tmp / QS.INK2_IMAGE_FILE).unlink()                                         # no file: never, whatever the engine
        for engine in (False, True):
            self.assertFalse(QS.ink2_vision_planned(str(tmp), None, environ={}, engine=engine))


class OfferTests(W.LauncherTests):
    """main() with an Inkling-2 Mini folder of 0.3.4 (no image file) and the switch on (patched)."""

    def setUp(self):
        super().setUp()
        self.files = new_files()
        self.src = self.tmp / "hf_rev"                                             # only the new files: the text weights are not there
        self.src.mkdir()
        for n, b in self.files.items():
            (self.src / n).write_bytes(b)
        self.answers = []
        self.prompts = []
        for name, value in (("ask", self.answer),):
            p = mock.patch.object(QS, name, value)
            p.start()
            self.addCleanup(p.stop)
        e = mock.patch.dict(os.environ, {SRC_VAR: str(self.src), "YASAKANI_INK2_VISION": ""})
        e.start()
        self.addCleanup(e.stop)
        self.hashed = []
        real = QS.sha256_of
        h = mock.patch.object(QS, "sha256_of", lambda p: self.hashed.append(os.path.basename(p)) or real(p))
        h.start()
        self.addCleanup(h.stop)

    def answer(self, prompt, default="", eof=None):
        self.prompts.append(prompt)
        if not self.answers:
            return default if eof is None else eof
        a = self.answers.pop(0)
        return default if a == "" else a

    def ink_home(self):
        h = self.home(checked=("inkling2",), downloaded=("inkling2",), model="inkling2")
        m = h / "models" / "inkling2"
        (m / "yasakani.json").write_text(json.dumps({"architecture": "inkling2", "weights": OLD_W, "drafter": OLD_D}),
                                         encoding="utf-8")
        (m / "NOTICE").write_bytes(b"old notice\n")
        (m / OLD_W["file"]).write_bytes(b"w" * OLD_W["size"])
        (m / OLD_D["file"]).write_bytes(b"d" * OLD_D["size"])
        note = m / ".cache" / "huggingface" / "download"
        note.mkdir(parents=True)
        (note / "yasakani.json.metadata").write_text("old")
        (note / (OLD_W["file"] + ".metadata")).write_text("w")
        return h, m

    def run_on(self, h, *more, image=None, engine=True):
        """main() with the switch on and (as the switch needs) an engine zip with image input."""
        self.out.truncate(0)
        self.out.seek(0)
        with mock.patch.object(QS, "INK2_IMAGE", image or entry(self.files)), mock.patch.object(QS, "ENGINE_IMAGE", engine):
            return self.run_main(h, "--action", "none", *more)

    def snapshot(self, m):
        return {p.name: p.read_bytes() for p in m.iterdir() if p.is_file()}

    def test_yes_fetches_checks_and_keeps_the_text_weights(self):
        h, m = self.ink_home()
        w_stat = os.stat(m / OLD_W["file"])
        self.answers = ["y"]
        self.assertEqual(self.run_on(h), 0)
        out = self.out.getvalue()
        self.assertEqual(len(self.prompts), 1, self.prompts)
        gb = "%.2f" % (sum(len(b) for b in self.files.values()) / 1e9)
        self.assertEqual(self.prompts[0], "Image input for Inkling-2 Mini needs one more file (about %s GB). Download it now? "
                                          "[y/N]: " % gb)
        self.assertIn("The file for image input was checked and put in place. Images work now.", out)
        for n, b in self.files.items():
            self.assertEqual((m / n).read_bytes(), b, n)                           # the new manifest and NOTICE replace the old ones
        self.assertTrue(QS.ink2_has_image(str(m)))
        st2 = os.stat(m / OLD_W["file"])
        self.assertEqual((st2.st_mtime_ns, st2.st_size), (w_stat.st_mtime_ns, w_stat.st_size))   # not fetched again
        self.assertEqual(sorted(self.hashed), sorted(self.files))                 # only the new files were read
        self.assertFalse((m / ".cache" / "huggingface" / "download" / "yasakani.json.metadata").exists())
        self.assertTrue((m / ".cache" / "huggingface" / "download" / (OLD_W["file"] + ".metadata")).exists())
        self.assertFalse((h / "downloads" / "inkling2_image").exists())
        self.assertNotIn("ink2_image_no", self.state(h))
        self.assertEqual(self.state(h)["models_ok"], {"inkling2": True})          # the checks are not run again
        # the next start: nothing asked, nothing said about images
        self.prompts.clear()
        self.assertEqual(self.run_on(h), 0)
        self.assertEqual(self.prompts, [])
        self.assertNotIn("image input", self.out.getvalue().lower())

    def test_no_is_kept_and_said_in_one_line(self):
        h, m = self.ink_home()
        before = self.snapshot(m)
        self.answers = ["n"]
        self.assertEqual(self.run_on(h), 0)
        self.assertEqual(len(self.prompts), 1)
        self.assertEqual(self.snapshot(m), before)
        self.assertEqual(self.state(h)["ink2_image_no"], REV)
        line = "Inkling-2 Mini runs with text only. For image input: START.cmd --image-file (about "
        self.assertEqual(self.out.getvalue().count(line), 1)
        for _ in range(2):                                                         # later starts: no question, the one line
            self.prompts.clear()
            self.assertEqual(self.run_on(h), 0)
            self.assertEqual(self.prompts, [])
            self.assertEqual(self.out.getvalue().count(line), 1)
            self.assertEqual(sum("image" in ln.lower() for ln in self.out.getvalue().splitlines()), 1)
            self.assertEqual(self.snapshot(m), before)
        self.assertEqual(self.hashed, [])
        self.assertEqual(self.run_on(h, "--image-file"), 0)                       # asks again by itself: no question
        self.assertEqual(self.prompts, [])
        self.assertTrue(QS.ink2_has_image(str(m)))
        self.assertNotIn("ink2_image_no", self.state(h))

    def test_enter_is_no_and_no_keyboard_keeps_nothing(self):
        h, m = self.ink_home()
        self.answers = []                                                          # the end of the input (no keyboard)
        self.assertEqual(self.run_on(h), 0)
        self.assertEqual(len(self.prompts), 1)
        self.assertNotIn("ink2_image_no", self.state(h))
        self.assertIn("For image input: START.cmd --image-file", self.out.getvalue())
        self.answers = [""]                                                        # Enter alone: the default no
        self.assertEqual(self.run_on(h), 0)
        self.assertEqual(self.state(h)["ink2_image_no"], REV)
        self.assertFalse(QS.ink2_has_image(str(m)))
        h2_rev = dict(entry(self.files), revision="f" * 40)                        # a newer upload asks once more
        self.prompts.clear()
        self.answers = ["n"]
        self.assertEqual(self.run_on(h, image=h2_rev), 0)
        self.assertEqual(len(self.prompts), 1)

    def test_wrong_file_is_refused_and_text_only_goes_on(self):
        h, m = self.ink_home()
        before = self.snapshot(m)
        (self.src / QS.INK2_IMAGE_FILE).write_bytes(IMG[:-1] + b"X")               # same size, other bytes
        self.answers = ["y"]
        self.assertEqual(self.run_on(h), 0)
        out = self.out.getvalue()
        self.assertIn("The file for image input differs from the published values and was not used (%s: SHA-256 "
                      % QS.INK2_IMAGE_FILE, out)
        self.assertIn("Going on with text only.", out)
        self.assertEqual(self.snapshot(m), before)
        self.assertFalse((h / "downloads" / "inkling2_image").exists())
        self.assertNotIn("ink2_image_no", self.state(h))                           # a failure is not a no: asked again next time
        self.assertEqual(self.run_on(h), 0)
        self.assertEqual(len(self.prompts), 2)

    def test_manifest_of_other_weights_is_refused(self):
        h, m = self.ink_home()
        before = self.snapshot(m)
        files = new_files(weights=dict(OLD_W, sha256="cc" * 32))                   # the new folder names other text weights
        for n, b in files.items():
            (self.src / n).write_bytes(b)
        self.answers = ["y"]
        self.assertEqual(self.run_on(h, image=entry(files)), 0)
        self.assertIn("(yasakani.json: the weights differ from this folder's)", self.out.getvalue())
        self.assertEqual(self.snapshot(m), before)

    def test_offline_and_disk_full_leave_the_folder(self):
        h, m = self.ink_home()
        before = self.snapshot(m)
        st, out = {}, io.StringIO()
        calls = []
        with contextlib.redirect_stdout(out):
            got = QS.offer_image_part(QS.Ui("en"), st, lambda: None, "py", {}, str(h), str(m), True, True, image=entry(self.files),
                                      call=lambda c, env=None: calls.append(env) or 1, environ={})
        self.assertFalse(got)
        self.assertEqual(len(calls), 2)                                            # once more without Xet, as the model download
        self.assertIn("The file for image input could not be downloaded (exit code 1). Going on with text only.", out.getvalue())
        self.assertEqual(self.snapshot(m), before)
        Usage = namedtuple("Usage", "total used free")
        with contextlib.redirect_stdout(out), \
                mock.patch.object(QS.shutil, "disk_usage", lambda p: Usage(1 << 40, 1 << 40, 100 << 20)):
            self.assertFalse(QS.offer_image_part(QS.Ui("en"), st, lambda: None, "py", {}, str(h), str(m), True, True,
                                                 image=entry(self.files), call=lambda *a, **k: self.fail("net"), environ={}))
        self.assertIn("Not enough free disk space for the file for image input", out.getvalue())
        self.assertEqual(self.snapshot(m), before)

    def test_no_question_when_nobody_looks(self):
        """A run started by the page's update (the page waits for the server): the line only, nothing kept."""
        h, m = self.ink_home()
        st, out = {}, io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertFalse(QS.offer_image_part(QS.Ui("ja"), st, lambda: None, "py", {}, str(h), str(m), False, False,
                                                 image=entry(self.files), environ={}))
        self.assertEqual(self.prompts, [])
        self.assertEqual(st, {})
        self.assertIn("Inkling-2 Mini は文章だけで使います。画像も使う時は START.cmd --image-file（約 ", out.getvalue())

    def test_switch_off_says_nothing_and_keeps_the_folder(self):
        h, m = self.ink_home()
        before = self.snapshot(m)
        self.out.truncate(0)
        self.out.seek(0)
        self.assertEqual(self.run_main(h, "--action", "none"), 0)                  # QS.INK2_IMAGE is None here
        self.assertEqual(self.prompts, [])
        self.assertNotIn("image", self.out.getvalue().lower())
        self.assertEqual(self.snapshot(m), before)
        self.assertEqual(self.hashed, [])

    def test_note_counts_the_image_part_when_the_folder_has_it(self):
        """The image part (about 728 MiB) takes room of the context: 300 MiB above the full context without it, 12,288 with it."""
        used = 24564 - 19300
        card = dict(index=0, uuid="GPU-b", name="NVIDIA GeForce RTX 4090", cc=(8, 9), total=24564, used=used, driver="616.92")
        without, with_img = (QS.estimate_ink2_context(19300, fast=True, vision=v) for v in (False, True))
        self.assertEqual((without, with_img), (32768, 12288))
        h, m = self.ink_home()
        with mock.patch.object(QS, "query_gpus", lambda run=None: [dict(card)]), \
                mock.patch.dict(os.environ, {"YASAKANI_INK2_FAST": ""}):
            self.answers = ["n"]
            self.assertEqual(self.run_on(h), 0)
            self.assertNotIn("tokens with what is free now", self.out.getvalue())          # the full context: no note
            for n, b in self.files.items():
                (m / n).write_bytes(b)
            self.assertEqual(self.run_on(h), 0)
            self.assertIn("(%s tokens with what is free now" % "{:,}".format(with_img), self.out.getvalue())
            self.out.truncate(0)
            self.out.seek(0)
            self.assertEqual(self.run_main(h, "--action", "none"), 0)              # switch off, folder with the file, 0.3.4 engine:
            self.assertNotIn("tokens with what is free now", self.out.getvalue())          # not counted (serve does not load it)
            self.out.truncate(0)
            self.out.seek(0)
            with mock.patch.object(QS, "ENGINE_IMAGE", True):                       # switch off, an engine with image input: counted
                self.assertEqual(self.run_main(h, "--action", "none"), 0)
            self.assertIn("(%s tokens with what is free now" % "{:,}".format(with_img), self.out.getvalue())

    def test_switch_off_note_is_0_3_4(self):
        """INK2_IMAGE None, a folder without the image file: the note is the one of 0.3.4 (999b5e2), on every engine."""
        h, m = self.ink_home()
        for used, want in ((5000, None), (5264, None), (6000, "12,288"), (9000, 0)):
            card = dict(index=0, uuid="GPU-b", name="NVIDIA GeForce RTX 4090", cc=(8, 9), total=24564, used=used, driver="616.92")
            for engine in (False, True):
                self.out.truncate(0)
                self.out.seek(0)
                with mock.patch.object(QS, "query_gpus", lambda run=None: [dict(card)]), \
                        mock.patch.dict(os.environ, {"YASAKANI_INK2_FAST": ""}), mock.patch.object(QS, "ENGINE_IMAGE", engine):
                    self.assertEqual(self.run_main(h, "--action", "none"), 0)
                out = self.out.getvalue()
                self.assertNotIn("image", out.lower())
                if want is None:
                    self.assertNotIn("context length", out)
                elif want:
                    self.assertIn("The context length of Inkling-2 Mini is chosen from the free GPU memory each time it starts "
                                  "(%s tokens with what is free now; at most 32,768)" % want, out)
                else:
                    self.assertIn("There is not enough free GPU memory now (9000 MiB in use)", out)


def load_tests(loader, tests, pattern):
    """Only the tests of this file (OfferTests inherits those of LauncherTests, which test_quickstart_web runs)."""
    suite = unittest.TestSuite()
    for cls in (SwitchTests, MemoryRuleTests):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    for n in sorted(n for n in vars(OfferTests) if n.startswith("test_")):
        suite.addTest(OfferTests(n))
    return suite


if __name__ == "__main__":
    unittest.main()
