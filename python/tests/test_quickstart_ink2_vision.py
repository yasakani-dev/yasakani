# SPDX-License-Identifier: Apache-2.0
"""The quickstart and the image file of Inkling-2 Mini (INK2_IMAGE, the one switch of quickstart/quickstart.py).

Switch on (as shipped since 0.3.5: the published commit with the image file, and an engine zip with image input; the cases below
use a stand-in entry): a new download gets the folder of the new revision with the image file; a folder of 0.3.2 to 0.3.4 is
offered the extra files once (Enter alone, y or --yes: they are fetched, checked and put in place all or nothing, the text
weights are not fetched or read; n: text only and one line per start); a wrong file is refused and the folder stays as it was.
The context note counts the image part (the rule of yasakani/gpu.py), and is said again after the install when it changes.  Switch off (INK2_IMAGE None and ENGINE_IMAGE False, patched or as a module of its own): the launcher
is that of 0.3.4.

No GPU and no network: the launcher runs with the stand-ins of test_quickstart_web.LauncherTests, and the files come from a folder
(YASAKANI_QS_TEST_MODEL_SRC_INKLING2).
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
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


SWITCH_RE = re.compile(r"\nINK2_IMAGE = (?:None|\{.*?\}\})\n", re.S)       # the shipped value: None, or {"revision": .., "files": {..}}
ENGINE_RE = re.compile(r"\nENGINE_IMAGE = (?:True|False)\n")


def switched_module(image, engine=True):
    """quickstart.py with the switch set to ``image`` (the value of ``INK2_IMAGE = ...``; None: off) and the engine it needs
    (``ENGINE_IMAGE``: a zip with image input or not), loaded as its own module."""
    text = Path(QS.__file__).read_text(encoding="utf-8")
    assert len(SWITCH_RE.findall(text)) == 1 and len(ENGINE_RE.findall(text)) == 1
    tmp = Path(tempfile.mkdtemp(prefix="yskn_qs_sw_"))
    p = tmp / "quickstart_switched.py"
    text = SWITCH_RE.sub(lambda m: "\nINK2_IMAGE = %r\n" % (image,), text)
    p.write_text(ENGINE_RE.sub(lambda m: "\nENGINE_IMAGE = %r\n" % (engine,), text), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("quickstart_switched", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil.rmtree(str(tmp), True)
    return mod


class SwitchTests(unittest.TestCase):

    def test_on_is_0_3_5(self):
        """As shipped since 0.3.5: the published commit with the image file and an engine zip with image input."""
        self.assertTrue(QS.ENGINE_IMAGE)                                           # the engine zip of 0.3.5 reads images
        self.assertEqual((QS.QS_VERSION, QS.ENGINE_ZIP), ("0.3.5", "yasakani-0.3.5-win-x64.zip"))
        self.assertRegex(QS.ENGINE_SHA256, r"^[0-9a-f]{64}$")                     # written by the release step with the zip
        self.assertEqual(QS.INK2_IMAGE, {
            "revision": "e186e163e6d140f767a0fd1f1d31ed93e1a3e1a8",
            "files": {"inkling_2_mini_vision.yskn": (637136896, "858a5a5cacf28c1fdae0dcee23350565ac3eba77d86bc02057ce6c3bb249eabb"),
                      "yasakani.json": (3162, "e201cb7f0c1247565b07dd583ca335ded8352ef09b335344fb1481570fbe28fb"),
                      "NOTICE": (2166, "26efb9e1322dcbd3000266093f05b8718778a0c37a7da33366e0700594e282d9"),
                      "README.md": (7581, "7c3b12a4a71eb8d35e1c0bae002289be3c4d1bab44d53fa134732e2ba100326c")}})
        e = QS.MODELS["inkling2"]
        self.assertEqual((e["revision"], e["get_gb"], e["need_gb"]), ("e186e163e6d140f767a0fd1f1d31ed93e1a3e1a8", "15.9", 17))
        self.assertEqual(QS.ink2_download_bytes(QS.INK2_IMAGE), 15925534376)      # the 15 files of the folder at that commit
        self.assertEqual(QS.INK2_FILES_BYTES, (15052423172, 209688040))           # the text weights and the drafter as before
        self.assertIn("文章と画像", QS.Ui("ja").t("pick", d="4"))
        self.assertIn("text and images", QS.Ui("en").t("pick", d="4"))
        self.assertFalse(QS.parse_args([]).image_file)
        self.assertTrue(QS.parse_args(["--image-file"]).image_file)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            QS.parse_args(["--help"])
        self.assertIn("--image-file", out.getvalue())

    def test_off_is_0_3_4(self):
        """The switch off (INK2_IMAGE None, ENGINE_IMAGE False as in the 0.3.4 zip): the launcher of 0.3.4."""
        off = switched_module(None, engine=False)
        self.assertIsNone(off.INK2_IMAGE)
        self.assertFalse(off.ENGINE_IMAGE)
        e = off.MODELS["inkling2"]
        self.assertEqual((e["revision"], e["get_gb"]), ("8de37e9980397da63b050fe008493a35f1c483e9", "15.3"))
        self.assertEqual(off.ink2_download_bytes(), 15288396146)                   # the 14 files of the published folder
        self.assertEqual(off.INK2_FILES_BYTES, (15052423172, 209688040))
        for lang, line in (("ja", "  4) Inkling-2 Mini〈試験的〉 文章だけ。依頼は 1 件ずつ。GPU 24 GB 以上。ダウンロード約 15.3 GB\n"),
                           ("en", "  4) Inkling-2 Mini (experimental) text only, one request at a time, a GPU with 24 GB or more;\n"
                                  "                                  about 15.3 GB download\n")):
            text = off.Ui(lang).t("pick", d="4")
            self.assertIn(line, text)
            self.assertNotIn("画像", text)
            self.assertNotIn("image", text)
        self.assertFalse(off.parse_args([]).image_file)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            off.parse_args(["--image-file"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            off.parse_args(["--help"])
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
        with mock.patch.object(QS, "INK2_IMAGE", None), mock.patch.object(QS, "ENGINE_IMAGE", False):
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
        library reads images, as vision_plan_mib(); the launcher's own engine (ENGINE_IMAGE) does since 0.3.5 (the 0.3.4 zip did not)."""
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
        with mock.patch.object(QS, "ENGINE_IMAGE", False):                         # default: the launcher's engine pin (0.3.4)
            self.assertFalse(QS.ink2_vision_planned(str(tmp), None, environ={}))
            self.assertFalse(QS.ink2_vision_planned(str(tmp), QS.INK2_IMAGE, environ={}))
        with mock.patch.object(QS, "ENGINE_IMAGE", True):                          # 0.3.5 and later
            self.assertTrue(QS.ink2_vision_planned(str(tmp), None, environ={}))
            self.assertTrue(QS.ink2_vision_planned(str(tmp), QS.INK2_IMAGE, environ={}))
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
                                          "[Y/n]: " % gb)
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

    def test_enter_is_yes_and_no_keyboard_keeps_nothing(self):
        """Since 0.3.5 the model with image input is the standard: Enter alone fetches the file.  Without a keyboard nothing is
        fetched and nothing is kept (asked again at the next start)."""
        h, m = self.ink_home()
        before = self.snapshot(m)
        self.answers = []                                                          # the end of the input (no keyboard)
        self.assertEqual(self.run_on(h), 0)
        self.assertEqual(len(self.prompts), 1)
        self.assertNotIn("ink2_image_no", self.state(h))
        self.assertIn("For image input: START.cmd --image-file", self.out.getvalue())
        self.assertEqual(self.snapshot(m), before)
        self.assertEqual(self.hashed, [])
        self.answers = [""]                                                        # Enter alone: the default yes
        self.assertEqual(self.run_on(h), 0)
        self.assertEqual(len(self.prompts), 2)
        self.assertIn("The file for image input was checked and put in place. Images work now.", self.out.getvalue())
        self.assertTrue(QS.ink2_has_image(str(m)))
        self.assertNotIn("ink2_image_no", self.state(h))

    def test_prompts_say_enter_is_yes(self):
        for lang, want in (("ja", "Inkling-2 Mini で画像を読むには、ファイルがもう 1 つ要ります（約 0.64 GB）。"
                                  "いまダウンロードしますか？ [Y/n]: "),
                           ("en", "Image input for Inkling-2 Mini needs one more file (about 0.64 GB). Download it now? [Y/n]: ")):
            self.assertEqual(QS.Ui(lang).t("img_ask", g="0.64"), want)
        self.assertEqual("%.2f" % (sum(s for s, _ in QS.INK2_IMAGE["files"].values()) / 1e9), "0.64")

    def test_explicit_no_is_kept_and_a_newer_upload_asks_again(self):
        h, m = self.ink_home()
        for no in ("n", "N", "no"):
            self.prompts.clear()
            self.answers = [no]
            st = self.state(h)
            st.pop("ink2_image_no", None)
            (h / "quickstart_state.json").write_text(json.dumps(st), encoding="utf-8")
            self.assertEqual(self.run_on(h), 0)
            self.assertEqual(len(self.prompts), 1)
            self.assertEqual(self.state(h)["ink2_image_no"], REV, no)
            self.assertFalse(QS.ink2_has_image(str(m)))
        h2_rev = dict(entry(self.files), revision="f" * 40)                        # a newer upload asks once more
        self.prompts.clear()
        self.answers = ["n"]
        self.assertEqual(self.run_on(h, image=h2_rev), 0)
        self.assertEqual(len(self.prompts), 1)

    def test_yes_flag_answers_the_question(self):
        h, m = self.ink_home()
        self.assertEqual(self.run_on(h, "--yes"), 0)
        self.assertEqual(self.prompts, [])                                         # no waiting for input
        self.assertTrue(QS.ink2_has_image(str(m)))
        self.assertIn("The file for image input was checked and put in place.", self.out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            QS.parse_args(["--help"])
        self.assertIn("also the question about the image file of Inkling-2 Mini", " ".join(out.getvalue().split()))

    def test_yes_flag_keeps_an_earlier_no_and_the_page_update_asks_nothing(self):
        """--yes answers the question; after a kept no there is no question (START.cmd --image-file gets the file).  The run the
        page's update starts (--web-port, and --yes when the run before had it) says the line only."""
        h, m = self.ink_home()
        self.answers = ["n"]
        self.assertEqual(self.run_on(h), 0)
        self.prompts.clear()
        self.assertEqual(self.run_on(h, "--yes"), 0)
        self.assertEqual(self.prompts, [])
        self.assertFalse(QS.ink2_has_image(str(m)))
        st, out = {}, io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertFalse(QS.offer_image_part(QS.Ui("en"), st, lambda: None, "py", {}, str(h), str(m), False, False,
                                                 image=entry(self.files), environ={}, yes=True,
                                                 call=lambda *a, **k: self.fail("net")))
        self.assertEqual(st, {})
        self.assertIn("For image input: START.cmd --image-file", out.getvalue())

    def test_image_file_flag_says_why_it_does_nothing(self):
        h, m = self.ink_home()
        mine = self.tmp / "mine"                                                   # for --model-dir below: without the file
        shutil.copytree(str(m), str(mine))
        for n, b in self.files.items():                                            # the file is there already
            (m / n).write_bytes(b)
        before = self.snapshot(m)
        self.assertEqual(self.run_on(h, "--image-file"), 0)
        self.assertEqual(self.out.getvalue().count(
            "--image-file: the file for image input is already there. Nothing is downloaded."), 1)
        self.assertEqual(self.prompts, [])
        self.assertEqual(self.snapshot(m), before)
        self.assertEqual(self.run_on(h), 0)                                        # without the flag: nothing said
        self.assertNotIn("image", self.out.getvalue().lower())
        mine_before = self.snapshot(mine)
        self.assertEqual(self.run_on(h, "--model-dir", str(mine), "--skip-checks", "--image-file"), 0)
        self.assertIn("--image-file: a --model-dir folder is never changed. The file for image input is not downloaded.",
                      self.out.getvalue())
        self.assertEqual(self.snapshot(mine), mine_before)
        self.assertEqual(self.prompts, [])
        shutil.rmtree(str(h))                                                      # another model
        hq = self.home(checked=("quality",), downloaded=("quality",), model="quality")
        self.assertEqual(self.run_on(hq, "--image-file"), 0)
        self.assertIn("--image-file: the file for image input is for Inkling-2 Mini only. Nothing to do for this model.",
                      self.out.getvalue())
        self.assertEqual(self.hashed, [])
        self.assertEqual(tuple(QS.Ui("ja").t(k) for k in ("img_have", "img_not_dir", "img_not_model")),
                         ("--image-file: 画像用のファイルはもう入っています。ダウンロードはしません。",
                          "--image-file: --model-dir のフォルダーは変えません。画像用のファイルはダウンロードしません。",
                          "--image-file: 画像用のファイルは Inkling-2 Mini だけのものです。このモデルでは何もしません。"))

    def test_note_after_the_install_names_the_context_with_the_image_part(self):
        """Step 2 counts no image part for a folder without the file; once the file is in place the server loads it, so the note
        is said again with the context it then chooses (and only when that differs)."""
        used = 24564 - 19300
        card = dict(index=0, uuid="GPU-b", name="NVIDIA GeForce RTX 4090", cc=(8, 9), total=24564, used=used, driver="616.92")
        h, m = self.ink_home()
        with mock.patch.object(QS, "query_gpus", lambda run=None: [dict(card)]), \
                mock.patch.dict(os.environ, {"YASAKANI_INK2_FAST": ""}):
            self.answers = ["y"]
            self.assertEqual(self.run_on(h), 0)
        out = self.out.getvalue()
        done = out.index("The file for image input was checked and put in place.")
        note = "(12,288 tokens with what is free now; at most 32,768)"
        self.assertEqual(out.count(note), 1)
        self.assertGreater(out.index(note), done)                                  # after the install, not in step 2
        shutil.rmtree(str(h))                                                      # a card with room for both: no note at all
        h, m = self.ink_home()
        self.answers = ["y"]
        self.assertEqual(self.run_on(h), 0)                                        # the RTX 5090 of LauncherTests, 1,200 MiB used
        self.assertTrue(QS.ink2_has_image(str(m)))
        self.assertNotIn("tokens with what is free now", self.out.getvalue())
        shutil.rmtree(str(h))                                                      # the same note before and after: said once
        h, m = self.ink_home()
        card2 = dict(card, used=24564 - 16000)
        self.assertEqual({QS.estimate_ink2_context(16000, fast=True, vision=v) for v in (False, True)}, {0})
        with mock.patch.object(QS, "query_gpus", lambda run=None: [dict(card2)]), \
                mock.patch.dict(os.environ, {"YASAKANI_INK2_FAST": ""}):
            self.answers = ["y"]
            self.assertEqual(self.run_on(h), 0)
        self.assertTrue(QS.ink2_has_image(str(m)))
        self.assertEqual(self.out.getvalue().count("There is not enough free GPU memory now"), 1)

    def install_case(self):
        """A folder of 0.3.4 and a download folder with the 4 checked files (README.md too, as in the published entry)."""
        for d in ("home", "dl"):
            shutil.rmtree(str(self.tmp / d), True)
        h, m = self.ink_home()
        (m / "README.md").write_bytes(b"old readme\n")
        files = dict(self.files, **{"README.md": b"new readme\n"})
        ddir = self.tmp / "dl"
        ddir.mkdir()
        for n, b in files.items():
            (ddir / n).write_bytes(b)
        return m, ddir, entry(files), files

    def tree(self, d):
        return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}

    def test_install_is_all_or_nothing(self):
        """Every replace can fail (a file held open by another program: PermissionError on Windows): the folder is then as it was
        (the old NOTICE, README.md and yasakani.json, no image file, no file moved aside), the new files are back in the download
        folder and the error goes on (img_fail, text only)."""
        m, ddir, img, files = self.install_case()
        before, dl = self.tree(m), self.tree(ddir)
        seen = []
        QS.install_image_files(str(ddir), img, str(m), replace=lambda a, b: seen.append(1) or os.replace(a, b))
        self.assertEqual(len(seen), 7)                                             # the image file, then 3 x (aside, in)
        for n, b in files.items():
            self.assertEqual((m / n).read_bytes(), b)
        self.assertEqual([p.name for p in m.iterdir() if p.name.endswith(QS.IMAGE_OLD_SUFFIX)], [])
        self.assertFalse((m / ".cache" / "huggingface" / "download" / "yasakani.json.metadata").exists())
        self.assertTrue(QS.ink2_has_image(str(m)))
        for fail_at in range(1, len(seen) + 1):
            m, ddir, img, files = self.install_case()
            calls = []

            def failing(a, b, fail_at=fail_at):
                calls.append((os.path.basename(a), os.path.basename(b)))
                if len(calls) == fail_at:
                    raise PermissionError(13, "The process cannot access the file because it is being used by another process", a)
                os.replace(a, b)
            with self.assertRaises(PermissionError, msg=fail_at):
                QS.install_image_files(str(ddir), img, str(m), replace=failing)
            self.assertEqual(self.tree(m), before, (fail_at, calls))
            self.assertEqual(self.tree(ddir), dl, fail_at)
            self.assertFalse(QS.ink2_has_image(str(m)))

    def test_install_says_what_could_not_be_put_back(self):
        m, ddir, img, files = self.install_case()
        calls = []

        def failing(a, b):
            calls.append(1)
            if len(calls) >= 4:                                                    # README.md aside fails, then every way back
                raise PermissionError(13, "in use", a)
            os.replace(a, b)
        with self.assertRaises(OSError) as cm:
            QS.install_image_files(str(ddir), img, str(m), replace=failing)
        self.assertIn("; not put back: NOTICE, " + QS.INK2_IMAGE_FILE, str(cm.exception))   # README.md is still in place

    def test_failing_install_in_main_keeps_the_folder_and_asks_again(self):
        h, m = self.ink_home()
        before = self.tree(m)
        real = QS.install_image_files
        calls = []

        def failing(a, b):
            calls.append(1)
            if len(calls) == 3:                                                    # NOTICE was moved aside, the new one cannot go in
                raise PermissionError(13, "in use", a)
            os.replace(a, b)
        self.answers = ["y"]
        with mock.patch.object(QS, "install_image_files", lambda d, i, md: real(d, i, md, replace=failing)):
            self.assertEqual(self.run_on(h), 0)
        self.assertIn("The file for image input could not be downloaded (PermissionError: ", self.out.getvalue())
        self.assertIn("Going on with text only.", self.out.getvalue())
        self.assertEqual(self.tree(m), before)
        self.assertFalse((h / "downloads" / "inkling2_image").exists())
        self.assertNotIn("ink2_image_no", self.state(h))

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
        with mock.patch.object(QS, "INK2_IMAGE", None), mock.patch.object(QS, "ENGINE_IMAGE", False):   # the switch off (0.3.4)
            self.assertEqual(self.run_main(h, "--action", "none"), 0)
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
            with mock.patch.object(QS, "INK2_IMAGE", None), mock.patch.object(QS, "ENGINE_IMAGE", False):
                self.assertEqual(self.run_main(h, "--action", "none"), 0)          # switch off, folder with the file, 0.3.4 engine:
            self.assertNotIn("tokens with what is free now", self.out.getvalue())          # not counted (serve does not load it)
            self.out.truncate(0)
            self.out.seek(0)
            with mock.patch.object(QS, "INK2_IMAGE", None), mock.patch.object(QS, "ENGINE_IMAGE", True):
                self.assertEqual(self.run_main(h, "--action", "none"), 0)          # switch off, an engine with image input: counted
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
                        mock.patch.dict(os.environ, {"YASAKANI_INK2_FAST": ""}), mock.patch.object(QS, "ENGINE_IMAGE", engine), \
                        mock.patch.object(QS, "INK2_IMAGE", None):
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
