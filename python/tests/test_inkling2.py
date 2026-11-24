# SPDX-License-Identifier: Apache-2.0
"""Inkling-2 Mini path: converter (published and quality schemes), the .yskn file of version 2, manifest and verify, chat
template, output stream, sampling, the engine on its host backend, the server, and the dispatch of the commands.

Needs (else the tests skip):
  YASAKANI_TEST_INK2_SOURCE  the downloaded hf-internal-testing/tiny-inkling2-mini folder (config.json, model.safetensors, tokenizer)
  YASAKANI_TEST_INK2_DLL     the host build of the Inkling-2 engine library (build/inkling2/ink2_host.dll of src/inkling2/build.cmd
                             host; only for the engine and server tests)
The engine tests run on the CPU (about 20 s each class).
"""
import argparse
import http.client
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import support  # noqa: F401  (puts the package on the path)
from yasakani import inkling2 as ink
from yasakani._native import YasakaniError

HERE = Path(__file__).resolve().parent
CONVERT_DIR = HERE.parent.parent / "tools" / "convert"
if not (CONVERT_DIR / "inkling2_convert.py").is_file():
    # the engine zip holds the package and its tests without tools/: these tests need the converter of the source tree
    raise unittest.SkipTest("tools/convert is not beside the package (run these tests from the source tree)")
sys.path.insert(0, str(CONVERT_DIR))
import inkling2_convert as conv  # noqa: E402

SOURCE = os.environ.get("YASAKANI_TEST_INK2_SOURCE")
DLL = os.environ.get("YASAKANI_TEST_INK2_DLL")
HAVE_SOURCE = bool(SOURCE) and (Path(SOURCE) / "model.safetensors").is_file() and (Path(SOURCE) / "tokenizer.json").is_file()
HAVE_DLL = bool(DLL) and Path(DLL).is_file()
NEED_SOURCE = "set YASAKANI_TEST_INK2_SOURCE to the downloaded tiny-inkling2-mini folder"
NEED_DLL = "set YASAKANI_TEST_INK2_DLL to the host build of the Inkling-2 engine library"

_FOLDER = {}


def write_fake_drafter(path: Path) -> None:
    """A small safetensors file in the place of the model's drafter (the tiny checkpoint has none; the package only places,
    lists and checks the file, and the engine wrapper loads it)."""
    data = np.arange(8, dtype=np.float32).tobytes()
    head = json.dumps({"mtp.fc.weight": {"dtype": "F32", "shape": [2, 4], "data_offsets": [0, len(data)]}}).encode()
    head += b" " * (-len(head) % 8)
    path.write_bytes(struct.pack("<Q", len(head)) + head + data)


def tiny_source() -> Path:
    """The tiny checkpoint folder (hard links) with a stand-in drafter file next to its weights."""
    if "src" not in _FOLDER:
        d = Path(tempfile.mkdtemp(prefix="ink2_src_"))

        def link(a, b):
            try:
                os.link(a, b)
            except OSError:
                shutil.copy2(a, b)
        shutil.copytree(SOURCE, d / "src", copy_function=link)
        write_fake_drafter(d / "src" / conv.DRAFTER_FILE)
        _FOLDER["src"] = d
    return _FOLDER["src"] / "src"


def tiny_folder() -> Path:
    """The tiny checkpoint converted once per test run (hard link of the weights where possible)."""
    if "dir" not in _FOLDER:
        d = Path(tempfile.mkdtemp(prefix="ink2_tiny_"))
        conv.convert(tiny_source(), d / "tiny", repo="hf-internal-testing/tiny-inkling2-mini",
                     revision=conv.hf_metadata(Path(SOURCE), "model.safetensors")[0] or "test")
        _FOLDER["dir"] = d
    return _FOLDER["dir"] / "tiny"


def tiny_quality_folder() -> Path:
    """The tiny checkpoint converted once per test run with --scheme quality (no reference run: selfcheck.json has no cases)."""
    if "qdir" not in _FOLDER:
        d = Path(tempfile.mkdtemp(prefix="ink2_tinyq_"))
        conv.convert(tiny_source(), d / "tinyq", repo="hf-internal-testing/tiny-inkling2-mini",
                     revision=conv.hf_metadata(Path(SOURCE), "model.safetensors")[0] or "test", scheme="quality", threads=4)
        _FOLDER["qdir"] = d
    return _FOLDER["qdir"] / "tinyq"


def tearDownModule():
    for k in ("dir", "qdir", "src"):
        d = _FOLDER.pop(k, None)
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class ConverterTests(unittest.TestCase):
    def test_manifest_and_verify(self):
        root = tiny_folder()
        meta = json.loads((root / "yasakani.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["architecture"], "inkling2")
        self.assertEqual(meta["source"]["repository"], "hf-internal-testing/tiny-inkling2-mini")
        self.assertEqual(meta["weights"], dict(file="model.safetensors", format="hf-safetensors",
                                               note="published file, unchanged; the image tower is read from it"))
        self.assertEqual(meta["stop_token_ids"], [200006])
        self.assertEqual(meta["drafter"]["file"], "mtp.safetensors")
        self.assertEqual(meta["sampling"], dict(temperature=1.0, top_p=1.0, top_k=0))
        for name in ("model.safetensors", "config.json", "tokenizer.json", "chat_template.jinja", "NOTICE", "selfcheck.json"):
            self.assertIn(name, meta["sha256"])
            self.assertEqual(meta["sizes"][name], (root / name).stat().st_size)
        pack = ink.Ink2Pack.load(root)
        self.assertTrue(ink.is_inkling2(root))
        self.assertEqual(pack.weights.name, "model.safetensors")
        self.assertTrue(all(pack.verify().values()))

    def test_verify_finds_a_changed_file(self):
        src = tiny_folder()
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "copy"
            root.mkdir()
            for n in json.loads((src / "yasakani.json").read_text(encoding="utf-8"))["sha256"]:
                if n != "model.safetensors":
                    shutil.copyfile(src / n, root / n)
            shutil.copyfile(src / "yasakani.json", root / "yasakani.json")
            try:
                os.link(src / "model.safetensors", root / "model.safetensors")
            except OSError:
                shutil.copyfile(src / "model.safetensors", root / "model.safetensors")
            p = root / "chat_template.jinja"
            p.write_bytes(p.read_bytes().replace(b"Thinking effort level", b"Thinking effort Level"))   # same size, other bytes
            res = ink.Ink2Pack.load(root).verify()
            self.assertEqual([n for n, ok in res.items() if not ok], ["chat_template.jinja"])

    def test_weights_entry_names_the_file(self):
        """A later weights format changes only the weights entry: an unknown format is refused with a clear error."""
        src = tiny_folder()
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "f"
            shutil.copytree(src, root, ignore=shutil.ignore_patterns("model.safetensors"))
            (root / "small.bin").write_bytes(b"x")
            meta = json.loads((root / "yasakani.json").read_text(encoding="utf-8"))
            meta["weights"] = {"file": "small.bin", "format": "yskn-ink2-v1"}
            (root / "yasakani.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "weights format 'yskn-ink2-v1' is not supported"):
                ink.Ink2Pack.load(root)

    def test_not_an_inkling2_folder(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(ink.is_inkling2(t))
            (Path(t) / "yasakani.json").write_text(json.dumps({"model_id": "qwen3.8-27b"}), encoding="utf-8")
            self.assertFalse(ink.is_inkling2(t))


# greedy tokens of the PyTorch reference implementation of the model on the decoded weights of the tiny checkpoint converted with
# --scheme quality (official prompt; dense matrices rounded to bfloat16 as the engine is given them)
TINY_QUALITY_TOKENS = [165105, 147669, 95722, 153201, 34066, 31311, 193569, 41817, 169626, 169626, 169626, 169626]


def flip_byte(path: Path, offset: int) -> None:
    with open(path, "r+b") as f:
        f.seek(offset)
        b = f.read(1)
        f.seek(offset)
        f.write(bytes([b[0] ^ 0x10]))


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class QualitySchemeTests(unittest.TestCase):
    """--scheme quality on the tiny checkpoint: the .yskn file read back bit for bit, the manifest, the refusals, damage."""

    @classmethod
    def setUpClass(cls):
        from yasakani import ink2_yskn
        cls.Y = ink2_yskn
        cls.root = tiny_quality_folder()
        cls.meta = json.loads((cls.root / "yasakani.json").read_text(encoding="utf-8"))
        cls.path = cls.root / cls.meta["weights"]["file"]

    def test_read_back_bit_exact(self):
        """Every tensor: copied ones equal to the published bytes, quantised ones equal to the quantiser on the published values,
        the MXFP8 experts re-packed expert by expert; the decode gives the values of the formula."""
        import inkling2_quant as NQ
        st = conv.SafeTensors(Path(SOURCE) / "model.safetensors")
        counts = dict(copied=0, quantised=0, experts_copied=0, experts_repacked=0)
        with self.Y.YsknFile(self.path) as yf:
            self.assertEqual(yf.check(), [])
            for name, e in yf.tensors.items():
                enc = e["encoding"]
                if e["role"] == "experts":
                    if name + "_weight_amax" in st.entries:
                        for suf, part in conv.EXPERT_PARTS.items():
                            self.assertTrue(np.array_equal(np.asarray(yf.part(name, part)).ravel(),
                                                           np.asarray(st.view(name + suf)).ravel()), name)
                        counts["experts_copied"] += 1
                    else:
                        w, sc = st.view(name + "_weight"), st.view(name + "_weight_scale")
                        for j in range(e["shape"][0]):
                            q = NQ.nvfp4_quant(self.Y.decode_mxfp8(np.asarray(w[j]), np.asarray(sc[j])), conv.NVFP4_SEARCH, 1)
                            self.assertTrue(np.array_equal(q["weight"], yf.part(name, "weight")[j]), (name, j))
                            self.assertTrue(np.array_equal(q["scale"], yf.part(name, "scale")[j]), (name, j))
                            self.assertEqual(q["amax"][0], yf.part(name, "amax")[j])
                        counts["experts_repacked"] += 1
                elif enc in ("nvfp4", "mxfp8"):
                    x = st.f32(name)[:e["shape"][0]]
                    q = NQ.nvfp4_quant(x, conv.NVFP4_SEARCH) if enc == "nvfp4" else NQ.mxfp8_quant(x, conv.MXFP8_SEARCH)
                    for k, v in q.items():
                        self.assertTrue(np.array_equal(v.ravel(), np.asarray(yf.part(name, k)).ravel()), (name, k))
                    counts["quantised"] += 1
                else:
                    self.assertTrue(np.array_equal(np.asarray(yf.part(name, "data")).ravel(), np.asarray(st.view(name)).ravel()), name)
                    counts["copied"] += 1
            self.assertEqual(counts["experts_copied"] + counts["experts_repacked"], 2 * 18)
            self.assertEqual(counts["experts_repacked"], 8)                     # layers 14 to 17 of the tiny model are MXFP8
            self.assertGreater(counts["quantised"], 100)
            self.assertFalse(any(n.startswith("model.visual.") for n in yf.tensors))
            self.assertEqual(yf.entry("model.llm.unembed.weight")["shape"], [200058, 256])
            # the decode follows the formula of docs/FORMAT.md, section 10.3: element (3, 10) is in the low nibble of byte (3, 5)
            pr = yf.parts("model.llm.layers.0.attn.wq_du.weight")
            code = int(pr["weight"][3, 5]) & 0xF
            factor = np.float32(pr["amax"][0]) / np.float32(pr["denominator"][0])
            want = np.float32(self.Y.E2M1[code] * self.Y.E4M3[pr["scale"][3, 0]]) * factor
            self.assertEqual(yf.decode("model.llm.layers.0.attn.wq_du.weight")[3, 10], want)
            for p in (p for e in yf.tensors.values() for p in e["parts"]):
                self.assertEqual(p["offset"] % 4096, 0)

    def test_manifest_notice_and_verify(self):
        w = self.meta["weights"]
        self.assertEqual((w["format"], w["version"], w["scheme"], w["file"]), ("yskn", 2, "quality", self.path.name))
        self.assertEqual(w["size"], self.path.stat().st_size)
        self.assertEqual(w["sha256"], self.meta["sha256"][self.path.name])
        self.assertEqual(self.meta["model_id"], "tiny-inkling2-mini-quality")
        self.assertNotIn("model.safetensors", self.meta["sha256"])
        notice = (self.root / "NOTICE").read_text(encoding="utf-8")
        for text in ("re-quantised by this tool", "https://thinkingmachines.ai/model-acceptable-use-policy/", "Apache License 2.0",
                     "hf-internal-testing/tiny-inkling2-mini", "revision"):
            self.assertIn(text, notice)
        pack = ink.Ink2Pack.load(self.root)
        self.assertEqual(pack.weights_format, "yskn")
        self.assertTrue(all(pack.verify().values()))
        self.assertEqual(pack.check_tensors(), [])
        sc = json.loads((self.root / "selfcheck.json").read_text(encoding="utf-8"))
        self.assertEqual((sc["architecture"], sc["cases"]), ("inkling2", []))      # no reference run in this test

    def test_old_readers_refuse(self):
        """The reader of version 1 stops at the version field; the version 2 reader refuses a version 1 file; the package refuses
        a weights entry of another version."""
        sys.path.insert(0, str(HERE.parent.parent / "tools"))
        import yskn_decode
        with self.assertRaisesRegex(yskn_decode.FormatError, "unsupported version 2"):
            yskn_decode.Container(str(self.path))
        v1 = HERE.parent.parent / "tools" / "testdata" / "tiny.yskn"
        if v1.is_file():
            with self.assertRaisesRegex(self.Y.FormatError, "version 1"):
                self.Y.YsknFile(v1)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "f"
            shutil.copytree(self.root, root, ignore=shutil.ignore_patterns("*.yskn"))
            (root / self.path.name).write_bytes(b"YSKN")
            meta = dict(self.meta)
            meta["weights"] = dict(self.meta["weights"], version=3)
            (root / "yasakani.json").write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "weights format 'yskn' is not supported"):
                ink.Ink2Pack.load(root)

    def test_damage_is_found(self):
        """A flipped byte in a tensor: verify (file hash and tensor CRC) names it; a damaged header is refused."""
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "c"
            shutil.copytree(self.root, root)
            path = root / self.path.name
            with self.Y.YsknFile(path) as yf:
                p = yf.part_entry("model.llm.unembed.weight", "weight")
                off = p["offset"] + p["nbytes"] // 2
            flip_byte(path, off)
            pack = ink.Ink2Pack.load(root)
            res = pack.verify()
            self.assertEqual([n for n, ok in res.items() if not ok], [path.name])
            self.assertEqual(pack.check_tensors(), ["model.llm.unembed.weight"])
            flip_byte(path, off)                                                 # back
            self.assertEqual(pack.check_tensors(), [])
            flip_byte(path, 200)                                                 # in the JSON header
            with self.assertRaisesRegex(self.Y.FormatError, "header is damaged"):
                self.Y.YsknFile(path)


@unittest.skipUnless(HAVE_SOURCE and HAVE_DLL, NEED_SOURCE + " and " + NEED_DLL)
class HostEngineQualityTests(unittest.TestCase):
    """The engine on its host backend loads the .yskn folder (packed experts, decoded dense matrices): greedy tokens equal to those
    of the reference implementation on the same decoded weights; a damaged tensor is refused by the loader."""

    def test_greedy_equals_the_reference(self):
        pack = ink.Ink2Pack.load(tiny_quality_folder())
        fmt, eng, runner = ink.open_runner(pack, 128, dll=DLL, warm=False)
        try:
            ids, _ = fmt.prompt(conv.OFFICIAL_MESSAGES, conv.OFFICIAL_TEMPLATE_KWARGS)
            logits, _ = eng.read_prompt(ids)
            got = []
            for k in range(len(TINY_QUALITY_TOKENS)):
                got.append(int(np.argmax(logits)))
                if k + 1 < len(TINY_QUALITY_TOKENS):
                    logits = eng.step(got[-1])
            self.assertEqual(got, TINY_QUALITY_TOKENS)
            from yasakani import ink2_yskn as Y
            packed = Y.engine_takes_packed_dense(eng.mod.open_dll(str(eng.dll))) and bool(getattr(eng.mod, "TAKES_PACKED_DENSE", False))
            self.assertIn("packed (" if packed else "decoded to bfloat16", eng.weights_note)
        finally:
            eng.close()

    def test_dense_decoded_on_request(self):
        """YASAKANI_INK2_DENSE=bf16: the dense matrices decoded to bfloat16 even for an engine that takes them packed; the same
        greedy tokens."""
        pack = ink.Ink2Pack.load(tiny_quality_folder())
        old = os.environ.get("YASAKANI_INK2_DENSE")
        os.environ["YASAKANI_INK2_DENSE"] = "bf16"
        try:
            fmt, eng, runner = ink.open_runner(pack, 128, dll=DLL, warm=False)
        finally:
            if old is None:
                del os.environ["YASAKANI_INK2_DENSE"]
            else:
                os.environ["YASAKANI_INK2_DENSE"] = old
        try:
            ids, _ = fmt.prompt(conv.OFFICIAL_MESSAGES, conv.OFFICIAL_TEMPLATE_KWARGS)
            logits, _ = eng.read_prompt(ids)
            got = []
            for k in range(len(TINY_QUALITY_TOKENS)):
                got.append(int(np.argmax(logits)))
                if k + 1 < len(TINY_QUALITY_TOKENS):
                    logits = eng.step(got[-1])
            self.assertEqual(got, TINY_QUALITY_TOKENS)
            self.assertIn("decoded to bfloat16", eng.weights_note)
        finally:
            eng.close()

    def test_loader_refuses_a_damaged_tensor(self):
        from yasakani import ink2_yskn as Y
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "c"
            shutil.copytree(tiny_quality_folder(), root)
            pack = ink.Ink2Pack.load(root)
            with Y.YsknFile(pack.weights) as yf:
                p = yf.part_entry("model.llm.layers.3.attn.wo_ud.weight", "scale")
                off = p["offset"] + 7
            flip_byte(pack.weights, off)
            with self.assertRaisesRegex(Exception, r"model\.llm\.layers\.3\.attn\.wo_ud\.weight"):
                ink.open_runner(pack, 64, dll=DLL, warm=False)


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class TemplateAndStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fmt = ink.Ink2Formatter(tiny_folder())

    def test_official_prompt_ids(self):
        ids, in_answer = self.fmt.prompt(conv.OFFICIAL_MESSAGES, conv.OFFICIAL_TEMPLATE_KWARGS)
        self.assertEqual(ids, conv.OFFICIAL_PROMPT_IDS)
        self.assertFalse(in_answer)
        text = self.fmt.render(conv.OFFICIAL_MESSAGES, **conv.OFFICIAL_TEMPLATE_KWARGS)
        self.assertEqual(text, "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
                               "<|message_user|><|content_text|>Explain in one sentence why the sky is usually blue.<|end_message|>")

    def test_effort_and_thinking_off(self):
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
        default = self.fmt.render(msgs)
        self.assertTrue(default.startswith("<|message_system|><|content_text|>Be brief.<|end_message|>"
                                           "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"))
        self.assertIn("Thinking effort level: 0.2<", self.fmt.render(msgs, reasoning_effort="low"))
        ids, in_answer = self.fmt.prompt(msgs, {"enable_thinking": False})
        self.assertTrue(in_answer)
        self.assertEqual(ids[-2:], [self.fmt.ids["<|message_model|>"], self.fmt.ids["<|content_text|>"]])
        self.assertIn("Thinking effort level: 0<", self.fmt.decode(ids))
        from jinja2.exceptions import TemplateError
        with self.assertRaises(TemplateError):
            self.fmt.prompt(msgs, {"reasoning_effort": "huge"})

    def test_earlier_turn_with_reasoning(self):
        msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A", "reasoning_content": "R"},
                {"role": "user", "content": "Q2"}]
        self.assertIn("<|message_model|><|content_thinking|>R<|end_message|><|message_model|><|content_text|>A<|end_message|>"
                      "<|content_model_end_sampling|><|message_user|>", self.fmt.render(msgs))

    def _split(self, ids, in_answer=False):
        s = ink.OutputStream(self.fmt, in_answer)
        parts = {"reasoning": [], "content": []}
        pieces = 0
        for t in ids:
            for kind, text in s.feed(t):
                parts[kind].append(text)
                pieces += 1
        for kind, text in s.flush():
            parts[kind].append(text)
        return "".join(parts["reasoning"]), "".join(parts["content"]), s, pieces

    def test_thinking_answer_split(self):
        think = "The user asks 2+2. Need: 4 — 日本語も 😀."
        answer = "The answer is 4. 答えは４です 👍🏽"
        ids = self.fmt.encode("<|message_model|><|content_thinking|>" + think + "<|end_message|>"
                              "<|message_model|><|content_text|>" + answer + "<|end_message|><|content_model_end_sampling|>")
        r, c, s, pieces = self._split(ids)
        self.assertEqual((r, c), (think, answer))
        n_think = len(self.fmt.encode("<|message_model|><|content_thinking|>" + think + "<|end_message|><|message_model|>"))
        self.assertEqual(s.reasoning_count, n_think)
        self.assertNotIn("�", r + c)
        self.assertLessEqual(pieces, len(ids))

    def test_answer_only_and_name_header(self):
        ids = self.fmt.encode("Sure, here it is.<|end_message|><|content_model_end_sampling|>")
        r, c, s, _ = self._split(ids, in_answer=True)
        self.assertEqual((r, c, s.reasoning_count), ("", "Sure, here it is.", 0))
        ids = self.fmt.encode("<|message_model|>helper<|content_text|>x<|end_message|><|message_model|><|content_text|>y<|end_message|>")
        r, c, s, _ = self._split(ids)
        self.assertEqual((r, c), ("", "xy"))           # the name after <|message_model|> is not shown; blocks are joined

    def test_split_characters_are_held_back(self):
        text = "𠮷野家 👨‍👩‍👧‍👦 🇯🇵"
        ids = self.fmt.encode("<|message_model|><|content_text|>" + text)
        s = ink.OutputStream(self.fmt)
        out = []
        for t in ids:
            for kind, piece in s.feed(t):
                self.assertNotIn("�", piece)
                out.append(piece)
        out += [p for _, p in s.flush()]
        self.assertEqual("".join(out), text)


class SamplerTests(unittest.TestCase):
    def test_greedy_and_seeded(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=5000).astype(np.float32)
        self.assertEqual(ink.Sampler(0.0)(logits), int(np.argmax(logits)))
        self.assertEqual(ink.Sampler(1.0, top_k=1, seed=4)(logits), int(np.argmax(logits)))
        self.assertEqual(ink.Sampler(1.0, top_p=1e-6, seed=4)(logits), int(np.argmax(logits)))
        a = [ink.Sampler(1.0, 0.9, 0, 0.0, seed=7)(logits) for _ in range(3)]
        s = ink.Sampler(1.0, 0.9, 0, 0.0, seed=7)
        b = [s(logits), s(logits), s(logits)]
        self.assertEqual(a[0], b[0])
        self.assertEqual(len(set(ink.Sampler(1.0, seed=i)(logits) for i in range(20))) > 1, True)

    def test_top_p_keeps_the_nucleus(self):
        logits = np.full(10000, -50.0, np.float32)
        logits[[3, 9, 77]] = [5.0, 4.9, 4.8]
        picks = {ink.Sampler(1.0, top_p=0.9, seed=i)(logits) for i in range(200)}
        self.assertTrue(picks <= {3, 9, 77} and len(picks) >= 2)
        picks = {ink.Sampler(1.0, top_k=2, seed=i)(logits) for i in range(200)}
        self.assertEqual(picks, {3, 9})


@unittest.skipUnless(HAVE_SOURCE and HAVE_DLL, NEED_SOURCE + " and " + NEED_DLL)
class HostEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = ink.Ink2Pack.load(tiny_folder())
        cls.fmt, cls.eng, cls.runner = ink.open_runner(cls.pack, 256, dll=DLL, warm=False)

    @classmethod
    def tearDownClass(cls):
        cls.eng.close()

    def test_selfcheck_reference_tokens(self):
        """Greedy on the official prompt: the 20 tokens the official test expects."""
        cases = ink.selfcheck_cases(self.pack)
        self.assertEqual(cases[0]["tokens"], [34066, 91318, 123193] + [169626] * 17)
        out = []
        self.assertEqual(ink.run_selfcheck(self.eng, self.fmt, cases, say=out.append), 0, out)

    def test_runner_usage_and_prefix_reuse(self):
        from yasakani.chat import ChatParams
        p = ChatParams(messages=[{"role": "user", "content": "hi"}], max_tokens=3, temperature=0.0, ignore_eos=True,
                       template_kwargs={"reasoning_effort": "none"})
        self.eng.reset()
        u = self.runner.run(p, lambda kind, text: None)
        self.assertEqual((u.completion_tokens, u.finish, u.cached_tokens), (3, "length", 0))
        first = list(self.eng.fed)
        self.assertEqual(len(first), u.prompt_tokens + 2)          # the last sampled token is not read
        # the same conversation one turn on: only the new ids are read
        ids, _ = self.fmt.prompt(p.messages, p.template_kwargs)
        longer = first + self.fmt.encode("<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>more<|end_message|>")
        _, cached = self.eng.read_prompt(longer)
        self.assertEqual(cached, len(first))
        logits, cached = self.eng.read_prompt(ids)                 # shorter: the positions after its first part are dropped
        self.assertEqual(cached, len(ids) - 1)
        self.assertEqual(self.eng.fed, ids)
        self.assertEqual(self.eng.native.position(), len(ids))
        self.eng.reset()
        fresh, cached = self.eng.read_prompt(ids)                  # the same logits as reading it from the start
        self.assertEqual(cached, 0)
        np.testing.assert_allclose(logits, fresh, rtol=0, atol=1e-3)
        other = self.fmt.encode("<|message_user|><|content_text|>something else<|end_message|>")
        _, cached = self.eng.read_prompt(other)                    # nothing in common: read from the start
        self.assertEqual((cached, self.eng.fed), (0, other))

    def test_prompt_after_an_engine_error_starts_from_a_reset(self):
        ids, _ = self.fmt.prompt([{"role": "user", "content": "hi"}], {"reasoning_effort": "none"})
        self.eng.read_prompt(ids)
        with mock.patch.object(self.eng.native, "feed_many", side_effect=RuntimeError("feed_many: test")):
            with self.assertRaisesRegex(YasakaniError, "engine step failed"):
                self.eng.read_prompt(ids + self.fmt.encode("more words"))
        self.assertIsNone(self.eng.fed)
        _, cached = self.eng.read_prompt(ids)
        self.assertEqual((cached, self.eng.fed), (0, ids))
        with self.assertRaisesRegex(YasakaniError, "context of 256 positions is full"):
            self.eng.read_prompt(list(ids) * (256 // len(ids) + 1))
        self.assertEqual(self.eng.fed, ids)

    def test_requests_that_are_refused(self):
        from yasakani.chat import ChatParams, ContextLengthError, RequestError
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        prepared = self.runner.prepare_checked(ChatParams(messages=[{"role": "user", "content": "x"}], tools=tools))
        self.assertEqual(len(prepared), 3)                     # tools are taken now (tests/test_inkling2_tools.py)
        with self.assertRaises(RequestError):
            self.runner.prepare_checked(ChatParams(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]))
        with self.assertRaises(ContextLengthError):
            self.runner.prepare_checked(ChatParams(messages=[{"role": "user", "content": "word " * 400}]))


@unittest.skipUnless(HAVE_SOURCE and HAVE_DLL, NEED_SOURCE + " and " + NEED_DLL)
class HostServerTests(unittest.TestCase):
    """The OpenAI-compatible server on the tiny model (host backend): models, health, a streamed request with usage, a refused one."""

    @classmethod
    def setUpClass(cls):
        from yasakani.server import ApiState, ServerConfig, bind
        cls.pack = ink.Ink2Pack.load(tiny_folder())
        cls.fmt, cls.eng, runner = ink.open_runner(cls.pack, 256, dll=DLL, warm=False)
        cfg = ServerConfig(port=0, model_id=cls.pack.model_id)
        cfg.sampling_defaults = dict(cls.pack.sampling)
        cfg.effort_aliases = dict(ink.EFFORT_ALIASES)
        cls.state = ApiState(cfg)
        cls.httpd = bind(cls.state)
        cfg.port = cls.httpd.server_address[1]
        cls.state.runner = runner
        threading.Thread(target=cls.httpd.serve_forever, kwargs=dict(poll_interval=0.1), daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.eng.close()

    def _req(self, method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.state.cfg.port, timeout=120)
        c.request(method, path, body=json.dumps(body) if body is not None else None,
                  headers={"Content-Type": "application/json", "Host": "127.0.0.1:%d" % self.state.cfg.port})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def test_models_health_stream(self):
        st, data = self._req("GET", "/v1/models")
        self.assertEqual((st, json.loads(data)["data"][0]["id"]), (200, "tiny-inkling2-mini"))
        st, data = self._req("GET", "/health")
        self.assertEqual((st, json.loads(data)["max_context_tokens"]), (200, 256))
        st, data = self._req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2,
                                                              "stream": True, "stream_options": {"include_usage": True},
                                                              "reasoning_effort": "none", "temperature": 0})
        self.assertEqual(st, 200)
        events = [json.loads(ln[6:]) for ln in data.decode().splitlines() if ln.startswith("data: {")]
        self.assertTrue(data.decode().rstrip().endswith("data: [DONE]"))
        usage = [e["usage"] for e in events if e.get("usage")][0]
        self.assertEqual(usage["completion_tokens"], 2)
        self.assertEqual(usage["prompt_tokens"], len(self.fmt.prompt([{"role": "user", "content": "hi"}], {"reasoning_effort": "none"})[0]))
        self.assertIn("length", [c.get("finish_reason") for e in events for c in e.get("choices", [])])

    def test_refused_requests(self):
        st, data = self._req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 2,
                                                              "tools": [{"type": "function", "function": {"name": "f"}}]})
        self.assertEqual(st, 200)                              # tools are taken now (tests/test_inkling2_tools.py)
        st, data = self._req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "huge"})
        self.assertEqual(st, 400)
        self.assertIn("none, minimal, low, medium, high, xhigh, max", json.loads(data)["error"]["message"])
        st, data = self._req("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]})
        self.assertEqual(st, 400)


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class DrafterFolderTests(unittest.TestCase):
    """The converter places the drafter for both schemes and names it in the manifest; verify checks it; a folder without it,
    or with its file missing, still loads (generation without drafting)."""

    def test_both_schemes_list_the_drafter(self):
        want = (tiny_source() / conv.DRAFTER_FILE).stat().st_size
        for root in (tiny_folder(), tiny_quality_folder()):
            meta = json.loads((root / "yasakani.json").read_text(encoding="utf-8"))
            d = meta["drafter"]
            self.assertEqual((d["file"], d["format"], d["size"]), ("mtp.safetensors", "hf-safetensors", want), root.name)
            self.assertEqual(d["sha256"], conv.sha256_of(root / "mtp.safetensors"))
            self.assertEqual(meta["sha256"]["mtp.safetensors"], d["sha256"])
            self.assertTrue(os.path.samefile(root / "mtp.safetensors", tiny_source() / "mtp.safetensors"))   # hard link
            pack = ink.Ink2Pack.load(root)
            self.assertEqual((pack.drafter, pack.drafter_problem), (root / "mtp.safetensors", ""))
            self.assertIn("mtp.safetensors", pack.verify())
            self.assertIn("drafter (mtp.safetensors)", (root / "NOTICE").read_text(encoding="utf-8"))

    def test_verify_finds_a_changed_drafter_and_folders_without_it_load(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "c"
            shutil.copytree(tiny_folder(), root)
            p = root / "mtp.safetensors"
            b = bytearray(p.read_bytes())
            b[-1] ^= 1
            p.unlink()                                          # the copy may be a hard link of the source: write a new file
            p.write_bytes(bytes(b))
            res = ink.Ink2Pack.load(root).verify()
            self.assertEqual([n for n, ok in res.items() if not ok], ["mtp.safetensors"])
            p.unlink()
            pack = ink.Ink2Pack.load(root)                      # listed, file missing: loads, no drafter, verify names it
            self.assertIsNone(pack.drafter)
            self.assertIn("missing", pack.drafter_problem)
            self.assertFalse(pack.verify()["mtp.safetensors"])
            meta = json.loads((root / "yasakani.json").read_text(encoding="utf-8"))
            del meta["drafter"]
            del meta["sha256"]["mtp.safetensors"], meta["sizes"]["mtp.safetensors"]
            (root / "yasakani.json").write_text(json.dumps(meta), encoding="utf-8")
            pack = ink.Ink2Pack.load(root)                      # a folder of the earlier converter: no drafter entry
            self.assertEqual((pack.drafter, pack.drafter_entry, pack.drafter_problem), (None, {}, ""))
            self.assertTrue(all(pack.verify().values()))


class DraftOptionTests(unittest.TestCase):
    def test_modes_and_depth(self):
        from yasakani._native import YasakaniError
        self.assertEqual(ink.draft_mode_of(None, {}), "auto")
        self.assertEqual(ink.draft_mode_of(None, {"YASAKANI_DRAFT": " OFF "}), "off")
        self.assertEqual(ink.draft_mode_of("on", {"YASAKANI_DRAFT": "off"}), "on")
        with self.assertRaisesRegex(YasakaniError, "YASAKANI_DRAFT / --draft must be one of auto, on, off"):
            ink.draft_mode_of(None, {"YASAKANI_DRAFT": "yes"})
        self.assertIsNone(ink.draft_depth_of(None, {}))
        self.assertEqual(ink.draft_depth_of(None, {"YASAKANI_DRAFT_DEPTH": "5"}), 5)
        self.assertEqual(ink.draft_depth_of(2, {"YASAKANI_DRAFT_DEPTH": "5"}), 2)
        for bad in ("0", "8", "x"):
            with self.assertRaisesRegex(YasakaniError, "from 1 to 7"):
                ink.draft_depth_of(None, {"YASAKANI_DRAFT_DEPTH": bad})

    def test_command_line(self):
        from yasakani import cli
        for cmd in ("chat", "serve", "bench", "selfcheck"):
            args = cli.build_parser().parse_args([cmd])
            self.assertEqual((args.draft, args.draft_depth), (None, None), cmd)
            args = cli.build_parser().parse_args([cmd, "--draft", "on", "--draft-depth", "4"])
            self.assertEqual((args.draft, args.draft_depth), ("on", 4), cmd)

    def test_sampler_seek_gives_the_same_draws(self):
        """Draw k of a seed is the same whether draws for rows that were not kept came in between or not."""
        rng = np.random.default_rng(1)
        rows = [rng.normal(size=3000).astype(np.float32) * 3 for _ in range(6)]
        a = ink.Sampler(1.0, 0.95, seed=11)
        plain = [a(x) for x in rows]
        b = ink.Sampler(1.0, 0.95, seed=11)
        got = [b(rows[0]), b(rows[1])]
        b(rows[4])                                             # a draw for a drafted row that is thrown away
        b(rows[5])
        b.seek(2)
        got += [b(x) for x in rows[2:]]
        self.assertEqual(got, plain)


def standin_class(base):
    """The drafter calls of the engine wrapper (draft_load, draft_mode, spec_step, spec_stats), implemented in Python on
    top of the engine's own step_many and truncate, for tests without the real drafter.  The proposals come from a script:
    ``ref`` (the ids of a plain run: prompt and generated tokens) gives the true continuation while the context equals its start;
    the positions in ``wrong`` (and every position with ``bad``) get a token that is surely not the model's choice."""

    class DraftStandIn(base):
        def draft_load(self, path):
            if not os.path.isfile(path):
                raise RuntimeError("drafter file not found: %s" % path)
            self.sd = dict(mode="auto", depth=3, ref=[], wrong=set(), bad=False)
            self.sd_stats = dict.fromkeys(("passes", "proposed", "accepted", "tokens", "draft_ms", "verify_ms", "active_passes"), 0)
            self.pick_calls = []

        def draft_loaded(self):
            return hasattr(self, "sd")

        def draft_mode(self, mode=None, depth=None):
            if mode is not None:
                assert mode in ("off", "on", "auto"), mode
                self.sd["mode"] = mode
            if depth is not None:
                assert 1 <= depth <= 7, depth
                self.sd["depth"] = int(depth)
            return dict(mode=self.sd["mode"], depth=self.sd["depth"], active=self.sd["mode"] != "off")

        def spec_stats(self, reset=False):
            out = dict(self.sd_stats)
            if reset:
                self.sd_stats = dict.fromkeys(self.sd_stats, 0)
            return out

        def _propose(self, p0, anchor):
            sd, ref = self.sd, self.sd["ref"]
            ctx = list(self.seen_ids) + [int(anchor)]
            known = not sd["bad"] and ref[:len(ctx)] == ctx
            out = []
            for pos in range(p0 + 1, p0 + 1 + sd["depth"]):
                true = ref[pos] if pos < len(ref) else 0
                out.append(int(true) if known and pos < len(ref) and pos not in sd["wrong"] else (int(true) + 1) % self.vocab)
            return out

        def spec_step(self, anchor, pick=None, want_logits=False):
            st = self.sd_stats
            st["passes"] += 1
            if self.sd["mode"] == "off":
                logits = self.step(anchor)
                t = int(np.argmax(logits)) if pick is None else int(pick(logits))
                self.pick_calls.append(1)
                st["tokens"] += 1
                return [], t, (logits if want_logits else None)
            p0 = self.num_seen
            drafts = self._propose(p0, anchor)
            logits, top, _ = self.step_many([int(anchor)] + drafts, logits=pick is not None or want_logits)
            accepted, calls = [], 0
            for r in range(len(drafts) + 1):
                t = int(top[r]) if pick is None else int(pick(logits[r]))
                calls += 1
                if r < len(drafts) and t == drafts[r]:
                    accepted.append(t)
                    continue
                break
            self.truncate(p0 + 1 + len(accepted))
            self.pick_calls.append(calls)
            st["active_passes"] += 1
            st["proposed"] += len(drafts)
            st["accepted"] += len(accepted)
            st["tokens"] += 1 + len(accepted)
            return accepted, t, (logits[len(accepted)].copy() if want_logits else None)

    return DraftStandIn


@unittest.skipUnless(HAVE_SOURCE and HAVE_DLL, NEED_SOURCE + " and " + NEED_DLL)
class HostDraftTests(unittest.TestCase):
    """The generation loop with bursts (a stand-in of the drafter calls on the host engine) against plain stepping: the same tokens,
    text, finish reason and engine state when the reply ends in the middle of a burst; prompt reuse afterwards; sampling with a seed;
    a drafter file the engine cannot load; selfcheck and the server."""

    @classmethod
    def setUpClass(cls):
        cls.pack = ink.Ink2Pack.load(tiny_folder())
        cls.fmt, cls.eng, cls.runner = ink.open_runner(cls.pack, 256, dll=DLL, warm=False)
        cls.fallback_note = cls.eng.draft_note                  # the stand-in drafter file does not fit the tiny model
        cls.eng.native.__class__ = standin_class(type(cls.eng.native))
        cls.note = cls.eng.set_drafter(cls.pack, "on", 3)
        cls.prompt = cls.fmt.encode("hi") + cls.fmt.answer_prefix          # short: the host engine reads one id per step
        cls.ref = cls._plain(dict(max_tokens=14))[3]                        # prompt + 13 greedy tokens
        cls.ref_s = cls._plain(dict(max_tokens=12, temperature=1.0, seed=3))[3]

    @classmethod
    def tearDownClass(cls):
        cls.eng.close()

    @classmethod
    def _run(cls, kw, stops=None):
        from yasakani.chat import ChatParams
        p = ChatParams(messages=[{"role": "user", "content": "hi"}], ignore_eos=stops is None,
                       **dict(dict(temperature=0.0), **kw))
        pieces = []
        saved = cls.runner.stop_ids
        if stops is not None:
            cls.runner.stop_ids = frozenset(stops)
        try:
            u = cls.runner.run(p, lambda kind, text: pieces.append((kind, text)), prepared=(cls.prompt, True))
        finally:
            cls.runner.stop_ids = saved
        assert cls.eng.native.position() == len(cls.eng.fed)
        return u, "".join(t for _, t in pieces), u.finish, list(cls.eng.fed)

    @classmethod
    def _plain(cls, kw, stops=None):
        cls.eng.drafting = False
        try:
            return cls._run(kw, stops)
        finally:
            cls.eng.drafting = True

    def _oracle(self, ref, wrong=(), depth=3, bad=False):
        sd = self.eng.native.sd
        sd.update(ref=list(ref), wrong=set(wrong), depth=depth, bad=bad)
        self.eng.draft_depth = depth
        self.eng.native.pick_calls = []

    def _same(self, kw, stops=None, **oracle):
        want = self._plain(kw, stops)
        self._oracle(oracle.pop("ref", self.ref), **oracle)
        got = self._run(kw, stops)
        self.assertEqual(got[1:], want[1:])
        self.assertEqual((got[0].completion_tokens, got[0].prompt_tokens), (want[0].completion_tokens, want[0].prompt_tokens))
        return got[0], want

    def test_notes(self):
        self.assertIn("mtp.safetensors could not be loaded", self.fallback_note)
        self.assertIn("generation without drafting", self.fallback_note)
        self.assertEqual(self.note, "drafter: mtp.safetensors loaded, mode on, depth 3")

    def test_greedy_bursts_equal_plain_greedy(self):
        p = len(self.prompt)
        u, _ = self._same(dict(max_tokens=13), wrong={p + 4, p + 9})
        self.assertGreater(u.spec["tokens_per_pass"], 1.5)
        self.assertEqual(u.passes, u.spec["draft_passes"])
        self.assertLess(u.passes, u.completion_tokens)
        self.assertGreater(u.draft_n_accepted, 0)
        u, _ = self._same(dict(max_tokens=8), bad=True)            # every proposal wrong: one token per pass, same output
        self.assertEqual((u.spec["tokens_per_pass"], u.draft_n_accepted), (1.0, 0))

    def test_limit_stop_token_and_stop_string_in_a_burst(self):
        p = len(self.prompt)
        gen = self.ref[p:]
        # depth 4 with no wrong proposals: token 0 comes from the prompt, tokens 1..4 are accepted in the first pass, so the
        # third token (index 2) and the stop token at index 3 end the reply in the middle of a burst
        u, (_, _, finish, fed) = self._same(dict(max_tokens=3), depth=4)
        self.assertEqual((finish, fed), ("length", self.ref[:p + 2]))
        j = next(i for i in range(2, len(gen)) if gen[i] not in gen[:i])
        u, (_, _, finish, fed) = self._same(dict(max_tokens=12), stops=[gen[j]], depth=4)
        self.assertEqual((finish, u.completion_tokens, fed), ("stop", j + 1, self.ref[:p + j]))
        text = self.fmt.decode(gen[2:3])
        self.assertTrue(text.strip())
        u, (_, content, finish, fed) = self._same(dict(max_tokens=12, stop=[text]), depth=4)
        self.assertEqual((finish, u.completion_tokens), ("stop", 3))
        self.assertNotIn(text, content)
        # the next prompt goes on from what was emitted: only its new ids are read, and the state equals a fresh read
        longer = fed + gen[len(fed) - p:len(fed) - p + 1] + self.fmt.encode(" more")
        logits, cached = self.eng.read_prompt(longer)
        self.assertEqual(cached, len(fed))
        self.eng.reset()
        fresh, cached = self.eng.read_prompt(longer)
        self.assertEqual(cached, 0)
        np.testing.assert_allclose(logits, fresh, rtol=0, atol=1e-4)
        self.assertEqual(int(np.argmax(logits)), int(np.argmax(fresh)))

    def test_sampled_with_a_seed_equals_plain(self):
        p = len(self.prompt)
        kw = dict(max_tokens=11, temperature=1.0, seed=3)
        u, _ = self._same(kw, ref=self.ref_s, wrong={p + 5})
        self.assertGreater(u.draft_n_accepted, 0)
        calls = self.eng.native.pick_calls
        self.assertEqual(sum(calls), u.draft_n_accepted + len(calls))     # one pick per accepted token and one per pass
        self._same(dict(kw, max_tokens=4), ref=self.ref_s, depth=5)       # the limit inside the first burst

    def test_forced_tokens_in_a_burst(self):
        """generate(): tokens forced after a token in the middle of a burst (as the thinking budget does) give the same
        sequence and state as plain stepping."""
        forced = self.fmt.encode("<|end_message|><|message_model|><|content_text|>")

        def once(draft):
            self.eng.drafting = draft
            self._oracle(self.ref, depth=4)
            try:
                logits, _ = self.eng.read_prompt(self.prompt)
                out = []

                def on_token(t):
                    out.append(t)
                    if len(out) == 3:
                        return forced
                    return ink.END if len(out) >= 9 else None
                passes = ink.generate(self.eng, int(np.argmax(logits)), on_token)
                return out, list(self.eng.fed), passes
            finally:
                self.eng.drafting = True
                self.eng.reset()
        plain, burst = once(False), once(True)
        self.assertEqual(burst[:2], plain[:2])
        self.assertEqual(plain[0][3:6], forced)
        self.assertLess(burst[2], plain[2])

    def test_selfcheck_with_bursts(self):
        cases = ink.selfcheck_cases(self.pack)
        ids = self.fmt.encode(self.fmt.render(cases[0]["messages"], **cases[0]["template_kwargs"]))
        self._oracle(ids + cases[0]["tokens"], depth=4)
        out = []
        self.assertEqual(ink.run_selfcheck(self.eng, self.fmt, cases, say=out.append), 0, out)
        self.assertIn("tokens per pass", out[0])

    def test_serve_streams_bursts(self):
        from yasakani.chat import ChatParams
        from yasakani.server import ApiState, ServerConfig, bind
        cfg = ServerConfig(port=0, model_id=self.pack.model_id)
        cfg.sampling_defaults = dict(self.pack.sampling)
        cfg.effort_aliases = dict(ink.EFFORT_ALIASES)
        state = ApiState(cfg)
        httpd = bind(state)
        cfg.port = httpd.server_address[1]
        state.runner = self.runner
        threading.Thread(target=httpd.serve_forever, kwargs=dict(poll_interval=0.1), daemon=True).start()
        msgs = [{"role": "user", "content": "hi"}]
        body = {"messages": msgs, "max_tokens": 6, "stream": True, "stream_options": {"include_usage": True},
                "reasoning_effort": "none", "temperature": 0}
        try:
            prompt, _ = self.fmt.prompt(msgs, {"reasoning_effort": "none"})
            self.eng.drafting = False
            try:
                plain = []
                self.runner.run(ChatParams(messages=msgs, max_tokens=7, temperature=0.0, template_kwargs={"reasoning_effort": "none"}),
                                lambda kind, text: plain.append(text))
                ref = list(self.eng.fed)
            finally:
                self.eng.drafting = True
            self.eng.reset()
            self._oracle(ref, depth=3)
            c = http.client.HTTPConnection("127.0.0.1", cfg.port, timeout=300)
            c.request("POST", "/v1/chat/completions", body=json.dumps(body),
                      headers={"Content-Type": "application/json", "Host": "127.0.0.1:%d" % cfg.port})
            r = c.getresponse()
            data = r.read().decode()
            c.close()
            self.assertEqual(r.status, 200)
            events = [json.loads(ln[6:]) for ln in data.splitlines() if ln.startswith("data: {")]
            usage = [e["usage"] for e in events if e.get("usage")][0]
            timings = [e["timings"] for e in events if e.get("timings")][0]
            self.assertEqual(usage["completion_tokens"], 6)
            self.assertEqual(timings["draft_mode"], "on")
            self.assertGreater(timings["tokens_per_pass"], 1.0)
            self.assertLess(timings["draft_passes"], 6)
            self.assertEqual(self.eng.fed, ref[:len(prompt) + 5])
        finally:
            httpd.shutdown()
            httpd.server_close()


class ServerDefaultsTests(unittest.TestCase):
    def test_model_defaults_and_efforts(self):
        from yasakani.server import ServerConfig, parse_chat_request
        q = ServerConfig()
        p = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "high"}, q)
        self.assertEqual((p.temperature, p.top_p, p.top_k, p.template_kwargs["reasoning_effort"]), (1.0, 0.95, 20, "xhigh"))
        c = ServerConfig(model_id="inkling-2-mini")
        c.sampling_defaults = dict(temperature=1.0, top_p=1.0, top_k=0)
        c.effort_aliases = dict(ink.EFFORT_ALIASES)
        p = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "high"}, c)
        self.assertEqual((p.temperature, p.top_p, p.top_k, p.template_kwargs["reasoning_effort"]), (1.0, 1.0, 0, "high"))
        p = parse_chat_request({"messages": [{"role": "user", "content": "x"}], "top_p": 0.5, "temperature": 0.3, "seed": 9}, c)
        self.assertEqual((p.temperature, p.top_p, p.seed), (0.3, 0.5, 9))


class DispatchTests(unittest.TestCase):
    def test_commands_go_by_architecture(self):
        from yasakani import cli, inkling2_cli
        with tempfile.TemporaryDirectory() as t:
            q = Path(t) / "qwen"
            q.mkdir()
            (q / "yasakani.json").write_text(json.dumps({"model_id": "qwen3.8-27b"}), encoding="utf-8")
            k = Path(t) / "ink"
            k.mkdir()
            (k / "yasakani.json").write_text(json.dumps({"architecture": "inkling2"}), encoding="utf-8")
            for cmd in ("info", "verify", "selfcheck", "chat", "serve", "bench"):
                args = cli.build_parser().parse_args([cmd, "--model-dir", str(q)])
                self.assertIsNone(cli._inkling2_command(args), cmd)
                args = cli.build_parser().parse_args([cmd, "--model-dir", str(k)])
                self.assertIs(cli._inkling2_command(args), inkling2_cli.COMMANDS[cmd], cmd)
            args = cli.build_parser().parse_args(["update", "--check"])
            self.assertIsNone(cli._inkling2_command(args))

    def test_unsupported_options(self):
        from yasakani import inkling2_cli
        from yasakani._native import YasakaniError
        with self.assertRaisesRegex(YasakaniError, "--parallel"):
            inkling2_cli.refuse_unsupported(argparse.Namespace(parallel=2))
        inkling2_cli.refuse_unsupported(argparse.Namespace(parallel=None, edit=False, write=None, mmproj=None))

    def test_chat_defaults_left_to_the_model(self):
        from yasakani import cli
        args = cli.build_parser().parse_args(["chat"])
        self.assertEqual((args.temperature, args.top_p, args.top_k, args.ask), (None, None, None, None))


# ---- the expected self-check tokens shipped for --scheme quality ---------------------------------------------------------------
QUALITY_DIR = os.environ.get("YASAKANI_TEST_INK2_QUALITY_DIR")   # optional: the folder of Inkling-2 Mini converted with --scheme quality


class ShippedSelfcheckTests(unittest.TestCase):
    """data/inkling2_selfcheck.json: one entry for the quality conversion of the published revision, with the three cases."""

    def test_the_data_file(self):
        d = json.loads(conv.KNOWN_SELFCHECK.read_text(encoding="utf-8"))
        entries = conv.known_conversions("thinkingmachines/Inkling-2-Mini", "ed40e7da625f594fc9c4990f5f66be24873c22d7", "quality")
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(d["conversions"], [e])
        self.assertEqual((e["file"], e["size"], e["sha256"]), ("inkling_2_mini_quality.yskn", 15052423172,
                                                               "44ef7ea5eade7a1e6cfb4f7ad3f7568a701ce9c621f7e4d00d38b624f4de138d"))
        self.assertEqual(sorted(e["cases"]), sorted(c["name"] for c in conv.SELFCHECK_CASES))
        self.assertEqual(sorted(e["cases"]), ["code", "english", "japanese"])
        for c in e["cases"].values():
            self.assertEqual(c["prompt_ids"][:2], conv.OFFICIAL_PROMPT_IDS[:2])          # the chat template's first ids
        for c in e["cases"].values():
            self.assertEqual(len(c["tokens"]), conv.SELFCHECK_TOKENS)
        self.assertEqual(conv.known_conversions("thinkingmachines/Inkling-2-Mini", "0" * 40, "quality"), [])
        self.assertEqual(conv.known_conversions("thinkingmachines/Inkling-2-Mini", e["revision"], "published"), [])

    @unittest.skipUnless(QUALITY_DIR, "set YASAKANI_TEST_INK2_QUALITY_DIR to a folder converted with --scheme quality")
    def test_equal_to_the_converted_folder(self):
        root = Path(QUALITY_DIR)
        meta = json.loads((root / "yasakani.json").read_text(encoding="utf-8"))
        sc = json.loads((root / "selfcheck.json").read_text(encoding="utf-8"))
        e = conv.known_conversions(meta["source"]["repository"], meta["source"]["revision"], meta["weights"]["scheme"])[0]
        self.assertEqual((meta["weights"]["file"], meta["weights"]["sha256"], meta["weights"]["size"]),
                         (e["file"], e["sha256"], e["size"]))
        self.assertEqual({c["name"]: (c["prompt_ids"], c["tokens"]) for c in sc["cases"]},
                         {n: (c["prompt_ids"], c["tokens"]) for n, c in e["cases"].items()})


@unittest.skipUnless(HAVE_SOURCE, NEED_SOURCE)
class ShippedSelfcheckConvertTests(unittest.TestCase):
    """The converter writes the shipped tokens only when its .yskn file has the SHA-256 of the entry (tiny checkpoint)."""

    def convert_with(self, sha: str, dest: Path) -> dict:
        src_meta = json.loads((tiny_quality_folder() / "yasakani.json").read_text(encoding="utf-8"))
        entry = dict(repository=src_meta["source"]["repository"], revision=src_meta["source"]["revision"], scheme="quality",
                     file=src_meta["weights"]["file"], sha256=sha, size=src_meta["weights"]["size"], implementation="a test run",
                     cases={"official": {"prompt_ids": conv.OFFICIAL_PROMPT_IDS, "tokens": TINY_QUALITY_TOKENS}})
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = dest.parent / "known.json"
        data.write_text(json.dumps({"format": 1, "conversions": [entry]}), encoding="utf-8")
        official = [dict(name="official", messages=conv.OFFICIAL_MESSAGES, template_kwargs=conv.OFFICIAL_TEMPLATE_KWARGS)]
        with mock.patch.object(conv, "KNOWN_SELFCHECK", data), mock.patch.object(conv, "SELFCHECK_CASES", official):
            conv.convert(tiny_source(), dest, repo=entry["repository"], revision=entry["revision"], scheme="quality", threads=4)
        return json.loads((dest / "selfcheck.json").read_text(encoding="utf-8"))

    def test_known_and_other_sha256(self):
        want = json.loads((tiny_quality_folder() / "yasakani.json").read_text(encoding="utf-8"))["weights"]["sha256"]
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "known" / "f"
            sc = self.convert_with(want, root)                                   # the same conversion again: the same file
            self.assertEqual([(c["name"], c["tokens"], c["prompt_ids"]) for c in sc["cases"]],
                             [("official", TINY_QUALITY_TOKENS, conv.OFFICIAL_PROMPT_IDS)])
            self.assertEqual(sc["cases"][0]["source"], "greedy output of the converted weights: a test run")
            self.assertNotIn("note", sc)
            meta = json.loads((root / "yasakani.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["weights"]["sha256"], want)
            pack = ink.Ink2Pack.load(root)
            self.assertTrue(all(pack.verify().values()))
            self.assertEqual(ink.selfcheck_cases(pack)[0]["tokens"], TINY_QUALITY_TOKENS)
            if HAVE_DLL:                                                         # and the folder passes selfcheck
                fmt, eng, _ = ink.open_runner(pack, 128, dll=DLL, warm=False)
                try:
                    out = []
                    self.assertEqual(ink.run_selfcheck(eng, fmt, ink.selfcheck_cases(pack), say=out.append), 0, out)
                finally:
                    eng.close()
            sc = self.convert_with("0" * 64, Path(t) / "other" / "f")            # another SHA-256: no cases, the usual note
            self.assertEqual(sc["cases"], [])
            self.assertIn("no expected tokens were made", sc["note"])


# ---- malformed model files: clean errors -------------------------------------------------------------------------------------------
def yskn_bytes(header) -> bytes:
    """A YSKN version 2 file with this JSON header (CRC-32 correct) and no data."""
    import zlib
    from yasakani import ink2_yskn as Y
    js = json.dumps(header).encode("ascii")
    data_off = -(-(Y.PREFIX_BYTES + len(js)) // Y.ALIGN) * Y.ALIGN
    head = Y.MAGIC + struct.pack("<IQQQII", Y.VERSION, len(js), data_off, data_off, zlib.crc32(js), 0)
    return (head + js).ljust(data_off, b"\0")


def safetensors_bytes(header, data: bytes = b"", header_len=None) -> bytes:
    js = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(js) if header_len is None else header_len) + js + data


class ManifestPathTests(unittest.TestCase):
    """Every file the manifest names must be inside the model folder."""

    def folder(self, **extra):
        root = Path(tempfile.mkdtemp(prefix="ink2_paths_"))
        self.addCleanup(shutil.rmtree, root, True)
        m = root / "m"
        m.mkdir()
        for name in ("w.safetensors", "config.json", "tokenizer.json", "chat_template.jinja"):
            (m / name).write_bytes(b"{}")
        (root / "outside.safetensors").write_bytes(b"x")
        meta = {"architecture": "inkling2", "weights": {"file": "w.safetensors"}}
        meta.update(extra)
        (m / "yasakani.json").write_text(json.dumps(meta), encoding="utf-8")
        return m

    def test_plain_names_are_taken(self):
        pack = ink.Ink2Pack.load(str(self.folder()))
        self.assertEqual(pack.weights.name, "w.safetensors")

    def test_paths_that_leave_the_folder_are_refused(self):
        outside = "C:/x.safetensors" if os.name == "nt" else "/x.safetensors"
        for bad in ("../outside.safetensors", "sub/../../outside.safetensors", outside):
            for extra in ({"drafter": {"file": bad}}, {"weights": {"file": bad}}, {"sha256": {bad: "00"}}):
                with self.subTest(bad=bad, entry=list(extra)[0]):
                    with self.assertRaisesRegex(YasakaniError, "is not a file inside the model folder"):
                        ink.Ink2Pack.load(str(self.folder(**extra)))


class MalformedFileTests(unittest.TestCase):
    def test_yskn_header_of_wrong_types(self):
        from yasakani import ink2_yskn as Y
        base = {"format": "yskn", "version": 2}
        t16 = {"name": "a", "encoding": "bf16", "shape": [2, 16],
               "parts": [{"part": "data", "dtype": "bf16", "shape": [2, 16], "nbytes": 64, "offset": 4096}]}
        cases = [
            ([], "does not describe a YSKN version 2 file"),
            (dict(base, tensors={"a": 1}), r'"tensors" is not a list'),
            (dict(base, tensors=[1]), "tensor entry 0 of the header has no name"),
            (dict(base, tensors=[dict(t16, shape=["2", 16])]), "not an integer"),
            (dict(base, tensors=[dict(t16, shape=[-2, 16])]), "negative size"),
            (dict(base, tensors=[dict(t16, parts={"data": {}})]), "not a list of named parts"),
            (dict(base, tensors=[dict(t16, parts=[dict(t16["parts"][0], nbytes=64.0)])]), "nbytes is not an integer"),
            (dict(base, tensors=[dict(t16, parts=[dict(t16["parts"][0], offset="4096")])]), "offset is not an integer"),
            (dict(base, tensors=[dict(t16, parts=[dict(t16["parts"][0], offset=-4096)])]), "offset out of range"),
            (dict(base, tensors=[dict(t16, role="experts")]), "experts stored as bf16"),
            (dict(base, alignment=1.5, tensors=[]), "alignment"),
            (dict(base, model=[1], tensors=[]), r'"model" is not an object'),
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            for k, (header, msg) in enumerate(cases):
                path = Path(t) / ("h%d.yskn" % k)
                path.write_bytes(yskn_bytes(header))
                with self.assertRaisesRegex(Y.FormatError, msg):
                    Y.YsknFile(path).close()
            path = Path(t) / "ok.yskn"
            path.write_bytes(yskn_bytes(dict(base, tensors=[])))
            with Y.YsknFile(path) as yf:
                self.assertEqual(yf.tensors, {})

    def test_safetensors_header_and_offsets(self):
        from yasakani import _ink2_ckpt as C
        entry = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
        cases = [
            (safetensors_bytes({"a": entry}, b"\0" * 8, header_len=1 << 40), "larger than the file"),
            (safetensors_bytes([1], b""), "not a JSON object"),
            (b"\x05\0\0\0\0\0\0\0{abc}", "not JSON"),
            (safetensors_bytes({"a": dict(entry, data_offsets=[-8, 0])}, b"\0" * 8), "not inside the file"),
            (safetensors_bytes({"a": dict(entry, data_offsets=[0, 16])}, b"\0" * 8), "not inside the file"),
            (safetensors_bytes({"a": dict(entry, data_offsets=[8, 0])}, b"\0" * 8), "not inside the file"),
            (safetensors_bytes({"a": dict(entry, data_offsets=[0, "8"])}, b"\0" * 8), "not two integers"),
            (safetensors_bytes({"a": dict(entry, shape="2")}, b"\0" * 8), "not a list of sizes"),
            (safetensors_bytes({"a": 3}, b""), "not an object"),
            (b"\0\0\0", "not a safetensors file"),
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
            for k, (raw, msg) in enumerate(cases):
                path = Path(t) / ("c%d.safetensors" % k)
                path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, msg):
                    C.Checkpoint(str(path))
            path = Path(t) / "size.safetensors"
            path.write_bytes(safetensors_bytes({"a": dict(entry, shape=[3])}, b"\0" * 8))
            ck = C.Checkpoint(str(path))
            with self.assertRaisesRegex(ValueError, "8 bytes of data for a F32 tensor of shape"):
                ck.raw("a")
            del ck
            path = Path(t) / "good.safetensors"
            path.write_bytes(safetensors_bytes({"a": entry}, np.array([1, 2], np.float32).tobytes()))
            ck = C.Checkpoint(str(path))
            self.assertEqual(ck.f32("a").tolist(), [1.0, 2.0])
            ck.data._mmap.close()

    def test_config_beyond_the_engine(self):
        import types
        with tempfile.TemporaryDirectory() as t:
            cfg = Path(t) / "config.json"
            for text, msg in ((dict(num_hidden_layers=ink.MAX_LAYERS + 1), "num_hidden_layers is 65"),
                              (dict(num_hidden_layers=0), "num_hidden_layers is 0"),
                              (dict(num_hidden_layers="24"), "num_hidden_layers is '24'"),
                              (dict(num_hidden_layers=6, layer_types=["hybrid"] * 5), "layer_types"),
                              (dict(loop_config=[1]), "loop_config")):
                cfg.write_text(json.dumps({"text_config": text}), encoding="utf-8")
                with self.assertRaisesRegex(YasakaniError, msg):
                    ink.read_config(t)
                with self.assertRaisesRegex(YasakaniError, msg):             # before the library is looked for or called
                    ink.Ink2Engine(types.SimpleNamespace(root=Path(t)), 64, dll=str(Path(t) / "missing.dll"))
            cfg.write_text(json.dumps({"model_type": "inkling2_loop"}), encoding="utf-8")
            with self.assertRaisesRegex(YasakaniError, "no text_config"):
                ink.read_config(t)
            cfg.write_text(json.dumps({"text_config": {"num_hidden_layers": ink.MAX_LAYERS}}), encoding="utf-8")
            self.assertEqual(ink.read_config(t)["text_config"]["num_hidden_layers"], 64)

    def test_binding_template_is_sandboxed(self):
        from jinja2.exceptions import SecurityError
        from yasakani import _ink2_native
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "chat_template.jinja").write_text("{{ messages.append(1) }}", encoding="utf-8")
            with self.assertRaises(SecurityError):
                _ink2_native.render_chat(t, [{"role": "user", "content": "hi"}])
            (Path(t) / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }}{% break %}{% endfor %}",
                                                       encoding="utf-8")
            self.assertEqual(_ink2_native.render_chat(t, [{"role": "user", "content": "hi"}, {"role": "user", "content": "x"}]), "hi")


if __name__ == "__main__":
    unittest.main()
