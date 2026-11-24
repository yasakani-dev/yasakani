# SPDX-License-Identifier: Apache-2.0
"""Wide copy passes per GPU (first in the 0.1.7 test build; on RTX 30 / 40 by default since 0.1.7; yasakani/widecheck.py) and ``yasakani report`` (yasakani/report.py).

The rule, the kept results, the nvidia-smi parsing and the report text are plain Python (run everywhere).  The start-up flow runs on
the stub library built with -DSTUB_WIDE (a C compiler is needed: skipped on Windows), whose copy cycles take the times the tests set.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.support import Stub, make_pack
from yasakani import widecheck as wc
from yasakani._native import YasakaniError
from yasakani.engine import Engine
from yasakani.runtime import RuntimeProfile

UUID = "GPU-12345678-9abc-def0-0fed-cba987654321"     # the stub's (yskn_wide_state words 0x12345678 0x9abcdef0 0x0fedcba9 0x87654321)


def timing(r32=1.39, r48=1.94, r64=2.27, t16=11.0, rounds=3, sampled=None, jitter=0.0):
    t = wc.Timing()
    for k in range(rounds):
        j = 1.0 + (jitter if k % 2 else 0.0)
        t.rounds.append({32: (t16, t16 * r32 * j), 48: (t16, t16 * r48 * j), 64: (t16, t16 * r64 * j)})
    if sampled:
        t.sampled = {w: (t16, t16 * r) for w, r in sampled.items()}
    return t


class FakeRun:
    """A stand-in for subprocess.run that answers nvidia-smi queries by UUID."""

    def __init__(self, util=0, apps="", fields=None, fail_full=False, known=True):
        self.util, self.apps, self.fail_full, self.known = util, apps, fail_full, known
        self.fields = dict({"name": "NVIDIA GeForce RTX 3090", "driver_version": "616.92", "memory.total": "24576",
                            "driver_model.current": "WDDM", "power.limit": "350.00", "clocks.max.sm": "2100",
                            "clocks.max.memory": "9751", "display_attached": "Yes", "utilization.gpu": str(util),
                            "memory.used": "1200", "pstate": "P8", "temperature.gpu": "40", "clocks.sm": "210",
                            "clocks_event_reasons.active": "0x0000000000000001", "display_active": "Enabled"}, **(fields or {}))
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        out, rc = "", 0
        q = [a for a in cmd if a.startswith("--query-gpu=")]
        ids = [a for a in cmd if a.startswith("--id=")]
        if not self.known or (ids and ids[0] != "--id=" + UUID):
            rc = 6
        elif q:
            fs = q[0].split("=", 1)[1].split(",")
            if self.fail_full and len(fs) > 1:
                rc = 2
            elif any(f not in self.fields for f in fs):
                rc = 2
            else:
                out = ", ".join(self.fields[f] for f in fs) + "\n"
        elif any(a.startswith("--query-compute-apps") for a in cmd):
            out = self.apps
        return subprocess.CompletedProcess(cmd, rc, out, "")


class RuleTests(unittest.TestCase):
    def test_limits(self):
        self.assertEqual([round(wc.limit(w), 2) for w in wc.WIDTHS], [1.6, 2.4, 3.2])

    def test_rtx5090_values_keep_every_width(self):
        for native in (True, False):
            mask, ws = wc.decide(timing(), native)
            self.assertEqual(mask, 7, native)
            self.assertTrue(all(w.used and not w.unsure for w in ws))

    def test_native_drops_a_width_only_when_every_round_is_over(self):
        mask, _ = wc.decide(timing(r48=2.8), True)                # RTX 50: limit 0.9 x 48 / 16 = 2.70 (since 0.3.4)
        self.assertEqual(mask, 5)
        t = timing(r48=2.8)
        t.rounds[1][48] = (11.0, 11.0 * 2.3)                     # one round under the limit: kept
        self.assertEqual(wc.decide(t, True)[0], 7)
        self.assertEqual(wc.decide(t, False)[0], 5)               # RTX 30 / 40: every round must be under it

    def test_unsure_rounds(self):
        t = timing(r64=3.0, jitter=0.2)                           # rounds differ by 20 %: unsure
        mask, ws = wc.decide(t, False)
        self.assertEqual(mask & 4, 0)
        self.assertTrue(next(w for w in ws if w.rows == 64).unsure)
        t = timing(r64=3.5, jitter=0.2)                           # over the limit but unsure: RTX 50 keeps it
        self.assertEqual(wc.decide(t, True)[0] & 4, 4)

    def test_sampled_round_counts(self):
        t = timing(sampled={32: 1.7, 48: 1.9, 64: 2.2})           # sampling makes 32 rows too slow
        self.assertEqual(wc.decide(t, False)[0], 6)
        self.assertEqual(wc.decide(t, True)[0], 7)                # greedy rounds are under: RTX 50 keeps it

    def test_recheck_rate(self):
        # M3: a rate over a minimum sample, not a count of back-offs
        self.assertFalse(wc.recheck_due(wc.RECHECK_MIN_PASSES - 1, wc.RECHECK_MIN_PASSES))      # too few wide passes
        self.assertTrue(wc.recheck_due(128, 13))
        self.assertFalse(wc.recheck_due(128, 12))
        self.assertFalse(wc.recheck_due(1000, 99))
        self.assertTrue(wc.recheck_due(1000, 100))
        self.assertFalse(wc.recheck_due(10000, 8))                 # 8 back-offs (the old rule) over a long serve: no new check

    def test_times(self):
        self.assertEqual(wc.times_us(timing()), [11000, 15290, 21340, 24970])
        self.assertEqual(wc.times_us(wc.Timing()), [0, 0, 0, 0])

    def test_robust_ms_ignores_one_stall(self):
        self.assertAlmostEqual(wc.robust_ms([10, 10, 10, 10, 10, 90, 10, 10]), 10.0)
        self.assertAlmostEqual(wc.robust_ms([12.0]), 12.0)

    def test_widths(self):
        self.assertEqual(wc.widths_of(5), [32, 64])
        self.assertEqual(wc.mask_of([48, 64]), 6)
        self.assertEqual(wc.widths_text(0), "none")
        self.assertEqual(wc.widths_text(7), "32, 48, 64 rows")


class FakeGateEngine:
    """Records the prompt reads and the passes of the correctness check; ``wrong`` = rows of the passes that differ (NLL)."""

    def __init__(self, wrong=()):
        self.pos, self.reads, self.passes, self.wrong = 0, [], [], set(wrong)

    def read_prompt(self, ids):
        self.reads.append(len(ids))
        self.pos = len(ids)

    def verify_rows(self, ids, targets, commit):
        m = len(ids)
        self.passes.append((self.pos, m, commit))
        nll = np.zeros(m, np.float32) + (np.float32(0.5) if (m in self.wrong and not commit) else np.float32(0.0))
        if commit:
            self.pos += m
        return np.asarray(ids, np.int32), nll

    def reset(self):
        self.pos = 0


class CorrectnessTests(unittest.TestCase):
    """Passes of 64 rows are compared before and after position 4,096 (from there the engine splits the attention into 42 parts and
    passes of 64 rows use kernels of their own)."""

    def test_positions(self):
        self.assertEqual(wc.CHECK_VERSION, 3)
        self.assertEqual((wc.LONG_START, wc.LONG_ROWS, wc.CHECK_IDS), (4064, (64, 64), 4193))
        self.assertLessEqual(wc.CHECK_IDS, wc.CONTEXT + 128)                 # the context the check asks for is enough
        eng = FakeGateEngine()
        g = wc.correctness(eng, list(range(wc.CHECK_IDS)))
        self.assertTrue(g.ok)
        self.assertEqual(eng.reads, [wc.GATE_START, wc.LONG_START])
        self.assertEqual(g.positions, sum(wc.GATE_ROWS) + sum(wc.LONG_ROWS))
        wide = [(p, m) for p, m, commit in eng.passes if not commit]
        self.assertEqual(wide[-2:], [(4064, 64), (4128, 64)])
        self.assertTrue(wide[-2][0] + 1 < 4096 <= wide[-2][0] + 64)          # the first pass crosses position 4,096 ...
        self.assertGreaterEqual(wide[-1][0] + 1, 4096)                        # ... the second lies after it
        self.assertTrue(all(m <= 16 for p, m, commit in eng.passes if commit))
        with self.assertRaises(ValueError):
            wc.correctness(FakeGateEngine(), list(range(wc.CHECK_IDS - 1)))

    def test_a_difference_after_4096_fails(self):
        eng = FakeGateEngine(wrong={64})
        g = wc.correctness(eng, list(range(wc.CHECK_IDS)), more=((wc.LONG_START, wc.LONG_ROWS),))
        self.assertFalse(g.ok)
        self.assertEqual(g.per_rows[64], (0, 64 * 3))                          # one pass of 64 before 4,096, two after
        g = wc.correctness(FakeGateEngine(wrong={64}), list(range(wc.CHECK_IDS)), rows=(17,), more=((wc.LONG_START, (64,)),))
        self.assertEqual((g.nll_diff, g.positions), (64, 81))


class SettingTests(unittest.TestCase):
    def test_mode(self):
        self.assertEqual(wc.mode_of(None, {}), "auto")
        self.assertEqual(wc.mode_of(None, {"YASAKANI_WIDE": "ON"}), "on")
        self.assertEqual(wc.mode_of("off", {"YASAKANI_WIDE": "on"}), "off")
        with self.assertRaises(YasakaniError):
            wc.mode_of(None, {"YASAKANI_WIDE": "yes"})

    def test_engine_env(self):
        self.assertEqual(wc.engine_env("auto"), {"YSKN_WIDE": "1"})     # 0.1.7: auto = on (up to 0.1.6: nothing, RTX 30 / 40 off)
        self.assertEqual([wc.engine_env(m)["YSKN_WIDE"] for m in ("off", "on", "force")], ["0", "1", "2"])

    def test_profile_keeps_test_settings(self):
        env = {"YSKN_TEST_CODE_ARCH": "86", "YSKN_OLD": "1", "PATH": "x"}
        removed = RuntimeProfile({"YSKN_Q35_KV8": "1"}).apply(env)
        self.assertEqual(removed, ["YSKN_OLD"])
        self.assertEqual(env["YSKN_TEST_CODE_ARCH"], "86")

    def test_state_from_array(self):
        a = np.zeros(27, np.int32)
        a[:10] = [1, 86, 86, 82, 1, 1, 16, 0, 1, 0]
        a[14:19] = [3000, 24576, 0, 1, 0]
        a[19:23] = np.array([0x12345678, 0x9abcdef0, 0x0fedcba9, 0x87654321], np.uint32).view(np.int32)
        a[24], a[26] = 86, 1
        st = wc.WideState.from_array(a)
        self.assertEqual((st.code_arch, st.cc_text, st.allowed, st.mask, st.pref, st.native, st.uuid), (86, "8.6", True, 0, 1, False, UUID))
        self.assertEqual(st.checkable_widths, [32, 48, 64])
        self.assertEqual(st.code_text, "sm_86")
        a[24] = 120
        self.assertIn("compiled at load time for sm_120", wc.WideState.from_array(a).code_text)
        a[23] = 1                                                  # a test build's override: no word about compiling at load time
        t = wc.WideState.from_array(a).code_text
        self.assertEqual(t, "sm_86 (set by YSKN_TEST_CODE_ARCH; machine code sm_120)")


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ywc_"))
        self.now = [1_800_000_000.0]
        self.cache = wc.Cache(self.tmp, now=lambda: self.now[0])

    def test_put_get_and_key(self):
        self.assertTrue(self.cache.put("k1", {"a": 1}, {"mask": 3}))
        self.assertEqual(self.cache.get("k1")["result"]["mask"], 3)
        self.assertIsNone(self.cache.get("k2"))
        self.assertEqual([p.name for p in self.tmp.iterdir()], [wc.CACHE_FILE])   # no temporary file left

    def test_expiry_and_recheck(self):
        self.cache.put("k1", {}, {"mask": 3})
        self.now[0] += wc.CACHE_DAYS * 86400 + 1
        self.assertIsNone(self.cache.get("k1"))
        self.now[0] -= wc.CACHE_DAYS * 86400 + 1
        self.assertTrue(self.cache.mark_recheck("k1"))
        self.assertIsNone(self.cache.get("k1"))
        self.assertFalse(self.cache.mark_recheck("nope"))

    def test_newest_first_and_bounded(self):
        for i in range(wc.CACHE_KEEP + 5):
            self.now[0] += 1
            self.cache.put("k%d" % i, {}, {"mask": i % 8})
        keys = [e["key"] for e in self.cache.load()]
        self.assertEqual(len(keys), wc.CACHE_KEEP)
        self.assertEqual(keys[0], "k%d" % (wc.CACHE_KEEP + 4))

    def test_broken_file(self):
        (self.tmp / wc.CACHE_FILE).write_text("{not json", encoding="utf-8")
        self.assertEqual(self.cache.load(), [])
        self.assertTrue(self.cache.put("k1", {}, {"mask": 1}))
        (self.tmp / wc.CACHE_FILE).write_text(json.dumps({"format": "other", "entries": []}), encoding="utf-8")
        self.assertEqual(self.cache.load(), [])

    def test_unwritable_folder(self):
        blocker = self.tmp / "file"
        blocker.write_text("x")
        c = wc.Cache(blocker / "sub")
        self.assertFalse(c.put("k", {}, {"mask": 1}))
        self.assertIn("could not be kept", c.warning)

    def test_cache_dir(self):
        self.assertEqual(wc.cache_dir({"YASAKANI_CACHE_DIR": "/x/y"}), Path("/x/y"))
        if sys.platform != "win32":
            self.assertEqual(wc.cache_dir({"XDG_CACHE_HOME": "/c"}), Path("/c/yasakani"))
        else:
            self.assertEqual(wc.cache_dir({"LOCALAPPDATA": "C:/L"}), Path("C:/L/yasakani"))


class SmiTests(unittest.TestCase):
    def setUp(self):
        self.env = {"YASAKANI_NVIDIA_SMI": "nvidia-smi-test"}

    def test_fields_by_uuid(self):
        run = FakeRun()
        f = wc.smi_fields(UUID, wc.KEY_FIELDS, run, self.env)
        self.assertEqual((f["driver_version"], f["driver_model.current"]), ("616.92", "WDDM"))
        self.assertTrue(all("--id=" + UUID in c for c in run.calls))
        self.assertIsNone(wc.smi_fields("", wc.KEY_FIELDS, run, self.env))
        self.assertIsNone(wc.smi_fields(UUID, wc.KEY_FIELDS, FakeRun(known=False), self.env))

    def test_display_attached_not_display_mode(self):
        # display_mode is deprecated in driver 616.92 ("[Requested functionality has been deprecated]")
        self.assertIn("display_attached", wc.KEY_FIELDS)
        self.assertNotIn("display_mode", wc.KEY_FIELDS)
        run = FakeRun()
        f = wc.smi_fields(UUID, wc.KEY_FIELDS, run, self.env)
        self.assertEqual(f["display_attached"], "Yes")
        self.assertEqual(len(run.calls), 1)                        # one query: every field is known to the driver

    def test_fields_one_by_one(self):
        run = FakeRun(fail_full=True, fields={"driver_model.current": "[N/A]"})
        f = wc.smi_fields(UUID, wc.KEY_FIELDS, run, self.env)
        self.assertEqual(f["power.limit"], "350.00")
        self.assertIsNone(f["driver_model.current"])

    def test_busy(self):
        nap = lambda s: None
        self.assertIsNone(wc.busy_reason(UUID, FakeRun(util=3), self.env, sleep=nap))
        self.assertIn("load is 55", wc.busy_reason(UUID, FakeRun(util=55), self.env, sleep=nap))
        self.assertIn("other program", wc.busy_reason(UUID, FakeRun(apps="4242, 9000\n"), self.env, sleep=nap))
        self.assertIsNone(wc.busy_reason(UUID, FakeRun(apps="%d, 20000\n" % os.getpid()), self.env, sleep=nap))
        # Windows (WDDM): every program that draws with the GPU is listed, without memory: the list cannot tell
        self.assertIsNone(wc.busy_reason(UUID, FakeRun(apps="10600, [N/A]\n14800, [N/A]\n"), self.env, sleep=nap))
        self.assertIsNone(wc.busy_reason(UUID, FakeRun(apps="4242, 100\n"), self.env, sleep=nap))   # a little memory: not counted
        self.assertIn("no GPU load", wc.busy_reason(UUID, FakeRun(known=False), self.env, sleep=nap))


def _rep(**over):
    m = dict(package="yasakani 0.3.5", os="Windows 10.0.26200", python="3.12.0",
             library=dict(file="yskn.dll", sha256="0123456789abcdef", build_info="cuda=13020;wide=1"),
             gpu=dict(name="NVIDIA GeForce RTX 3090", cc="8.6", total_mib=24576, sms=82, code="sm_86", driver="616.92",
                      driver_model="WDDM", power_limit_w="350.00", max_sm_mhz="2100", max_mem_mhz="9751", display_attached="Yes",
                      display_active="Enabled", pstate="P2", temperature_c="55", sm_mhz="1800", slowdown_reasons="0x0", free_mib_now=3000),
             model=dict(id="qwen3.8-27b", weights="qwen38_27b_speed.yskn", context=65536),
             wide=dict(setting="on", status="measured", mask=6, widths="48, 64 rows", reason="", cache="measured now", check_seconds=21.0,
                       free_after_mib=2900))
    mask, ws = wc.decide(timing(r32=1.8), False)
    res = wc.WideResult(status="measured", mask=mask, widths=ws, timing=timing(r32=1.8), times_us=[11000, 19800, 21340, 24970],
                        gate=wc.Gate(positions=335, per_rows={m: (0, 0) for m in wc.GATE_ROWS}))
    rows = []
    for k, col in enumerate(["off", "checked", "all", "all", "checked", "off"]):
        for title, tps in (("code edit", 100.0), ("JSONL edit", 120.0)):
            rows.append(dict(column=col, block=k, task=title, title=title, tokens_per_s=tps + (20 if col != "off" else 0), tokens=900,
                             wide_passes=0 if col == "off" else 12, wide_backoffs=0, ids_sha256="ab" * 32, same=True))
    rep = dict(machine=m, check=res.summary(), bench=dict(masks={"off": 0, "checked": 6, "all": 7}, rows=rows, same_as={}),
               selfcheck=dict(reference=[dict(name="short-answer", result="differs", first_diff=7)],
                              close=dict(lines=[], failures=[]),
                              consistency=dict(lines=["x"], failures=[], wide_checked=True, positions=1300, argmax_diff=0, nll_diff=0,
                                               wide_passes=20, max_rows=64)), seconds=600)
    rep.update(over)
    return rep


class ReportTextTests(unittest.TestCase):
    def test_markdown(self):
        from yasakani.report import render
        text = render(_rep())
        for want in ("### YASAKANI report (yasakani 0.3.5)", "| GPU | NVIDIA GeForce RTX 3090, compute capability 8.6",
                     "| 32 |", "1.60", "| no |", "**Correctness**", "same most likely ids and NLL bit for bit at 335 positions",
                     "| code edit | 100.0 | 120.0 | 120.0 |", "same output", "| yes |", "passes of 64 vs 16: same predictions",
                     "display attached Yes"):
            self.assertIn(want, text)

    def test_column_labels(self):
        # the wide passes and back-offs are per column (sums of its runs), and the header says which column is which
        from yasakani.report import render
        text = render(_rep())
        self.assertIn("| wide passes (off/checked/all) | back-offs (off/checked/all) |", text)
        self.assertIn("| 0/24/24 | 0/0/0 |", text)
        rep = _rep()
        rep["bench"]["same_as"] = {"all": "checked"}
        self.assertIn("| wide passes (off/checked) | back-offs (off/checked) |", render(rep))

    def test_no_private_data(self):
        from yasakani.report import render
        text = render(_rep())
        self.assertNotIn("GPU-", text)
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]|/home/|/root/|Users", text))

    def test_different_outputs_are_flagged(self):
        from yasakani.report import render
        rep = _rep()
        rep["bench"]["rows"][3]["ids_sha256"] = "cd" * 32
        self.assertIn("NO (2 different)", render(rep))


class CliTests(unittest.TestCase):
    def test_parser(self):
        from yasakani.cli import build_parser
        a = build_parser().parse_args(["serve", "--wide", "on", "--wide-recheck"])
        self.assertEqual((a.wide, a.wide_recheck), ("on", True))
        a = build_parser().parse_args(["bench", "--edit"])
        self.assertEqual((a.wide, a.wide_recheck), (None, False))
        a = build_parser().parse_args(["report", "--no-bench", "--rounds", "2"])
        self.assertEqual((a.wide, a.wide_recheck, a.rounds, a.seed0, a.no_bench), ("on", True, 2, 1, True))
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            build_parser().parse_args(["chat", "--wide", "maybe"])

    def test_version_line_of_a_release(self):
        import io
        from yasakani.cli import build_parser
        out = io.StringIO()
        with mock.patch("sys.stdout", out), self.assertRaises(SystemExit):
            build_parser().parse_args(["--version"])
        self.assertEqual(out.getvalue().strip(), "yasakani 0.3.5")      # a release: no "(test build)" as in 0.1.7.dev1

    def test_report_time_in_help(self):
        import io
        from yasakani.cli import build_parser
        out = io.StringIO()
        with mock.patch("sys.stdout", out), self.assertRaises(SystemExit):
            build_parser().parse_args(["--help"])
        text = " ".join(out.getvalue().split())
        self.assertIn("about 1.5 minutes on the RTX 5090, longer on other cards", text)
        self.assertNotIn("10 to 30", text)

    def test_version_is_pep440(self):
        import re as _re
        from yasakani import __version__
        self.assertEqual(__version__, "0.3.5")
        self.assertIsNotNone(_re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", __version__))


class StubTests(unittest.TestCase):
    """The start-up flow on the stub library built with -DSTUB_WIDE."""

    KNOBS = ("STUB_WIDE_ARCH", "STUB_WIDE_T16", "STUB_WIDE_R32", "STUB_WIDE_R48", "STUB_WIDE_R64", "STUB_WIDE_SMP", "STUB_WIDE_FREE",
             "STUB_WIDE_JITTER", "STUB_TF_WIDE_NLL_SHIFT", "STUB_GEN_BACKOFFS", "STUB_GEN_WIDE_PASSES", "YASAKANI_WIDE", "YSKN_WIDE",
             "YASAKANI_NVIDIA_SMI")

    @classmethod
    def setUpClass(cls):
        cls.stub = Stub("wide")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ywcs_"))
        self.saved = {k: os.environ.get(k) for k in self.KNOBS}
        for k in self.KNOBS:
            os.environ.pop(k, None)
        self.cache = wc.Cache(self.tmp / "cache")
        self.fast = mock.patch.multiple(wc, SLICE_S=0.002, WARM_S=0.002)
        self.fast.start()

    def tearDown(self):
        self.fast.stop()
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def open(self, arch=86, wide=None, **knobs):
        os.environ["STUB_WIDE_ARCH"] = str(arch)
        for k, v in knobs.items():
            os.environ[k] = str(v)
        return Engine(make_pack(self.tmp, "m%d" % len(list(self.tmp.iterdir()))), max_ctx=8192, lib=self.stub.lib, wide=wide)

    def setup(self, eng, run=None, busy=None, **kw):
        ids = list(range(1000, 1000 + wc.CHECK_IDS))
        return wc.setup(eng, ids, cache=self.cache, run=run or FakeRun(), environ={"YASAKANI_NVIDIA_SMI": "x"},
                        busy=busy or (lambda u: None), **kw)

    def test_auto_on_rtx30_measures(self):
        """0.1.7: the default (auto) opens on RTX 30 / 40 the widths the check finds correct and faster (up to 0.1.6 it left them off)."""
        with self.open(86) as eng:
            st = eng.wide_state()
            self.assertEqual((st.allowed, st.mask, st.xw_rows), (True, 0, 16))     # allowed, closed until the check opens widths
            r = self.setup(eng)
            self.assertEqual((r.status, r.mask), ("measured", 7))
            self.assertTrue(r.gate.ok)
            self.assertEqual(eng.wide_state().mask, 7)
        with self.open(89, "off") as eng:                             # --wide off still turns them off
            st = eng.wide_state()
            self.assertEqual((st.allowed, st.mask, st.xw_rows), (False, 0, 16))
            r = self.setup(eng)
            self.assertEqual((r.status, r.mask), ("off", 0))
            self.assertNotIn("--wide on", r.line)

    def test_on_rtx30_measures_and_keeps(self):
        with self.open(86, "on") as eng:
            self.assertEqual(eng.wide_state().mask, 0)           # closed until the check opens widths
            r = self.setup(eng)
            self.assertEqual((r.status, r.mask), ("measured", 7))
            self.assertTrue(r.gate.ok)
            self.assertEqual(r.gate.positions, sum(wc.GATE_ROWS) + sum(wc.LONG_ROWS))
            self.assertEqual([self.stub.opt(eng, k) for k in (39, 40, 41, 42)], [11000, 15290, 21340, 24970])
            self.assertEqual(eng.wide_state().mask, 7)
        with self.open(86, "on") as eng:                        # the same GPU, driver, library and model: the kept result
            r = self.setup(eng)
            self.assertEqual((r.status, r.mask, r.cache_hit), ("cached", 7, True))
            self.assertEqual(self.stub.opt(eng, 42), 24970)
            r = self.setup(eng, recheck=True)
            self.assertEqual(r.status, "measured")

    def test_on_rtx30_leaves_out_a_slow_width(self):
        with self.open(89, "on", STUB_WIDE_R32=1.75) as eng:
            r = self.setup(eng)
            self.assertEqual((r.status, r.mask), ("measured", 6))
            self.assertIn("32 1.75>1.60", r.line)

    def test_mismatch_closes_everything(self):
        with self.open(86, "on", STUB_TF_WIDE_NLL_SHIFT=0.001) as eng:
            r = self.setup(eng)
            self.assertEqual((r.status, r.mask), ("mismatch", 0))
            self.assertEqual(r.gate.nll_diff, sum(wc.GATE_ROWS) + sum(wc.LONG_ROWS))
            self.assertEqual(eng.wide_state().mask, 0)
            self.assertEqual(self.cache.load()[0]["result"]["status"], "mismatch")

    def test_mismatch_on_rtx50_stops_the_exact_sum_gemv(self):
        # should 7: on 12.x the passes of 17..64 rows (ends of prompt reads) go back to 16-row sums (set_opt 31 = 0), also from a kept result
        with self.open(120, STUB_TF_WIDE_NLL_SHIFT=0.001) as eng:
            self.assertEqual(eng.wide_state().xw_rows, 64)
            r = self.setup(eng)
            self.assertEqual((r.status, r.mask), ("mismatch", 0))
            st = eng.wide_state()
            self.assertEqual((st.mask, st.xw_rows), (0, 16))
            self.assertEqual(self.stub.opt(eng, 31), 0)
        with self.open(120, STUB_TF_WIDE_NLL_SHIFT=0.001) as eng:
            r = self.setup(eng)
            self.assertEqual((r.status, r.mask, r.cache_hit), ("mismatch", 0, True))
            self.assertEqual((eng.wide_state().mask, eng.wide_state().xw_rows), (0, 16))
        os.environ.pop("STUB_TF_WIDE_NLL_SHIFT")
        with self.open(120) as eng:                                  # no mismatch: set_opt 31 is not touched
            self.setup(eng, recheck=True)
            self.assertEqual(eng.wide_state().xw_rows, 64)

    def test_selfcheck_fails_after_a_mismatch(self):
        # should 1: the self-check must fail, not pass by leaving out the comparison of 64 and 16 positions
        from yasakani import selfcheck as sc

        class Fmt:
            def prompt(self, messages, **kw):
                return np.arange(100, 140, dtype=np.int32), False

        for arch in (86, 120):
            with self.open(arch, "on", STUB_TF_WIDE_NLL_SHIFT=0.001) as eng:
                eng.wide_result = self.setup(eng)
                self.assertEqual(eng.wide_result.status, "mismatch")
                os.environ.pop("STUB_TF_WIDE_NLL_SHIFT")
                res = sc.consistency(eng, Fmt())
                self.assertTrue(res.failures(), arch)
                self.assertIn("do not give the same predictions", res.failures()[0])
                self.assertFalse(res.wide_checked)

    def test_check_reads_in_pieces(self):
        # M2: no single prompt read of 1,024 ids or more (the engine would set up and keep a large work area)
        from yasakani.engine import READ_CHUNK
        self.assertEqual(READ_CHUNK, 1023)
        with self.open(86, "on") as eng:
            n0 = self.stub.d.stub_prefill_calls()
            eng.read_prompt(list(range(4096)))
            self.assertEqual(self.stub.d.stub_prefill_calls() - n0, 5)       # 4 x 1023 + 4
            self.assertEqual(eng.position(), 4096)
            r = self.setup(eng)
            self.assertEqual(r.status, "measured")
            self.assertLessEqual(self.stub.d.stub_prefill_max(), READ_CHUNK)

    def test_busy_keeps_nothing(self):
        with self.open(86, "on") as eng:
            r = self.setup(eng, busy=lambda u: "the GPU load is 40 %")
            self.assertEqual((r.status, r.mask), ("busy", 0))
        with self.open(120) as eng:
            r = self.setup(eng, busy=lambda u: "2 other program(s) hold GPU memory on this GPU")
            self.assertEqual((r.status, r.mask), ("busy", 7))      # RTX 50: as in 0.1.6
        self.assertEqual(self.cache.load(), [])

    def test_unknown_gpu(self):
        with self.open(86, "on") as eng:
            r = self.setup(eng, run=FakeRun(known=False))
            self.assertEqual((r.status, r.mask), ("unknown-gpu", 0))

    def test_rtx50_auto(self):
        with self.open(120, STUB_WIDE_R48=2.8) as eng:          # over the RTX 50 limit of 0.3.4: 0.9 x 48 / 16 = 2.70
            st = eng.wide_state()
            self.assertEqual((st.allowed, st.mask, st.native), (True, 7, True))
            r = self.setup(eng)
            self.assertEqual((r.status, r.mask), ("measured", 5))

    def test_force_and_off(self):
        with self.open(86, "force") as eng:
            r = self.setup(eng)
            self.assertEqual((r.status, r.mask), ("force", 7))
            self.assertIn("not checked", r.line)
        with self.open(120, "off") as eng:
            r = self.setup(eng)
            self.assertEqual((r.status, r.mask), ("off", 0))

    def test_old_library(self):
        plain = Stub()
        with Engine(make_pack(self.tmp, "old"), max_ctx=8192, lib=plain.lib, wide="on") as eng:
            r = wc.setup(eng, None)
            self.assertEqual(r.status, "old-library")
            self.assertIn("no effect", r.line)
        with Engine(make_pack(self.tmp, "old2"), max_ctx=8192, lib=plain.lib) as eng:
            self.assertIsNone(wc.setup(eng, None))

    def test_memory_guard(self):
        with self.open(86, "on", STUB_WIDE_FREE=64) as eng:
            r = self.setup(eng)
            self.assertEqual(r.mask, 7)
            wc.after_prepare(eng, r)
            self.assertEqual((r.status, r.mask, r.free_after_mib), ("memory", 0, 64))
            self.assertEqual(eng.wide_state().mask, 0)

    def test_short_context(self):
        os.environ["STUB_WIDE_ARCH"] = "86"
        with Engine(make_pack(self.tmp, "short"), max_ctx=4096, lib=self.stub.lib, wide="on") as eng:
            r = self.setup(eng)
            self.assertEqual((r.status, r.mask), ("no-check", 0))
            self.assertIn("shorter than the check needs", r.line)

    def test_without_a_check(self):
        with self.open(86, "on") as eng:
            r = wc.setup(eng, None, check=False, cache=self.cache, run=FakeRun(), environ={"YASAKANI_NVIDIA_SMI": "x"})
            self.assertEqual((r.status, r.mask), ("no-check", 0))

    def test_backoffs_mark_a_recheck(self):
        # M3: the next start checks again only when at least 10 of every 100 wide passes backed off, over at least 128 wide passes
        with self.open(86, "on") as eng:
            r = self.setup(eng)
            self.assertEqual(r.status, "measured")
            os.environ["STUB_GEN_WIDE_PASSES"], os.environ["STUB_GEN_BACKOFFS"] = "100", "8"
            eng.generate([5, 6, 7], 8)                              # 8 back-offs (the old rule) in 100 wide passes: too few passes
            self.assertFalse(self.cache.load()[0]["recheck"])
            eng.generate([5, 6, 7], 8)                              # 16 in 200: 8 %
            self.assertFalse(self.cache.load()[0]["recheck"])
            self.assertIsNotNone(self.cache.get(r.key))
            os.environ["STUB_GEN_BACKOFFS"] = "20"
            eng.generate([5, 6, 7], 8)                              # 36 in 300: 12 %
            self.assertTrue(self.cache.load()[0]["recheck"])
            self.assertIsNone(self.cache.get(r.key))

    def test_backoffs_without_wide_passes_are_not_counted(self):
        with self.open(86, "on") as eng:
            r = self.setup(eng)
            os.environ["STUB_GEN_WIDE_PASSES"], os.environ["STUB_GEN_BACKOFFS"] = "0", "50"
            for _ in range(4):
                eng.generate([5, 6, 7], 8)
            self.assertEqual((eng._wide_passes_seen, eng._wide_backoffs), (0, 0))
            self.assertFalse(self.cache.load()[0]["recheck"])
            self.assertIsNotNone(self.cache.get(r.key))

    def test_key_has_display_attached(self):
        with self.open(86, "on") as eng:
            self.setup(eng)
            self.assertEqual(self.cache.load()[0]["fields"]["display"], "Yes")
            self.assertEqual(self.cache.load()[0]["fields"]["check"], wc.CHECK_VERSION)

    def test_report_machine_reads_display_attached(self):
        from yasakani import report
        with self.open(120) as eng:
            st = eng.wide_state()
            got = {"display_attached": "Yes", "display_mode": None, "name": "NVIDIA GeForce RTX 5090"}
            with mock.patch.object(wc, "smi_fields", return_value=got):
                m = report.machine(eng, st, None, eng.pack)
            self.assertEqual(m["gpu"]["display_attached"], "Yes")
            self.assertEqual(m["library"]["build_info"], "cuda=0;wide=1;stub=1")

    def test_report_after_a_mismatch_opens_no_width(self):
        from yasakani import report

        class Fmt:
            def encode(self, text):
                return [1, 2, 3]

        class Runner:
            pass

        with self.open(86, "on", STUB_TF_WIDE_NLL_SHIFT=0.001) as eng:
            res = self.setup(eng)
            eng.wide_result = res
            runner = Runner()
            runner.engine, runner.fmt = eng, Fmt()
            b = report.bench_columns(eng, runner, [], res, eng.wide_state())
            self.assertEqual(b["masks"], {"off": 0, "checked": 0, "all": 0})
            self.assertEqual(eng.wide_state().mask, 0)

    def test_warmup_builds_graphs_after_the_check(self):
        n0 = self.stub.d.stub_prepare_calls()
        os.environ["YASAKANI_NVIDIA_SMI"] = "none"                # nvidia-smi gives nothing: RTX 30 stays closed, nothing measured
        with mock.patch.dict(os.environ, {"YASAKANI_CACHE_DIR": str(self.tmp / "c2")}):
            with self.open(86, "on") as eng:
                lines = []
                eng.warmup([[1, 2, 3]], n_gen=4, wide_ids=list(range(5000)), say=lines.append)
                self.assertEqual(eng.wide_result.status, "unknown-gpu")
                self.assertEqual(self.stub.d.stub_prepare_calls(), n0 + 2)   # greedy and sampled (Engine.prepare_wide)
                self.assertTrue(lines and lines[0].startswith("wide copy passes"))

    def test_selfcheck_skips_64_when_closed(self):
        from yasakani import selfcheck as sc

        class Fmt:
            def prompt(self, messages, **kw):
                return np.arange(100, 140, dtype=np.int32), False

        with self.open(86, "on") as eng:
            res = sc.consistency(eng, Fmt())
            self.assertFalse(res.wide_checked)
            self.assertTrue(any("no wide copy pass is open" in n for n in res.notes))


if __name__ == "__main__":
    unittest.main()
