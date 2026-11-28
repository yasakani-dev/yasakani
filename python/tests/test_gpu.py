# SPDX-License-Identifier: Apache-2.0
"""0.1.3: which GPU, what 0.1.3 does on it, and the context that fits its memory (yasakani/gpu.py and the quickstart's copy of the rule).

nvidia-smi is never run for real here: its output is given as text, or a stand-in program prints it.  The memory sizes of the cards are
what nvidia-smi reports for them (MiB); the rule's numbers are those of the published model files.
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tests import support
from yasakani import gpu as G
from yasakani._native import YasakaniError

ROOT = Path(__file__).resolve().parent.parent
FASTEST = (10311344128, 1164969472, 1411010584)       # the published files: weights, draft model, its output head
QUALITY = (12365242368, 1164969472, 1411010584)
MAXQ = (15723962368, 1164969472, 1411010584)

# index, name, compute capability, memory.total (MiB) -> what the quickstart does, whether the quality model fits
CARDS = [
    ("NVIDIA GeForce RTX 5090", "12.0", 32607, "tested", True),
    ("NVIDIA GeForce RTX 4090", "8.9", 24564, "other", True),
    ("NVIDIA GeForce RTX 3090 Ti", "8.6", 24564, "other", True),
    ("NVIDIA GeForce RTX 3090", "8.6", 24576, "other", True),
    ("NVIDIA GeForce RTX 5080", "12.0", 16303, "other", False),
    ("NVIDIA GeForce RTX 5070 Ti", "12.0", 16303, "other", False),
    ("NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, "other", False),
    ("NVIDIA GeForce RTX 4080 SUPER", "8.9", 16376, "other", False),
    ("NVIDIA GeForce RTX 4080", "8.9", 16376, "other", False),
    ("NVIDIA GeForce RTX 4070 Ti SUPER", "8.9", 16376, "other", False),
    ("NVIDIA GeForce RTX 4060 Ti", "8.9", 16380, "other", False),       # the 16 GB card
    ("NVIDIA GeForce RTX 4060 Ti", "8.9", 8188, "small", False),        # the 8 GB card of the same name
    ("NVIDIA GeForce RTX 5070", "12.0", 12227, "small", False),
    ("NVIDIA GeForce RTX 3060", "8.6", 12288, "small", False),
    ("NVIDIA GeForce RTX 5090 Laptop GPU", "12.0", 24463, "other", True),
    ("NVIDIA GeForce RTX 5090 D", "12.0", 32607, "other", True),
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "12.0", 97887, "other", True),
    ("NVIDIA RTX A6000", "8.6", 49140, "other", True),
    ("NVIDIA GeForce RTX 2080 Ti", "7.5", 11264, "old", False),
    ("NVIDIA GeForce GTX 1080 Ti", "6.1", 11264, "old", False),
    ("NVIDIA A100 80GB PCIe", "8.0", 81920, "old", True),
    ("NVIDIA H100 PCIe", "9.0", 81559, "nocode", True),
    ("NVIDIA B200", "10.0", 183359, "nocode", True),
]


def smi_line(i, name, cc, total, used=1000, uuid=None):
    return "%d, %s, %s, %s, %d, %d, 616.92" % (i, uuid or "GPU-%08d-0000-0000-0000-000000000000" % i, name, cc, total, used)


class FakeFile:
    def __init__(self, name, size):
        self.name, self._size = name, size

    def stat(self):
        return os.stat_result((0, 0, 0, 0, 0, 0, self._size, 0, 0, 0))


class FakePack:
    def __init__(self, sizes, max_ctx=262144):
        self.weights, self.draft, self.draft_head = (FakeFile(n, s) for n, s in zip(("w.yskn", "bdraft2_k8.yskn", "bdraft2_k8.shead"), sizes))
        self.max_ctx = max_ctx


def det_of(*rows, environ=None):
    gpus = G.parse("\n".join(rows) + "\n")
    return G.detect(0, environ=environ or {}, gpus=gpus)


def load_quickstart():
    """quickstart/quickstart.py of the repository (or the file YASAKANI_QUICKSTART names); None when it is not there (the engine zip)."""
    for p in (os.environ.get("YASAKANI_QUICKSTART"), ROOT.parent / "quickstart" / "quickstart.py"):
        if p and Path(p).is_file():
            spec = importlib.util.spec_from_file_location("yasakani_quickstart_under_test", str(p))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    return None


class ParseTests(unittest.TestCase):
    def test_rows(self):
        gs = G.parse(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607, 2644) + "\n" + smi_line(1, "NVIDIA GeForce RTX 3090", "8.6", 24576, 15))
        self.assertEqual(len(gs), 2)
        self.assertEqual((gs[0].index, gs[0].compute_cap, gs[0].total_mib, gs[0].used_mib, gs[0].free_mib), (0, (12, 0), 32607, 2644, 29963))
        self.assertEqual((gs[1].name, gs[1].cc_text, gs[1].driver), ("NVIDIA GeForce RTX 3090", "8.6", "616.92"))
        self.assertTrue(gs[0].uuid.startswith("GPU-"))

    def test_missing_values_and_noise(self):
        gs = G.parse("0, [N/A], NVIDIA GeForce RTX 4090, [N/A], 24564, [N/A], 581.57\nNo devices were found\n\n")
        self.assertEqual(len(gs), 1)
        g = gs[0]
        self.assertEqual((g.uuid, g.compute_cap, g.used_mib, g.free_mib, g.arch), (None, None, None, None, "unknown"))
        self.assertEqual(G.parse("No devices were found\n"), [])
        self.assertEqual(G.parse(""), [])

    def test_query_falls_back_without_compute_cap(self):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd[1])
            if "compute_cap" in cmd[1]:
                return subprocess.CompletedProcess(cmd, 2, stdout='Field "compute_cap" is not a valid field to query.\n', stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="0, GPU-1, NVIDIA GeForce RTX 3090, 24576, 500, 470.10\n", stderr="")
        gs = G.query({"YASAKANI_NVIDIA_SMI": "/x/nvidia-smi"}, run=run)
        self.assertEqual(len(calls), 2)
        self.assertEqual((gs[0].name, gs[0].compute_cap, gs[0].total_mib), ("NVIDIA GeForce RTX 3090", None, 24576))

    def test_query_without_nvidia_smi(self):
        self.assertIsNone(G.query({"YASAKANI_NVIDIA_SMI": "none"}))
        self.assertIsNone(G.query({"YASAKANI_NVIDIA_SMI": ""}))

        def boom(cmd, **kw):
            raise FileNotFoundError(cmd[0])
        self.assertIsNone(G.query({"YASAKANI_NVIDIA_SMI": "/nonexistent/nvidia-smi"}, run=boom))
        with mock.patch.object(G.shutil, "which", return_value=None):
            self.assertIsNone(G.query({}))
        self.assertFalse(G.detect(0, environ={"YASAKANI_NVIDIA_SMI": "none"}).known)


class DecisionTableTests(unittest.TestCase):
    def test_cards(self):
        for name, cc, total, qs, quality in CARDS:
            g = G.parse(smi_line(0, name, cc, total))[0]
            with self.subTest(card=name, total=total):
                self.assertEqual(g.tested, qs == "tested")
                self.assertEqual(g.arch, {"old": "old", "nocode": "no-code"}.get(qs, "ok"))
                line = G.describe(g)
                if qs == "tested":
                    self.assertIn("tested and tuned", line)
                elif qs == "other":
                    self.assertIn("not optimized or tested", line)
                    self.assertEqual("fastest model only" in line, not quality)
                elif qs == "small":
                    self.assertIn("too small", line)
                else:
                    self.assertIn("no code", line)

    def test_untested_note(self):
        self.assertIsNone(G.untested_note(det_of(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607))))
        self.assertIsNone(G.untested_note(G.Detected()))                         # nothing known: no note
        n = G.untested_note(det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376)))
        self.assertIn("RTX 4080 (compute capability 8.9)", n)
        self.assertIn("not optimized or tested", n)
        self.assertIn("RTX 5090 only", n)
        n = G.untested_note(det_of(smi_line(0, "NVIDIA GeForce RTX 2080 Ti", "7.5", 11264)))
        self.assertIn("will probably not start", n)


class SelectTests(unittest.TestCase):
    TWO = [smi_line(0, "NVIDIA GeForce RTX 3090", "8.6", 24576, 300, "GPU-aaaa1111-0000"),
           smi_line(1, "NVIDIA GeForce RTX 5090", "12.0", 32607, 2000, "GPU-bbbb2222-0000")]

    def gpus(self):
        return G.parse("\n".join(self.TWO))

    def test_one_gpu(self):
        gs = self.gpus()[:1]
        self.assertEqual(G.select(gs, 0, {}), gs)
        self.assertEqual(G.select(gs, 1, {}), [])
        self.assertEqual(G.select([], 0, {}), [])

    def test_two_gpus(self):
        gs = self.gpus()
        self.assertEqual(len(G.select(gs, 0, {})), 2)                           # CUDA's default order: not known which
        self.assertEqual(G.select(gs, 1, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})[0].index, 1)
        self.assertEqual(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": "GPU-bbbb2222"})[0].index, 1)     # a unique UUID prefix
        self.assertEqual(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})[0].index, 1)
        self.assertEqual(len(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": "1"})), 2)
        self.assertEqual(G.select(gs, 1, {"CUDA_VISIBLE_DEVICES": "1"}), [])    # only one GPU is visible
        self.assertEqual(G.select(gs, 0, {"CUDA_VISIBLE_DEVICES": ""}), [])

    def test_two_gpus_plan_for_the_smaller(self):
        det = G.detect(0, environ={}, gpus=self.gpus())
        self.assertTrue(det.untested)                                          # one of them is not an RTX 5090
        w = det.worst()
        self.assertEqual((w.total_mib, w.free_mib, w.compute_cap), (24576, 24276, (8, 6)))
        plan = G.plan_context(FakePack(QUALITY), det)
        self.assertIsNone(plan.max_ctx)                                         # the quality model fits both at 262,144
        self.assertTrue(any("2 GPUs" in n for n in plan.notes))
        det = G.detect(0, environ={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, gpus=self.gpus())
        self.assertEqual(det.gpu.index, 0)
        det = G.detect(1, environ={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, gpus=self.gpus())
        self.assertTrue(det.tested)


# issue #20: an RTX A2000 12 GB at PCI 01:00.0 (nvidia-smi's GPU 0, the display) and an RTX 5060 Ti 16 GB at 02:00.0
REPORTED = [smi_line(0, "NVIDIA RTX A2000 12GB", "8.6", 12282, 826, "GPU-a2000000-0000"),
            smi_line(1, "NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 400, "GPU-50600000-0000")]
PCI = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


class DeviceOrderTests(unittest.TestCase):
    """0.1.5: --device N is nvidia-smi's GPU N (the package sets CUDA_DEVICE_ORDER=PCI_BUS_ID), and the plan is made for that GPU."""

    def gpus(self):
        return G.parse("\n".join(REPORTED))

    def test_device_numbers(self):
        gs = self.gpus()
        self.assertEqual(G.select(gs, 0, PCI)[0].name, "NVIDIA RTX A2000 12GB")
        self.assertEqual(G.select(gs, 1, PCI)[0].name, "NVIDIA GeForce RTX 5060 Ti")
        self.assertEqual(G.select(gs, 2, PCI), [])
        self.assertTrue(G.detect(1, environ=PCI, gpus=gs).numbered)
        self.assertEqual(len(G.select(gs, 0, {})), 2)                          # CUDA's own order (0.1.4): not known which
        self.assertFalse(G.detect(0, environ={}, gpus=gs).numbered)
        self.assertTrue(G.detect(0, environ={}, gpus=gs[:1]).numbered)          # one GPU: always known
        det = G.detect(0, environ=dict(PCI, CUDA_VISIBLE_DEVICES="GPU-5060"), gpus=gs)
        self.assertEqual((det.gpu.name, det.numbered), ("NVIDIA GeForce RTX 5060 Ti", False))

    def test_plan_for_the_chosen_gpu(self):
        plan = G.plan_context(FakePack(FASTEST), G.detect(1, environ=PCI, gpus=self.gpus()))
        self.assertEqual((plan.max_ctx, plan.env, plan.free_mib), (98304, dict(G.SMALL_CARD_ENV, YSKN_BDR_SHEAD="0"), 15911))
        self.assertFalse(any("GPUs" in n for n in plan.notes))
        with self.assertRaises(YasakaniError) as cm:                           # GPU 0 is the A2000: it says so, and which GPU has room
            G.plan_context(FakePack(FASTEST), G.detect(0, environ=PCI, gpus=self.gpus()))
        msg = str(cm.exception)
        self.assertIn("(11456 of 12282 MiB free; it needs about 13906 MiB with a context of 16384 tokens)", msg)
        self.assertIn("That is nvidia-smi's GPU 0 (NVIDIA RTX A2000 12GB); GPU 1 (NVIDIA GeForce RTX 5060 Ti, 15911 MiB free) has room "
                      "for it: add --device 1.", msg)
        with self.assertRaises(YasakaniError) as cm:                           # CUDA_DEVICE_ORDER set to something else: why the numbers
            G.plan_context(FakePack(FASTEST), G.detect(0, environ={"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, gpus=self.gpus()))
        self.assertIn("This PC has 2 GPUs and CUDA's device 0 cannot be matched", str(cm.exception))
        self.assertNotIn("add --device", str(cm.exception))
        with self.assertRaises(YasakaniError) as cm:                           # named by UUID: no --device advice (it counts otherwise)
            G.plan_context(FakePack(FASTEST), G.detect(0, environ=dict(PCI, CUDA_VISIBLE_DEVICES="GPU-a2000"), gpus=self.gpus()))
        self.assertIn("That is nvidia-smi's GPU 0 (NVIDIA RTX A2000 12GB).  Close", str(cm.exception))

    def test_one_gpu_message_unchanged(self):
        for env in ({}, PCI):
            with self.assertRaises(YasakaniError) as cm:
                G.plan_context(FakePack(FASTEST), det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376, 14000), environ=env))
            self.assertEqual(str(cm.exception), "the model does not fit in the free GPU memory (2376 of 16376 MiB free; it needs about 13906 "
                             "MiB with a context of 16384 tokens).  Close other programs that use the GPU, or give --max-ctx to try anyway.")

    def test_numbering_line(self):
        gs = self.gpus()
        line = "--device N is GPU N above (PCI bus order)"
        self.assertEqual(G.numbering(PCI), line)                                            # nvidia-smi's list not given: the 0.1.5 line
        self.assertEqual(G.numbering(PCI, gs), line + "; the engine then sees only that GPU")
        no_uuid = G.parse("0, NVIDIA RTX A2000 12GB\n1, NVIDIA GeForce RTX 5060 Ti\n", ("index", "name"))
        self.assertEqual(G.numbering(PCI, no_uuid), line + "; nvidia-smi gave no UUIDs, so the engine also sees the other GPUs")
        mixed = G.parse("\n".join([REPORTED[0], smi_line(1, "NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 400, "[N/A]")]))
        self.assertEqual(G.numbering(PCI, mixed), line + "; the engine then sees only that GPU, except GPU 1 (nvidia-smi gave no UUID): "
                                                         "then it also sees the other GPUs")
        self.assertIn("CUDA_VISIBLE_DEVICES=GPU-50600000", G.numbering(dict(PCI, CUDA_VISIBLE_DEVICES="GPU-50600000"), gs))
        self.assertIn("CUDA_DEVICE_ORDER=FASTEST_FIRST", G.numbering({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, gs))


class VisibleOnlyTests(unittest.TestCase):
    """0.1.6 (issue #20, second report): with two GPUs visible, the engine failed with "graph instantiate failed" on the GPU that was
    not CUDA's device 0, and passed with only that GPU visible.  So the command line makes the GPU of --device the only one CUDA sees
    (CUDA_VISIBLE_DEVICES = its UUID) and opens the engine as device 0; the plan and the messages still name nvidia-smi's GPU N."""

    def gpus(self):
        return G.parse("\n".join(REPORTED))

    def test_which_gpu(self):
        gs = self.gpus()
        for device, uuid in ((1, "GPU-50600000-0000"), (0, "GPU-a2000000-0000")):
            g = G.visible_only(G.detect(device, environ=PCI, gpus=gs), PCI)
            self.assertEqual((g.index, g.uuid), (device, uuid))
        self.assertIsNone(G.visible_only(G.detect(2, environ=PCI, gpus=gs), PCI))           # no such GPU: the engine says so, as before
        self.assertIsNone(G.visible_only(G.detect(0, environ=PCI, gpus=gs[:1]), PCI))       # one GPU: nothing changes
        for env in (dict(PCI, CUDA_VISIBLE_DEVICES="1"), dict(PCI, CUDA_VISIBLE_DEVICES=""), dict(PCI, CUDA_VISIBLE_DEVICES="GPU-5060")):
            self.assertIsNone(G.visible_only(G.detect(0, environ=env, gpus=gs), env))      # the user's choice is kept
        env = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}                                       # not known which GPU --device is
        self.assertIsNone(G.visible_only(G.detect(0, environ=env, gpus=gs), env))
        no_uuid = G.parse("0, NVIDIA RTX A2000 12GB\n1, NVIDIA GeForce RTX 5060 Ti\n", ("index", "name"))
        self.assertIsNone(G.visible_only(G.detect(1, environ=PCI, gpus=no_uuid), PCI))      # an nvidia-smi without UUIDs: as 0.1.5
        self.assertIsNone(G.visible_only(G.Detected(), PCI))                               # no nvidia-smi

    def visible(self, device, env, gpus=None, started=False):
        from yasakani import cli
        det = G.detect(device, environ=env, gpus=self.gpus() if gpus is None else gpus)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            opened = cli._visible_gpu(det, device, environ=env, cuda_started=started)
        return opened, err.getvalue(), det

    def test_command_line(self):
        env = dict(PCI)
        opened, said, det = self.visible(1, env)
        self.assertEqual((opened, env.get("CUDA_VISIBLE_DEVICES")), (0, "GPU-50600000-0000"))
        self.assertEqual(said.strip(), "gpu: GPU 1 of nvidia-smi (NVIDIA GeForce RTX 5060 Ti); the engine sees only this GPU "
                                       "(CUDA_VISIBLE_DEVICES is set to its UUID)")
        self.assertNotIn("GPU-50600000", said)                                              # the UUID is not printed
        self.assertEqual(det.gpu.index, 1)                                                  # the plan still looks at GPU 1
        plan = G.plan_context(FakePack(FASTEST), det)
        self.assertEqual((plan.max_ctx, plan.free_mib), (98304, 15911))
        env = dict(PCI)
        opened, said, _ = self.visible(0, env)                                               # --device 0 (the default) too
        self.assertEqual((opened, env.get("CUDA_VISIBLE_DEVICES")), (0, "GPU-a2000000-0000"))
        self.assertIn("gpu: GPU 0 of nvidia-smi (NVIDIA RTX A2000 12GB)", said)
        for env, device in ((dict(PCI), 0), (dict(PCI), 1)):                                # one GPU: no change at all
            opened, said, _ = self.visible(device, env, gpus=self.gpus()[:1])
            self.assertEqual((opened, said, env), (device, "", PCI))
        env = dict(PCI, CUDA_VISIBLE_DEVICES="1")                                           # set by the user (yasakani_env.cmd): kept
        opened, said, _ = self.visible(0, env)
        self.assertEqual((opened, said, env["CUDA_VISIBLE_DEVICES"]), (0, "", "1"))
        env = dict(PCI, CUDA_VISIBLE_DEVICES="0,1")
        opened, said, _ = self.visible(1, env)
        self.assertEqual((opened, said, env["CUDA_VISIBLE_DEVICES"]), (1, "", "0,1"))
        env = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
        opened, said, _ = self.visible(1, env)
        self.assertEqual((opened, said, env), (1, "", {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}))
        env = dict(PCI)                                                                     # CUDA may have started: too late to change
        opened, said, _ = self.visible(1, env, started=True)
        self.assertEqual((opened, said, env), (1, "", PCI))
        no_uuid = G.parse("\n".join([REPORTED[0], smi_line(1, "NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 400, "[N/A]")]))
        env = dict(PCI)                                                                     # no UUID for GPU 1: nothing hidden, and it says so
        opened, said, _ = self.visible(1, env, gpus=no_uuid)
        self.assertEqual((opened, env), (1, PCI))
        self.assertEqual(said.strip(), "gpu: GPU 1 of nvidia-smi (NVIDIA GeForce RTX 5060 Ti); nvidia-smi gave no UUID for it, so the "
                                       "engine also sees the other GPUs")
        self.assertNotIn("sees only", said)
        env = dict(PCI)                                                                     # GPU 0 has its UUID: as above
        opened, said, _ = self.visible(0, env, gpus=no_uuid)
        self.assertEqual((opened, env.get("CUDA_VISIBLE_DEVICES")), (0, "GPU-a2000000-0000"))
        self.assertIn("the engine sees only this GPU", said)


class ContextRuleTests(unittest.TestCase):
    """The rule, with the numbers of the published files (yasakani/gpu.py: choose, plan_context)."""

    def test_choose(self):
        self.assertEqual(G.choose(16300 - 1000, *FASTEST, 262144), (65536, True))      # 16 GB that also drives the display
        self.assertEqual(G.choose(16311 - 0, *FASTEST, 262144), (65536, False))       # 16 GB without a display
        self.assertEqual(G.choose(16303 - 1500, *FASTEST, 262144), (49152, True))
        self.assertEqual(G.choose(16303 - 3000, *FASTEST, 262144)[0], 0)              # does not fit
        self.assertEqual(G.choose(24564 - 1000, *QUALITY, 262144), (262144, False))   # 24 GB: the quality model at full context
        self.assertEqual(G.choose(24564 - 4000, *QUALITY, 262144), (163840, False))
        self.assertEqual(G.choose(24564 - 1000, *FASTEST, 262144), (262144, False))
        self.assertEqual(G.choose(16376 - 800, *QUALITY, 262144)[0], 0)               # 16 GB: the quality model does not fit
        self.assertEqual(G.choose(12288 - 300, *FASTEST, 262144)[0], 0)               # 12 GB: nothing fits
        self.assertEqual(G.choose(20000, *FASTEST, 4096), (4096, False))              # a model whose own context is small

    def test_rule_is_safe_against_the_measured_overhead(self):
        """The chosen context must fit with the overhead measured on the RTX 5090 (2,090 / 2,291 MiB above the files)."""
        for sizes, measured in ((FASTEST, 2090), (QUALITY, 2291)):
            for total in (16303, 16376, 20470, 24564, 24576):
                for used in range(0, 6001, 250):
                    n, head_off = G.choose(total - used, *sizes, 262144)
                    if not n:
                        continue
                    files = sizes[0] + sizes[1] + (0 if head_off else sizes[2])
                    need = files / G.MIB + measured + n * G.KV_BYTES_PER_TOKEN / G.MIB
                    self.assertLessEqual(need + 500, total - used, (sizes, total, used, n, head_off))
                    self.assertEqual(n % G.CONTEXT_STEP if n < 262144 else 0, 0)

    def test_plan_unchanged_on_the_rtx_5090(self):
        for used in (0, 2644, 12000, 20000):                                   # however much else uses it
            plan = G.plan_context(FakePack(QUALITY), det_of(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607, used)))
            self.assertEqual((plan.max_ctx, plan.env, plan.notes), (None, {}, []))
        plan = G.plan_context(FakePack(QUALITY), G.Detected())                    # nvidia-smi gave nothing
        self.assertEqual((plan.max_ctx, plan.env, plan.notes), (None, {}, []))

    def test_plan_16gb(self):
        plan = G.plan_context(FakePack(FASTEST), det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376, 1000)))
        self.assertEqual(plan.max_ctx, 81920)
        self.assertEqual(plan.env, dict(G.SMALL_CARD_ENV, YSKN_BDR_SHEAD="0"))
        self.assertTrue(any("bdraft2_k8.shead" in n and "3%" in n for n in plan.notes))
        self.assertTrue(any(n.startswith("context: 81920 tokens") for n in plan.notes))
        plan = G.plan_context(FakePack(FASTEST), det_of(smi_line(0, "NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 0)))
        self.assertEqual((plan.max_ctx, plan.env), (65536, G.SMALL_CARD_ENV))

    def test_plan_24gb(self):
        plan = G.plan_context(FakePack(QUALITY), det_of(smi_line(0, "NVIDIA GeForce RTX 3090", "8.6", 24576, 1000)))
        self.assertEqual((plan.max_ctx, plan.env, plan.notes), (None, G.SMALL_CARD_ENV, []))
        plan = G.plan_context(FakePack(QUALITY), det_of(smi_line(0, "NVIDIA GeForce RTX 4090", "8.9", 24564, 4000)))
        self.assertEqual((plan.max_ctx, plan.env), (163840, G.SMALL_CARD_ENV))

    def test_plan_does_not_fit(self):
        with self.assertRaises(YasakaniError) as cm:
            G.plan_context(FakePack(QUALITY), det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376, 500)))
        self.assertIn("does not fit", str(cm.exception))
        self.assertIn("use the fastest model", str(cm.exception))
        with self.assertRaises(YasakaniError) as cm:
            G.plan_context(FakePack(FASTEST), det_of(smi_line(0, "NVIDIA GeForce RTX 3060", "8.6", 12288, 300)))
        self.assertIn("--max-ctx", str(cm.exception))
        self.assertNotIn("fastest model", str(cm.exception))

    def test_plan_unknown_memory(self):
        det = det_of("0, GPU-1, NVIDIA GeForce RTX 4080, 8.9, 16376, [N/A], 581.57")
        self.assertEqual(G.plan_context(FakePack(FASTEST), det).max_ctx, None)

    def test_given_context(self):
        """--max-ctx: the context stays as given; the draft model's output head only where it fits beside it (plan_given_context)."""
        ti = det_of(smi_line(0, "NVIDIA GeForce RTX 5070 Ti", "12.0", 16303, 340))     # no display on the card: 340 MiB in use
        plan = G.plan_given_context(FakePack(FASTEST), ti, 16384)                    # the head fits beside 16,384 tokens
        self.assertEqual((plan.max_ctx, plan.env, plan.notes), (16384, G.SMALL_CARD_ENV, []))
        plan = G.plan_given_context(FakePack(FASTEST), ti, 98304)                    # it does not beside 98,304 (0.1.4 loaded it anyway)
        self.assertEqual((plan.max_ctx, plan.env), (98304, dict(G.SMALL_CARD_ENV, YSKN_BDR_SHEAD="0")))
        self.assertEqual(len(plan.notes), 1)
        self.assertIn("bdraft2_k8.shead", plan.notes[0])
        self.assertIn("beside 98304 tokens", plan.notes[0])
        self.assertNotIn("context: ", plan.notes[0])                                 # only the rule's own choice is reported that way
        plan = G.plan_given_context(FakePack(FASTEST), ti, 1048576)                  # far too long: still tried as given
        self.assertEqual((plan.max_ctx, plan.env), (1048576, dict(G.SMALL_CARD_ENV, YSKN_BDR_SHEAD="0")))
        rtx4090 = det_of(smi_line(0, "NVIDIA GeForce RTX 4090", "8.9", 24564, 4000))
        self.assertEqual(G.plan_given_context(FakePack(QUALITY), rtx4090, 163840).env, G.SMALL_CARD_ENV)     # what plan_context chooses there
        self.assertEqual(G.plan_given_context(FakePack(QUALITY), rtx4090, 262144).env, dict(G.SMALL_CARD_ENV, YSKN_BDR_SHEAD="0"))
        for det in (det_of(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607, 20000)), G.Detected()):   # 30,000 MiB or more, or unknown
            plan = G.plan_given_context(FakePack(QUALITY), det, 262144)
            self.assertEqual((plan.max_ctx, plan.env, plan.notes), (262144, {}, []))
        plan = G.plan_given_context(FakePack(QUALITY), det_of("0, GPU-1, NVIDIA GeForce RTX 4080, 8.9, 16376, [N/A], 581.57"), 262144)
        self.assertEqual((plan.max_ctx, plan.env, plan.notes), (262144, G.SMALL_CARD_ENV, []))      # free unknown: only the sizes of 0.1.7

    def test_small_card_keeps_the_prompt_sizes_of_017(self):
        """Under FULL_CONTEXT_MIN_TOTAL_MIB the prompt-reading sizes stay those of 0.1.7 (OVERHEAD_MIB was measured with them); the
        RTX 5090 keeps the larger defaults of runtime.py.  The 16 GB case with the published sizes: the context is chosen as before."""
        self.assertEqual(G.SMALL_CARD_ENV, {"YSKN_Q35_PMAX": "1024", "YSKN_Q35_LAYER_ROWS": "4096"})
        for line in (smi_line(0, "NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 900), smi_line(0, "NVIDIA GeForce RTX 3090", "8.6", 24576, 1000)):
            det = det_of(line)
            for plan in (G.plan_context(FakePack(FASTEST), det), G.plan_given_context(FakePack(FASTEST), det, 32768),
                         G.plan_parallel(FakePack(FASTEST), det, 2), G.plan_given_parallel(FakePack(FASTEST), det, 16384, 2)):
                self.assertEqual({k: plan.env[k] for k in G.SMALL_CARD_ENV}, G.SMALL_CARD_ENV, line)
        plan = G.plan_context(FakePack(FASTEST), det_of(smi_line(0, "NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 900)))
        self.assertEqual(plan.max_ctx, G.choose(16311 - 900, *FASTEST, 262144)[0])
        r5090 = det_of(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607, 1107))
        for plan in (G.plan_context(FakePack(FASTEST), r5090), G.plan_parallel(FakePack(FASTEST), r5090, 2)):
            self.assertFalse(set(G.SMALL_CARD_ENV) & set(plan.env))
        from yasakani.runtime import RuntimeProfile
        with tempfile.TemporaryDirectory() as t:
            prof = RuntimeProfile.default(support.make_pack(Path(t)), G.SMALL_CARD_ENV)
        self.assertEqual((prof.env["YSKN_Q35_PMAX"], prof.env["YSKN_Q35_LAYER_ROWS"]), ("1024", "4096"))


class PrefillModeTests(unittest.TestCase):
    """--prefill / YASAKANI_PREFILL: auto (default; the new reading only on a card known to have 30,000 MiB or more, else old), new
    (the new reading on any card, with the sizes of 0.1.7 under 30,000 MiB or when the memory is not known) or old (YSKN_Q35_PF_V2=0
    and the sizes of 0.1.7)."""

    CARDS = {
        "32 GB": smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607, 1107),
        "24 GB": smi_line(0, "NVIDIA GeForce RTX 4090", "8.9", 24564, 4000),
        "16 GB": smi_line(0, "NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 900),
        "unknown": "0, GPU-1, NVIDIA GeForce RTX 4080, 8.9, [N/A], [N/A], 581.57",
    }
    # what each mode adds to the plan on each card, and the reading it chooses
    WANT = {
        ("auto", "32 GB"): ({}, "new"), ("auto", "24 GB"): (G.OLD_PREFILL_ENV, "old"),
        ("auto", "16 GB"): (G.OLD_PREFILL_ENV, "old"), ("auto", "unknown"): (G.OLD_PREFILL_ENV, "old"),
        ("new", "32 GB"): ({}, "new"), ("new", "24 GB"): (G.SMALL_CARD_ENV, "new"),
        ("new", "16 GB"): (G.SMALL_CARD_ENV, "new"), ("new", "unknown"): (G.SMALL_CARD_ENV, "new"),
        ("old", "32 GB"): (G.OLD_PREFILL_ENV, "old"), ("old", "24 GB"): (G.OLD_PREFILL_ENV, "old"),
        ("old", "16 GB"): (G.OLD_PREFILL_ENV, "old"), ("old", "unknown"): (G.OLD_PREFILL_ENV, "old"),
    }

    def plans(self, det):
        """The four plans _open_stack can start from, before use_prefill (a new one at each call)."""
        return [lambda: G.plan_context(FakePack(FASTEST), det), lambda: G.plan_given_context(FakePack(FASTEST), det, 32768),
                lambda: G.plan_parallel(FakePack(FASTEST), det, 2), lambda: G.plan_given_parallel(FakePack(FASTEST), det, 16384, 2)]

    def test_mode_of(self):
        self.assertEqual(G.PREFILL_MODES, ("auto", "new", "old"))
        self.assertEqual(G.prefill_mode_of(None, {}), "auto")
        self.assertEqual(G.prefill_mode_of("", {}), "auto")
        self.assertEqual(G.prefill_mode_of(None, {"YASAKANI_PREFILL": ""}), "auto")
        self.assertEqual(G.prefill_mode_of(None, {"YASAKANI_PREFILL": " OLD "}), "old")
        self.assertEqual(G.prefill_mode_of(None, {"YASAKANI_PREFILL": "new"}), "new")
        self.assertEqual(G.prefill_mode_of("auto", {"YASAKANI_PREFILL": "old"}), "auto")          # the option wins
        self.assertEqual(G.prefill_mode_of("new", {"YASAKANI_PREFILL": "old"}), "new")
        for bad in ("fast", "0", "on"):
            with self.assertRaises(YasakaniError) as cm:
                G.prefill_mode_of(bad, {})
            self.assertIn("--prefill must be one of auto, new, old", str(cm.exception))
        with self.assertRaises(YasakaniError):
            G.prefill_mode_of(None, {"YASAKANI_PREFILL": "off"})

    def test_every_mode_on_every_card(self):
        self.assertEqual(G.OLD_PREFILL_ENV, {"YSKN_Q35_PF_V2": "0", "YSKN_Q35_PMAX": "1024", "YSKN_Q35_LAYER_ROWS": "4096"})
        self.assertEqual(G.FULL_CONTEXT_MIN_TOTAL_MIB, 30000)
        planned = 0
        for (mode, card), (env, reading) in self.WANT.items():
            det = det_of(self.CARDS[card])
            self.assertEqual(G.prefill_choice(mode, det)[:2], (reading, env), (mode, card))
            self.assertEqual(G.prefill_choice(mode, det.gpu)[:2], (reading, env), (mode, card))       # a GpuInfo gives the same
            for make in self.plans(det):
                try:
                    before = make()
                except YasakaniError:                    # --parallel 2 does not fit beside the fastest model on every card
                    continue
                planned += 1
                plan = G.use_prefill(make(), mode, det)
                if before.given and before.room_mib is not None and (mode, card) == ("auto", "24 GB"):   # 0.3.4: room beside
                    env_w, reading_w = G.SMALL_CARD_ENV, "new"                                           # a given context
                    self.assertGreaterEqual(before.room_mib, G.NEW_PREFILL_RESERVE_MIB)
                else:
                    env_w, reading_w = env, reading
                self.assertEqual(plan.env, dict(before.env, **env_w), (mode, card))
                self.assertEqual((plan.max_ctx, plan.free_mib, plan.notes), (before.max_ctx, before.free_mib, before.notes))
                self.assertEqual(plan.prefill, reading_w, (mode, card))
                self.assertEqual(plan.env.get("YSKN_Q35_PF_V2"), "0" if reading_w == "old" else None, (mode, card))
                if card != "32 GB":                      # off the tested card every reading keeps the sizes of 0.1.7
                    self.assertEqual({k: plan.env[k] for k in G.SMALL_CARD_ENV}, G.SMALL_CARD_ENV, (mode, card))
        self.assertGreaterEqual(planned, 36)
        for mode in G.PREFILL_MODES:                     # no GPU found at all: as a card of unknown memory
            want = self.WANT[(mode, "unknown")]
            self.assertEqual(G.prefill_choice(mode, G.Detected())[:2], (want[1], want[0]))
            self.assertEqual(G.prefill_choice(mode, None)[:2], (want[1], want[0]))

    def test_the_limit(self):
        """30,000 MiB itself counts as large, one MiB less does not; several GPUs that cannot be told apart count as the smallest."""
        self.assertEqual(G.prefill_choice("auto", det_of(smi_line(0, "NVIDIA RTX A", "8.6", 30000)))[0], "new")
        self.assertEqual(G.prefill_choice("auto", det_of(smi_line(0, "NVIDIA RTX A", "8.6", 29999)))[0], "old")
        rows = (smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607), smi_line(1, "NVIDIA GeForce RTX 4090", "8.9", 24564))
        two = det_of(*rows)
        self.assertEqual(len(two.candidates), 2)         # CUDA's order is not PCI's here: either GPU may be device 0
        self.assertEqual(G.prefill_choice("auto", two)[0], "old")
        self.assertEqual(G.prefill_choice("auto", det_of(*rows, environ={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}))[0], "new")
        # the line says that the smallest of the GPUs decided, not that the engine runs on a 24 GB card
        self.assertEqual(G.prefill_choice("auto", two)[2], "prompt reading: as in 0.1.7 (setting auto: smallest of 2 GPUs: 24564 MiB "
                                                           "and no --max-ctx; --prefill new or YASAKANI_PREFILL=new to try it)")
        self.assertEqual(G.prefill_choice("new", two)[2],
                         "prompt reading: new (setting new; smallest of 2 GPUs: 24564 MiB, so prompt sizes of 0.1.7)")
        self.assertEqual(G.prefill_choice("auto", det_of(*rows, environ={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}))[2],
                         "prompt reading: new (setting auto: 32607 MiB card)")
        # free memory not readable, totals known: the totals still decide, and the line gives them
        bigs = det_of("0, GPU-0, NVIDIA GeForce RTX 5090, 12.0, 32607, [N/A], 616.92", "1, GPU-1, NVIDIA RTX A6000, 8.6, 49140, [N/A], 616.92")
        self.assertEqual(len(bigs.candidates), 2)
        self.assertEqual(G.prefill_choice("auto", bigs)[0::2], ("new", "prompt reading: new (setting auto: smallest of 2 GPUs: 32607 MiB)"))
        mixed = det_of("0, GPU-0, NVIDIA GeForce RTX 5090, 12.0, 32607, [N/A], 616.92", "1, GPU-1, NVIDIA GeForce RTX 4080, 8.9, [N/A], [N/A], 616.92")
        self.assertIn("card memory not known", G.prefill_choice("auto", mixed)[2])

    def test_auto_on_32_gb_is_as_before(self):
        """auto on the RTX 5090 adds nothing: the plan is the one the plan functions give alone, as before --prefill new."""
        det = det_of(self.CARDS["32 GB"])
        for make in self.plans(det):
            before, plan = make(), G.use_prefill(make(), "auto", det)
            self.assertEqual((plan.env, plan.max_ctx, plan.free_mib, plan.notes, plan.parallel),
                             (before.env, before.max_ctx, before.free_mib, before.notes, before.parallel))
        self.assertEqual(G.use_prefill(G.plan_context(FakePack(FASTEST), det), "auto", det).env, {})
        from yasakani.runtime import DEFAULT_ENV, RuntimeProfile
        with tempfile.TemporaryDirectory() as t:
            prof = RuntimeProfile.default(support.make_pack(Path(t)), G.use_prefill(G.ContextPlan(), "auto", det).env)
        for k in G.SMALL_CARD_ENV:
            self.assertEqual(prof.env[k], DEFAULT_ENV[k])
        self.assertNotIn("YSKN_Q35_PF_V2", prof.env)

    def test_line(self):
        """One short line per start: which reading and why."""
        lines = {k: G.prefill_choice(k[0], det_of(self.CARDS[k[1]]))[2] for k in self.WANT}
        self.assertEqual(lines[("auto", "32 GB")], "prompt reading: new (setting auto: 32607 MiB card)")
        self.assertEqual(lines[("auto", "24 GB")],
                         "prompt reading: as in 0.1.7 (setting auto: 24564 MiB card and no --max-ctx; "
                         "--prefill new or YASAKANI_PREFILL=new to try it)")
        self.assertEqual(lines[("auto", "unknown")], "prompt reading: as in 0.1.7 (setting auto: card memory not known; "
                         "--prefill new or YASAKANI_PREFILL=new to try it)")
        self.assertEqual(lines[("new", "32 GB")], "prompt reading: new (setting new)")
        self.assertEqual(lines[("new", "16 GB")], "prompt reading: new (setting new; 16311 MiB card, so prompt sizes of 0.1.7)")
        self.assertEqual(lines[("new", "unknown")], "prompt reading: new (setting new; card memory not known, so prompt sizes of 0.1.7)")
        self.assertEqual(lines[("old", "24 GB")], "prompt reading: as in 0.1.7 (setting old)")
        for k, line in lines.items():
            self.assertTrue(line.startswith("prompt reading: " + ("new" if self.WANT[k][1] == "new" else "as in 0.1.7")), k)
            self.assertNotIn("\n", line)
            self.assertLess(len(line), 140, k)
        plan = G.use_prefill(G.ContextPlan(), "auto", det_of(self.CARDS["16 GB"]))
        self.assertEqual((plan.prefill, plan.prefill_line), ("old", lines[("auto", "16 GB")]))

    def test_plan(self):
        r3090 = det_of(smi_line(0, "NVIDIA GeForce RTX 3090", "8.6", 24576, 1000))
        auto = G.plan_context(FakePack(FASTEST), r3090)
        old = G.use_prefill(G.plan_context(FakePack(FASTEST), r3090), "old", r3090)
        self.assertEqual(old.env, dict(auto.env, **G.OLD_PREFILL_ENV))
        self.assertEqual((old.max_ctx, old.free_mib), (auto.max_ctx, auto.free_mib))      # the context is chosen as for auto
        from yasakani.runtime import RuntimeProfile
        with tempfile.TemporaryDirectory() as t:
            prof = RuntimeProfile.default(support.make_pack(Path(t)), G.use_prefill(G.ContextPlan(), "old").env)
        self.assertEqual([prof.env[k] for k in ("YSKN_Q35_PF_V2", "YSKN_Q35_PMAX", "YSKN_Q35_LAYER_ROWS")], ["0", "1024", "4096"])


class PrefillReserveTests(unittest.TestCase):
    """--prefill new on a card under 30,000 MiB: the planners leave NEW_PREFILL_RESERVE_MIB free.  The new reading keeps a work area
    (NEW_PREFILL_AREA_MIB at the sizes of SMALL_CARD_ENV; measured +569 / +607 MiB at the peak on an RTX 5090 shown as a 24 GB / 16 GB
    card), the engine makes it only with NEW_PREFILL_SPARE_MIB to spare, and CUDA reported NEW_PREFILL_CUDA_GAP_MIB less free memory
    than nvidia-smi.  Not at the cost of more than half of the context.  Every other choice plans as before."""

    R = 1800
    CHOSEN = ("note: --prefill new: the context is chosen with 1800 MiB more left free (the new prompt reading keeps a work area of about "
              "579 MiB from the first long prompt on; the engine makes it only with 1024 MiB to spare, and reads prompts without it "
              "when it finds less)")
    NO_ROOM = ("note: --prefill new: keeping 1800 MiB free for the work area of the new prompt reading would cost more than half of the "
               "context, so the context is chosen without it; the engine reads prompts without that work area when it finds no room")

    def test_numbers(self):
        self.assertEqual((G.NEW_PREFILL_RESERVE_MIB, G.NEW_PREFILL_AREA_MIB, G.NEW_PREFILL_SPARE_MIB, G.NEW_PREFILL_CUDA_GAP_MIB),
                         (self.R, 579, 1024, 566))
        # after the load the planner's margin and the reserve are free in nvidia-smi: enough for the area (607,322,112 B and the
        # driver's rounding of its 17 allocations, about 34 MiB) with the engine's spare, as CUDA counts the free memory
        self.assertGreaterEqual(G.MARGIN_MIB + G.NEW_PREFILL_RESERVE_MIB,
                                607322112 / G.MIB + 34 + G.NEW_PREFILL_SPARE_MIB + G.NEW_PREFILL_CUDA_GAP_MIB + 64)

    def test_reserve(self):
        for name, cc, total, want in (("NVIDIA GeForce RTX 5090", "12.0", 32607, 0), ("NVIDIA RTX A6000", "8.6", 49140, 0),
                                      ("NVIDIA GeForce RTX 4090", "8.9", 24564, self.R), ("NVIDIA GeForce RTX 3090", "8.6", 24576, self.R),
                                      ("NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, self.R)):
            det = det_of(smi_line(0, name, cc, total, 500))
            self.assertEqual([G.prefill_reserve_mib(m, det) for m in ("auto", "old", "new")], [0, 0, want], name)
        self.assertEqual([G.prefill_reserve_mib(m, None) for m in ("auto", "old", "new")], [0, 0, self.R])
        self.assertEqual([G.prefill_reserve_mib(m, G.Detected()) for m in ("auto", "old", "new")], [0, 0, self.R])
        plan = G.plan_context(FakePack(FASTEST), G.Detected(), self.R)          # unknown memory: nothing is planned, no note
        self.assertEqual((plan.max_ctx, plan.notes, plan.reserve_mib), (None, [], 0))

    def test_24_gb(self):
        for sizes, used, ctx_old, ctx_new in ((QUALITY, 500, None, 229376), (QUALITY, 1000, None, 212992), (FASTEST, 500, None, None),
                                              (FASTEST, 4000, None, 180224)):
            det = det_of(smi_line(0, "NVIDIA GeForce RTX 4090", "8.9", 24564, used))
            old, new = G.plan_context(FakePack(sizes), det), G.plan_context(FakePack(sizes), det, self.R)
            self.assertEqual((old.max_ctx, new.max_ctx, old.reserve_mib, new.reserve_mib), (ctx_old, ctx_new, 0, self.R), used)
            self.assertEqual((new.env, new.notes[-1], len(new.notes)), (G.SMALL_CARD_ENV, self.CHOSEN, 1 if ctx_new is None else 2), used)
            # what nvidia-smi shows free after the load: room for the area with the engine's spare, as CUDA counts it
            ctx = ctx_new or 262144
            after_load = sum(sizes) / G.MIB + G.overhead_mib(sizes[0]) + ctx * G.KV_BYTES_PER_TOKEN / G.MIB
            self.assertGreaterEqual(24564 - used - after_load,
                                    G.NEW_PREFILL_AREA_MIB + 34 + G.NEW_PREFILL_SPARE_MIB + G.NEW_PREFILL_CUDA_GAP_MIB, used)

    def test_not_for_more_than_half_of_the_context(self):
        for name, total, used, sizes, ctx in (("NVIDIA GeForce RTX 4080", 16384, 500, FASTEST, 98304),
                                              ("NVIDIA GeForce RTX 5060 Ti", 16311, 900, FASTEST, 81920),
                                              ("NVIDIA GeForce RTX 4080", 16384, 1800, FASTEST, 32768),
                                              ("NVIDIA GeForce RTX 4090", 24564, 4000, QUALITY, 163840)):
            det = det_of(smi_line(0, name, "8.9", total, used))
            old, new = G.plan_context(FakePack(sizes), det), G.plan_context(FakePack(sizes), det, self.R)
            self.assertEqual((old.max_ctx, new.max_ctx, new.reserve_mib, new.env), (ctx, ctx, 0, old.env), (name, used))
            self.assertEqual(new.notes, [self.NO_ROOM] + old.notes, (name, used))
        det = det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16384, 3000))     # does not fit at all: the same error as without it
        errors = []
        for r in (0, self.R):
            with self.assertRaises(YasakaniError) as cm:
                G.plan_context(FakePack(FASTEST), det, r)
            errors.append(str(cm.exception))
        self.assertEqual(errors[0], errors[1])
        self.assertIn("13384 of 16384 MiB free", errors[0])

    def test_given_context_stays(self):
        det = det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16384, 500))
        p = G.plan_given_context(FakePack(FASTEST), det, 98304, self.R)
        self.assertEqual((p.max_ctx, p.reserve_mib, p.env.get("YSKN_BDR_SHEAD")), (98304, self.R, "0"))
        self.assertEqual(p.notes[0], "note: --prefill new: the new prompt reading needs about 2169 MiB free in nvidia-smi after the model "
                                     "is loaded (a work area of about 579 MiB, made only with 1024 MiB to spare); with less, prompts are "
                                     "read without that work area.  The context you gave is used as it is")
        q = G.plan_given_context(FakePack(FASTEST), det, 98304)
        self.assertEqual((q.max_ctx, q.reserve_mib), (98304, 0))
        self.assertFalse(any("--prefill new" in n for n in q.notes))

    def test_parallel(self):
        det = det_of(smi_line(0, "NVIDIA GeForce RTX 4090", "8.9", 24564, 500))
        for n, ctx_old, ctx_new, head_off in ((2, 131072, 98304, False), (4, 65536, 65536, True)):
            a, b = G.plan_parallel(FakePack(FASTEST), det, n), G.plan_parallel(FakePack(FASTEST), det, n, self.R)
            self.assertEqual((a.max_ctx, b.max_ctx, a.reserve_mib, b.reserve_mib), (ctx_old, ctx_new, 0, self.R), n)
            self.assertEqual(("YSKN_BDR_SHEAD" in a.env, "YSKN_BDR_SHEAD" in b.env), (False, head_off), n)
            self.assertEqual(b.notes[-1], self.CHOSEN, n)
        g = G.plan_given_parallel(FakePack(FASTEST), det, 32768, 2, self.R)
        self.assertEqual((g.max_ctx, g.reserve_mib), (32768, self.R))
        self.assertTrue(g.notes[0].endswith("The context you gave is used as it is"))
        det = det_of(smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16384, 500))      # 2 requests fit, but not with the reserve
        a, b = G.plan_parallel(FakePack(FASTEST), det, 2), G.plan_parallel(FakePack(FASTEST), det, 2, self.R)
        self.assertEqual((a.max_ctx, b.max_ctx, b.reserve_mib), (16384, 16384, 0))
        self.assertEqual(b.notes[0], self.NO_ROOM)


class EditCopyTests(unittest.TestCase):
    """--edit-copy / YASAKANI_EDIT_COPY: on | off, passed through the plan as YSKN_EDIT_COPY=1 / 0; EDIT_COPY_DEFAULT when neither is
    given.  The same on every card and with every prompt reading."""

    def test_default(self):
        self.assertEqual(G.EDIT_COPY_DEFAULT, "on")
        self.assertIn(G.EDIT_COPY_DEFAULT, G.EDIT_COPY_MODES)
        self.assertEqual(G.EDIT_COPY_MODES, ("on", "off"))

    def test_mode_of_with_either_default(self):
        """Changing EDIT_COPY_DEFAULT alone changes the default, and nothing else."""
        for default in ("off", "on"):
            with mock.patch.object(G, "EDIT_COPY_DEFAULT", default):
                self.assertEqual(G.edit_copy_of(None, {}), default)
                self.assertEqual(G.edit_copy_of("", {}), default)
                self.assertEqual(G.edit_copy_of(None, {"YASAKANI_EDIT_COPY": ""}), default)
                self.assertEqual(G.edit_copy_of(None, {"YASAKANI_EDIT_COPY": " ON "}), "on")
                self.assertEqual(G.edit_copy_of(None, {"YASAKANI_EDIT_COPY": "off"}), "off")
                self.assertEqual(G.edit_copy_of("off", {"YASAKANI_EDIT_COPY": "on"}), "off")      # the option wins
                self.assertEqual(G.edit_copy_of("on", {"YASAKANI_EDIT_COPY": "off"}), "on")
                plan = G.use_edit_copy(G.ContextPlan(), G.edit_copy_of(None, {}))
                self.assertEqual((plan.env, plan.edit_copy), ({"YSKN_EDIT_COPY": "1" if default == "on" else "0"}, default))
                from yasakani import cli
                with mock.patch.dict(os.environ, {"YASAKANI_EDIT_COPY": ""}):
                    sub = next(a for a in cli.build_parser()._actions if type(a).__name__ == "_SubParsersAction")
                    act = next(a for a in sub.choices["chat"]._actions if a.dest == "edit_copy")
                    self.assertTrue(act.help.endswith("$YASAKANI_EDIT_COPY or %s" % default), act.help)

    def test_wrong_value(self):
        for bad in ("yes", "1", "auto"):
            with self.assertRaises(YasakaniError) as cm:
                G.edit_copy_of(bad, {})
            self.assertIn("--edit-copy must be one of on, off", str(cm.exception))
            with self.assertRaises(YasakaniError):
                G.edit_copy_of(None, {"YASAKANI_EDIT_COPY": bad})
            with self.assertRaises(YasakaniError):
                G.use_edit_copy(G.ContextPlan(), bad)

    def test_plan_on_every_card(self):
        """Only YSKN_EDIT_COPY is added; the context and the prompt reading stay as they were."""
        for line in PrefillModeTests.CARDS.values():
            det = det_of(line)
            for prefill in G.PREFILL_MODES:
                before = G.use_prefill(G.plan_given_context(FakePack(FASTEST), det, 32768), prefill, det)
                for mode, v in (("on", "1"), ("off", "0")):
                    plan = G.use_edit_copy(G.use_prefill(G.plan_given_context(FakePack(FASTEST), det, 32768), prefill, det), mode)
                    self.assertEqual(plan.env, dict(before.env, YSKN_EDIT_COPY=v))
                    self.assertEqual((plan.max_ctx, plan.notes, plan.prefill, plan.prefill_line),
                                     (before.max_ctx, before.notes, before.prefill, before.prefill_line))
        from yasakani.runtime import RuntimeProfile
        with tempfile.TemporaryDirectory() as t:
            prof = RuntimeProfile.default(support.make_pack(Path(t)), G.use_edit_copy(G.ContextPlan(), "on").env)
        self.assertEqual(prof.env["YSKN_EDIT_COPY"], "1")


class QuickstartTests(unittest.TestCase):
    """The quickstart's own copy of the decision and the rule (it runs before the package is installed)."""

    @classmethod
    def setUpClass(cls):
        cls.qs = load_quickstart()
        if cls.qs is None:
            raise unittest.SkipTest("quickstart/quickstart.py not found (set YASAKANI_QUICKSTART)")

    def test_version(self):
        self.assertEqual(self.qs.QS_VERSION, "0.4.0")
        self.assertIn("/v0.4.0/", self.qs.ENGINE_URL)

    def test_decision_table(self):
        for name, cc, total, qs, quality in CARDS:
            g = self.qs.parse_gpus(smi_line(0, name, cc, total))[0]
            with self.subTest(card=name, total=total):
                self.assertEqual(self.qs.gpu_verdict(g), qs)
                if qs in ("tested", "other"):
                    self.assertEqual(total >= self.qs.QUALITY_MIN_TOTAL_MIB, quality)
        g = self.qs.parse_gpus("0, GPU-1, NVIDIA GeForce RTX 4090, [N/A], 24564, 900, 470.10")[0]    # an old driver: cc unknown
        self.assertEqual(self.qs.gpu_verdict(g), "other")

    def test_same_as_package(self):
        for k in ("FULL_CONTEXT_MIN_TOTAL_MIB", "QUALITY_MIN_TOTAL_MIB", "MIN_TOTAL_MIB", "KV_BYTES_PER_TOKEN", "CONTEXT_STEP", "MIN_CONTEXT",
                  "HEAD_OFF_BELOW", "MARGIN_MIB"):
            self.assertEqual(getattr(G, k), getattr(self.qs, k), k)
        for name, cc, total, qs, quality in CARDS:
            g = G.parse(smi_line(0, name, cc, total))[0]
            self.assertEqual(g.tested, self.qs.gpu_verdict(self.qs.parse_gpus(smi_line(0, name, cc, total))[0]) == "tested", name)
        for model, sizes in (("speed", FASTEST), ("quality", QUALITY), ("max", MAXQ)):
            self.assertEqual(self.qs.MODEL_BYTES[model], sizes)
            self.assertEqual(self.qs.OVERHEAD_MIB[model], G.overhead_mib(sizes[0]))
            for free in range(9000, 32001, 37):
                self.assertEqual(self.qs.estimate_context(free, model), G.choose(free, *sizes, 262144), (model, free))

    def test_max_model(self):
        """The max-quality model: the full context on the RTX 5090, a shorter one on a 24 GB card, none on a 16 GB card."""
        for free, want in ((32107, (262144, False)), (24064, (163840, False)), (24564, (196608, False)), (15884, (0, False)),
                           (16384, (0, False))):
            self.assertEqual(G.choose(free, *MAXQ, 262144), want, free)
            self.assertEqual(self.qs.estimate_context(free, "max"), want, free)
        self.assertEqual(G.overhead_mib(MAXQ[0]), 2300)
        self.assertEqual(self.qs.MODELS["max"]["repo"], "yasakani-dev/Qwen3.8-27B-YASAKANI-max-quality")
        self.assertEqual(set(self.qs.MODELS) - {"inkling2"}, set(self.qs.MODEL_BYTES))       # the Qwen3.8 rule's models
        self.assertEqual(set(self.qs.MODELS) - {"inkling2"}, set(self.qs.OVERHEAD_MIB))
        for m in self.qs.MODELS:
            self.assertIn("m_" + m, self.qs.T)

    def test_inkling2_rule_same_as_package(self):
        """0.3.2: the launcher's copy of the Inkling-2 rule (the published folder, drafter on, 512 rollback positions) gives the
        package's context for every free memory."""
        qs = self.qs
        self.assertEqual((qs.INK2_DEFAULT_CONTEXT, qs.INK2_CONTEXT_STEP, qs.INK2_MIN_CONTEXT, qs.INK2_OVERHEAD_MIB),
                         (32768, G.INK2_CONTEXT_STEP, G.INK2_MIN_CONTEXT, G.INK2_OVERHEAD_MIB))
        self.assertEqual(qs.INK2_BYTES_PER_POSITION, G.ink2_bytes_per_position(65536, True))
        self.assertAlmostEqual(qs.INK2_ROLLBACK_MIB, G.ink2_rollback_mib(512))
        self.assertEqual(sum(qs.INK2_FILES_BYTES), 15052423172 + 209688040)       # yasakani.json of the published folder
        for free in range(0, 40001, 13):
            want = G.ink2_context(free, sum(qs.INK2_FILES_BYTES), G.ink2_bytes_per_position(65536, True), 32768, 512)
            self.assertEqual(qs.estimate_ink2_context(free), want, free)
        self.assertEqual(qs.estimate_ink2_context(32607 - 3693), 32768)            # the RTX 5090 of the measurement
        self.assertEqual(qs.estimate_ink2_context(15000), 0)                       # a 16 GB card

    def test_pick(self):
        qs = self.qs
        two = qs.parse_gpus(smi_line(0, "NVIDIA GeForce RTX 3060", "8.6", 12288, 300, "GPU-aaaa") + "\n"
                            + smi_line(1, "NVIDIA GeForce RTX 5090", "12.0", 32607, 2000, "GPU-bbbb"))
        g, v, pin = qs.pick_gpu(two, environ={})
        self.assertEqual((g["index"], v, pin), (1, "tested", "GPU-bbbb"))
        g, v, pin = qs.pick_gpu(two[:1], environ={})
        self.assertEqual((g["index"], v, pin), (0, "small", None))
        g, v, pin = qs.pick_gpu(two, environ={"CUDA_VISIBLE_DEVICES": "GPU-aaaa"})       # the user chose one
        self.assertEqual((g["index"], v, pin), (0, "small", None))
        g, v, pin = qs.pick_gpu(two, environ={"CUDA_VISIBLE_DEVICES": "0"})              # 0.1.5: numbers are nvidia-smi's (PCI order)
        self.assertEqual((g["index"], v, pin), (0, "small", None))
        g, v, pin = qs.pick_gpu(two, environ={"CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual((g["index"], v, pin), (1, "tested", None))
        g, v, pin = qs.pick_gpu(two, environ={"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"})   # not known which:
        self.assertEqual((g["index"], v, pin), (0, "small", None))                                                # the less able one
        three = qs.parse_gpus("\n".join([smi_line(0, "NVIDIA GeForce RTX 4080", "8.9", 16376, 100, "GPU-c"),
                                         smi_line(1, "NVIDIA GeForce RTX 3090", "8.6", 24576, 900, "GPU-d"),
                                         smi_line(2, "NVIDIA GeForce RTX 2080 Ti", "7.5", 11264, 0, "GPU-e")]))
        g, v, pin = qs.pick_gpu(three, environ={})
        self.assertEqual((g["index"], v, pin), (1, "other", "GPU-d"))

    def test_query_fallback(self):
        def run(cmd, **kw):
            if "compute_cap" in cmd[1]:
                return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="0, GPU-1, NVIDIA GeForce RTX 3090, 24576, 500, 470.10\n", stderr="")
        gs = self.qs.query_gpus(run=run)
        self.assertEqual((gs[0]["name"], gs[0]["cc"], gs[0]["total"]), ("NVIDIA GeForce RTX 3090", None, 24576))

    def test_messages(self):
        T = self.qs.T
        self.assertIn("最適化も動作の確認もしていません（確かめたのは RTX 5090 だけです）", T["gpu_other"]["ja"])
        for key in ("no_smi", "no_gpu"):
            self.assertIn("AMD", T[key]["ja"])
            self.assertIn("Mac", T[key]["en"])
        self.assertNotIn("continue_q", T)
        for key, kw in (("gpu_is", dict(g="X", c="8.9", t=16376, d="1")), ("gpu_small", dict(t=8188)), ("gpu_old", dict(c="7.5")),
                        ("gpu_nocode", dict(c="9.0")), ("ctx_auto", dict(n="65,536")), ("ctx_short", dict(u=9000)),
                        ("gpu_pick", dict(n=2, g="X"))):
            for lang in ("ja", "en"):
                T[key][lang].format(**kw)


FAKE_SMI = r'''#!{python}
import json, os, sys
gpus = json.loads(os.environ["FAKE_GPUS"])
q = [a for a in sys.argv[1:] if a.startswith("--query-gpu=")]
if not q:
    sys.exit(0)
fields = q[0].split("=", 1)[1].split(",")
units = "nounits" not in " ".join(sys.argv)
known = set(gpus[0]) if gpus else set()
bad = [f for f in fields if f not in known]
if bad:
    print('Field "%s" is not a valid field to query.' % bad[0])
    sys.exit(2)
for g in gpus:
    print(", ".join(str(g[f]) + (" MiB" if units and f.startswith("memory.") else "") for f in fields))
'''


def load_lookahead_check():
    """quickstart/lookahead_check.py next to the quickstart under test; None when it is not there (the engine zip)."""
    qs = os.environ.get("YASAKANI_QUICKSTART")
    for p in ((Path(qs).parent / "lookahead_check.py") if qs else None, ROOT.parent / "quickstart" / "lookahead_check.py"):
        if p and p.is_file():
            spec = importlib.util.spec_from_file_location("yasakani_lookahead_check_under_test", str(p))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    return None


class CompareScriptTests(unittest.TestCase):
    """The quickstart's "compare" (lookahead_check.py) opens the engine with the settings `yasakani chat` uses: the context plan, the
    prompt reading and the edit copy; with an older package, which has no prompt reading or edit copy settings, the plan alone."""

    CARDS = PrefillModeTests.CARDS
    OLD_ENGINE = {"YSKN_Q35_PF_V2": "0", "YSKN_Q35_PMAX": "1024", "YSKN_Q35_LAYER_ROWS": "4096"}

    @classmethod
    def setUpClass(cls):
        cls.lc = load_lookahead_check()
        if cls.lc is None:
            raise unittest.SkipTest("quickstart/lookahead_check.py not found (set YASAKANI_QUICKSTART)")

    def plan(self, card, gpus=G, environ=None):
        said = []
        with mock.patch.object(G, "query", return_value=G.parse(self.CARDS[card] + "\n")), \
                mock.patch.dict(os.environ, environ or {"YASAKANI_PREFILL": "", "YASAKANI_EDIT_COPY": ""}):
            max_ctx, env = self.lc.engine_plan(FakePack(FASTEST), gpus, said.append)
        return max_ctx, env, said

    def test_settings_on_each_card(self):
        from yasakani.runtime import RuntimeProfile
        for card, prefill in (("32 GB", {}), ("24 GB", self.OLD_ENGINE), ("16 GB", self.OLD_ENGINE), ("unknown", self.OLD_ENGINE)):
            max_ctx, env, said = self.plan(card)
            det = det_of(self.CARDS[card])
            want = G.use_edit_copy(G.use_prefill(G.plan_context(FakePack(FASTEST), det), "auto", det), G.EDIT_COPY_DEFAULT)
            self.assertEqual((max_ctx, env), (want.max_ctx, want.env), card)                # the same as `yasakani chat`
            self.assertEqual({k: env[k] for k in prefill}, prefill, card)
            self.assertEqual(env.get("YSKN_Q35_PF_V2"), "0" if prefill else None, card)
            self.assertEqual(env["YSKN_EDIT_COPY"], "1" if G.EDIT_COPY_DEFAULT == "on" else "0", card)
            self.assertIn(want.prefill_line, said, card)
            with tempfile.TemporaryDirectory() as t:                                      # the profile the script opens the engine with
                prof = RuntimeProfile.default(support.make_pack(Path(t)), env)
            for k, v in prefill.items():
                self.assertEqual(prof.env[k], v, card)

    def test_environment_and_wrong_value(self):
        max_ctx, env, said = self.plan("24 GB", environ={"YASAKANI_PREFILL": "new", "YASAKANI_EDIT_COPY": "on"})
        self.assertNotIn("YSKN_Q35_PF_V2", env)
        self.assertEqual((env["YSKN_Q35_PMAX"], env["YSKN_EDIT_COPY"]), ("1024", "1"))
        self.assertIn("edit copy: on", said)
        with self.assertRaises(YasakaniError):
            self.plan("24 GB", environ={"YASAKANI_PREFILL": "fast", "YASAKANI_EDIT_COPY": ""})

    def test_prefill_new_keeps_room(self):
        """YASAKANI_PREFILL=new on a 24 GB card: the context of gpu.prefill_reserve_mib and its line; a 16 GB card stays as it is."""
        det = det_of(self.CARDS["24 GB"])
        auto = G.plan_context(FakePack(FASTEST), det)
        want = G.plan_context(FakePack(FASTEST), det, G.prefill_reserve_mib("new", det))
        self.assertEqual((auto.max_ctx, want.max_ctx, want.reserve_mib), (None, 180224, G.NEW_PREFILL_RESERVE_MIB))
        max_ctx, env, said = self.plan("24 GB", environ={"YASAKANI_PREFILL": "new"})
        self.assertEqual(max_ctx, 180224)
        self.assertNotIn("YSKN_Q35_PF_V2", env)
        self.assertTrue(any(line.startswith("note: --prefill new: the context is chosen with 1800 MiB more left free") for line in said))
        self.assertEqual(self.plan("24 GB")[0], None)                                     # auto: as before
        max_ctx, env, said = self.plan("16 GB", environ={"YASAKANI_PREFILL": "new"})      # not for more than half of the context
        self.assertEqual(max_ctx, 81920)
        self.assertTrue(any(line.startswith("note: --prefill new: keeping 1800 MiB free") for line in said))

    def test_older_package(self):
        """A package without use_prefill / use_edit_copy (before this release): the plan as it is, no error, no extra line."""
        import types
        older = types.SimpleNamespace(detect=G.detect, untested_note=G.untested_note, plan_context=G.plan_context)
        for card in ("32 GB", "24 GB", "unknown"):
            max_ctx, env, said = self.plan(card, gpus=older, environ={"YASAKANI_PREFILL": "old", "YASAKANI_EDIT_COPY": "on"})
            plan = G.plan_context(FakePack(FASTEST), det_of(self.CARDS[card]))
            self.assertEqual((max_ctx, env), (plan.max_ctx, plan.env), card)
            self.assertFalse([x for x in said if x.startswith(("prompt reading", "edit copy"))], card)


@unittest.skipUnless(support.tokenizer_dir() is not None and os.name == "posix",
                     "needs the Qwen3.8 tokenizer files (YASAKANI_TEST_MODEL_DIR) and a POSIX test machine")
class CliGpuTests(unittest.TestCase):
    """The command line with a stand-in nvidia-smi and the stub engine library."""

    @classmethod
    def setUpClass(cls):
        cls.dll = str(support.stub_path())
        cls.folder = support.make_real_folder("gpu")
        cls.addClassCleanup(shutil.rmtree, cls.folder.parent, True)
        with open(cls.folder / "bdraft2_k8.shead", "wb") as f:                # a sparse file of the published size: the rule sees it
            f.truncate(FASTEST[2])
        tmp = Path(tempfile.mkdtemp(prefix="fake_smi_"))
        cls.addClassCleanup(shutil.rmtree, tmp, True)
        cls.smi = tmp / "nvidia-smi"
        cls.smi.write_text(FAKE_SMI.replace("{python}", sys.executable), encoding="utf-8")
        cls.smi.chmod(0o755)

    def gpu(self, name, cc, total, used, index=0):
        return dict(index=index, uuid="GPU-%d" % index, name=name, compute_cap=cc, **{"memory.total": total, "memory.used": used},
                    driver_version="616.92")

    def run_cli(self, gpus, *args, timeout=60, stub=None):
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", YASAKANI_NVIDIA_SMI=str(self.smi), FAKE_GPUS=json.dumps(gpus))
        for k in [k for k in env if k.startswith(("YSKN_", "STUB_"))] + ["YASAKANI_DLL", "YASAKANI_MODEL_DIR", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"]:
            env.pop(k, None)
        env.update(stub or {})                                                   # knobs of the stub library (tests/stub/yskn_stub.c)
        return subprocess.run([sys.executable, "-m", "yasakani", *args], capture_output=True, text=True, timeout=timeout, cwd=str(ROOT),
                              env=env, input="/exit\n")

    def common(self):
        return ["--model-dir", str(self.folder), "--dll", self.dll]

    def test_fake_smi(self):
        r = subprocess.run([str(self.smi), "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True,
                           env=dict(os.environ, FAKE_GPUS=json.dumps([self.gpu("X", "8.9", 16376, 10)])))
        self.assertEqual(r.stdout.strip(), "X, 16376 MiB")

    def test_rtx_5090_unchanged(self):
        r = self.run_cli([self.gpu("NVIDIA GeForce RTX 5090", "12.0", 32607, 2644)], "chat", *self.common(), "--no-think")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(context 262144 tokens)", r.stderr)
        self.assertNotIn("note:", r.stderr)
        self.assertNotIn("context: ", r.stderr)

    def test_16gb_context_and_note(self):
        gpus = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 11418)]      # about 4,958 MiB free beside the placeholder weights
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NVIDIA GeForce RTX 4080 (compute capability 8.9)", r.stderr)
        self.assertIn("not optimized or tested", r.stderr)
        self.assertIn("bdraft2_k8.shead", r.stderr)
        self.assertIn("context: 98304 tokens", r.stderr)
        self.assertIn("(context 98304 tokens)", r.stderr)
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think", "--max-ctx", "8192")       # given: no rule
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(context 8192 tokens)", r.stderr)
        self.assertNotIn("context: ", r.stderr)
        self.assertNotIn("bdraft2_k8.shead", r.stderr)                                    # the head fits beside 8,192 tokens
        self.assertIn("not optimized or tested", r.stderr)

    def test_given_context_leaves_out_the_head(self):
        gpus = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 11418)]                 # about 4,958 MiB free beside the placeholders
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think", "--max-ctx", "98304")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("bdraft2_k8.shead", r.stderr)
        self.assertIn("is not loaded: it does not fit beside 98304 tokens", r.stderr)
        self.assertIn("(context 98304 tokens)", r.stderr)                               # the given context, unchanged
        self.assertNotIn("context: ", r.stderr)

    def test_does_not_fit(self):
        r = self.run_cli([self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 14000)], "chat", *self.common())
        self.assertEqual(r.returncode, 1)
        self.assertIn("does not fit in the free GPU memory", r.stderr)
        self.assertNotIn("loading the model", r.stderr)

    def test_no_nvidia_smi(self):
        env_gpus = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 14000)]
        env = dict(os.environ, PYTHONPATH=str(ROOT), YASAKANI_NVIDIA_SMI="none", FAKE_GPUS=json.dumps(env_gpus))
        r = subprocess.run([sys.executable, "-m", "yasakani", "chat", *self.common(), "--no-think"], capture_output=True, text=True,
                           timeout=60, cwd=str(ROOT), env=env, input="/exit\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(context 262144 tokens)", r.stderr)

    def test_info(self):
        r = self.run_cli([self.gpu("NVIDIA GeForce RTX 3090", "8.6", 24576, 500)], "info", *self.common())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("NVIDIA GeForce RTX 3090, 24576 MiB, 500 MiB", r.stdout)
        self.assertIn("GPU 0: compute capability 8.6 - runs since 0.1.3, but not optimized or tested", r.stdout)

    def test_two_gpus_device_numbers(self):
        """issue #20: --device N runs on nvidia-smi's GPU N, and the free-memory check looks at that GPU."""
        # the folder's weights are placeholders (only the head has its published size): the used memory stands in for the weights
        gpus = [self.gpu("NVIDIA RTX A2000 12GB", "8.6", 12282, 9500, 0), self.gpu("NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 11353, 1)]
        log = self.folder.parent / "open_two.log"
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think", "--device", "1", stub=dict(STUB_OPEN_LOG=str(log)))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NVIDIA GeForce RTX 5060 Ti (compute capability 12.0)", r.stderr)
        self.assertIn("context: 98304 tokens, chosen from the free GPU memory (4958 of 16311 MiB free)", r.stderr)
        self.assertIn("gpu: GPU 1 of nvidia-smi (NVIDIA GeForce RTX 5060 Ti); the engine sees only this GPU", r.stderr)
        self.assertNotIn("GPU-1", r.stderr)
        self.assertEqual(log.read_text().splitlines(), ["device=0 order=PCI_BUS_ID visible=GPU-1"])      # 0.1.6 (0.1.5: device=1, both visible)
        r = self.run_cli(gpus, "chat", *self.common(), "--device", "0", stub=dict(STUB_OPEN_LOG=str(log)))
        self.assertEqual(r.returncode, 1)
        self.assertIn("(2782 of 12282 MiB free", r.stderr)
        self.assertIn("add --device 1", r.stderr)
        self.assertNotIn("loading the model", r.stderr)
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think", "--device", "0", "--max-ctx", "8192",
                         stub=dict(STUB_OPEN_LOG=str(log), CUDA_DEVICE_ORDER="FASTEST_FIRST"))
        self.assertEqual(r.returncode, 0, r.stderr)                            # a value the user set is kept
        self.assertEqual(log.read_text().splitlines()[-1], "device=0 order=FASTEST_FIRST visible=(unset)")
        self.assertNotIn("gpu: ", r.stderr)
        r = self.run_cli(gpus, "info", *self.common())
        self.assertIn("--device N is GPU N above (PCI bus order); the engine then sees only that GPU", r.stdout)
        r = self.run_cli(gpus[:1], "info", *self.common())
        self.assertNotIn("--device", r.stdout)                                   # one GPU: as before
        nou = [dict(g, uuid="[N/A]") for g in gpus]                             # an nvidia-smi that gives no UUID: the lines say so
        r = self.run_cli(nou, "info", *self.common())
        self.assertIn("--device N is GPU N above (PCI bus order); nvidia-smi gave no UUIDs, so the engine also sees the other GPUs", r.stdout)
        self.assertNotIn("sees only", r.stdout)
        r = self.run_cli(nou, "chat", *self.common(), "--no-think", "--device", "1", stub=dict(STUB_OPEN_LOG=str(log)))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("gpu: GPU 1 of nvidia-smi (NVIDIA GeForce RTX 5060 Ti); nvidia-smi gave no UUID for it, so the engine also sees the "
                      "other GPUs", r.stderr)
        self.assertEqual(log.read_text().splitlines()[-1], "device=1 order=PCI_BUS_ID visible=(unset)")       # nothing hidden (as 0.1.5)

    def test_two_gpus_only_the_chosen_one_visible(self):
        """0.1.6 (issue #20): chat, selfcheck, bench and serve open the engine as device 0 with only the GPU of --device visible; the
        user's CUDA_VISIBLE_DEVICES is kept; one GPU is unchanged."""
        gpus = [self.gpu("NVIDIA RTX A2000 12GB", "8.6", 12282, 9500, 0), self.gpu("NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 11353, 1)]
        log = self.folder.parent / "open_visible.log"
        said = "gpu: GPU 1 of nvidia-smi (NVIDIA GeForce RTX 5060 Ti); the engine sees only this GPU (CUDA_VISIBLE_DEVICES is set to its UUID)"
        ref = self.folder.parent / "ref_visible.json"
        for args in (["chat", "--no-think"], ["selfcheck", "--max-ctx", "4096", "--write", str(ref)], ["bench", "--max-ctx", "4096", "--n", "8"]):
            if log.exists():
                log.unlink()
            r = self.run_cli(gpus, *args, *self.common(), "--device", "1", stub=dict(STUB_OPEN_LOG=str(log)))
            self.assertEqual(r.returncode, 0, (args, r.stdout, r.stderr))
            self.assertIn(said, r.stderr)
            self.assertEqual(log.read_text().splitlines(), ["device=0 order=PCI_BUS_ID visible=GPU-1"], args)
        self.assertEqual(json.loads(ref.read_text(encoding="utf-8"))["gpu"], "NVIDIA GeForce RTX 5060 Ti")   # the GPU of --device
        log.unlink()
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", YASAKANI_NVIDIA_SMI=str(self.smi), FAKE_GPUS=json.dumps(gpus),
                   STUB_OPEN_LOG=str(log))
        for k in [k for k in env if k.startswith(("YSKN_", "STUB_")) and k != "STUB_OPEN_LOG"] + ["YASAKANI_DLL", "YASAKANI_MODEL_DIR",
                                                                                                "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"]:
            env.pop(k, None)
        proc = subprocess.Popen([sys.executable, "-m", "yasakani", "serve", *self.common(), "--device", "1", "--port", "0", "--no-warmup"],
                                cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            t0 = time.time()
            while time.time() - t0 < 40 and not (log.exists() and log.read_text().strip()):
                time.sleep(0.2)
            time.sleep(0.5)
            proc.send_signal(signal.SIGINT)
            out, err = proc.communicate(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        self.assertEqual(log.read_text().splitlines(), ["device=0 order=PCI_BUS_ID visible=GPU-1"], err)
        self.assertIn(said, err)
        r = self.run_cli(gpus, "chat", *self.common(), "--no-think", "--max-ctx", "8192", stub=dict(STUB_OPEN_LOG=str(log)))
        self.assertEqual(r.returncode, 0, r.stderr)                            # no --device: GPU 0 of nvidia-smi, the same way
        self.assertEqual(log.read_text().splitlines()[-1], "device=0 order=PCI_BUS_ID visible=GPU-0")
        for user in ("GPU-1", "1"):                                             # the user's CUDA_VISIBLE_DEVICES: kept, as in 0.1.5
            r = self.run_cli(gpus, "chat", *self.common(), "--no-think", stub=dict(STUB_OPEN_LOG=str(log), CUDA_VISIBLE_DEVICES=user))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(log.read_text().splitlines()[-1], "device=0 order=PCI_BUS_ID visible=%s" % user)
            self.assertNotIn("gpu: ", r.stderr)
            self.assertIn("NVIDIA GeForce RTX 5060 Ti (compute capability 12.0)", r.stderr)       # the plan is for the 5060 Ti
        one = [self.gpu("NVIDIA GeForce RTX 5060 Ti", "12.0", 16311, 11353, 0)]
        r = self.run_cli(one, "chat", *self.common(), "--no-think", stub=dict(STUB_OPEN_LOG=str(log)))
        self.assertEqual(r.returncode, 0, r.stderr)                            # one GPU: nothing changes
        self.assertEqual(log.read_text().splitlines()[-1], "device=0 order=PCI_BUS_ID visible=(unset)")
        self.assertNotIn("gpu: ", r.stderr)

    def test_two_gpus_parallel_chooses_before_the_library_loads(self):
        """Several requests at once load the engine library early (to look for its multi-stream entries).  The GPU of --device must
        be chosen before that, or CUDA would see both GPUs."""
        gpus = [self.gpu("NVIDIA RTX A2000 12GB", "8.6", 12282, 9500, 0), self.gpu("NVIDIA GeForce RTX 5090", "12.0", 32607, 2644, 1)]
        log = self.folder.parent / "open_parallel.log"
        r = self.run_cli(gpus, "bench", *self.common(), "--device", "1", "--n", "8", "--concurrency", "2", "--parallel-check", "off",
                         stub=dict(STUB_OPEN_LOG=str(log)))
        self.assertEqual(r.returncode, 0, (r.stdout, r.stderr))
        self.assertIn("the engine sees only this GPU", r.stderr)
        lines = log.read_text().splitlines()
        self.assertTrue(lines, r.stderr)
        self.assertEqual(set(lines), {"device=0 order=PCI_BUS_ID visible=GPU-1"}, r.stderr)

    def test_report_prompt_reading(self):
        """yasakani report on a 24 GB card: the reading of 0.1.7 in machine.prefill and in the Markdown row."""
        from yasakani import report
        r = self.run_cli([self.gpu("NVIDIA GeForce RTX 4090", "8.9", 24564, 1500)], "report", *self.common(), "--no-bench",
                         "--no-selfcheck", "--json", timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        rep = json.loads(r.stdout)
        line = ("prompt reading: as in 0.1.7 (setting auto: 24564 MiB card and no --max-ctx; "
                "--prefill new or YASAKANI_PREFILL=new to try it)")
        self.assertEqual(rep["machine"]["prefill"], dict(reading="old", line=line))
        self.assertEqual(rep["machine"]["edit_copy"], G.EDIT_COPY_DEFAULT)
        self.assertIn(line, r.stderr)
        self.assertIn("| Prompt reading | %s |" % line, report.render(rep).split("\n"))

    def test_selfcheck_exit_codes(self):
        ref = self.folder.parent / "ref_gpu.json"
        args = ["--model-dir", str(self.folder), "--dll", self.dll, "--max-ctx", "4096"]
        g5090 = [self.gpu("NVIDIA GeForce RTX 5090", "12.0", 32607, 2644)]
        g4080 = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 1000)]
        diff = dict(STUB_GEN_DIFF_AT="5")                                         # the generated output differs from id #6 on
        r = self.run_cli(g5090, "selfcheck", *args, "--write", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        d = json.loads(ref.read_text(encoding="utf-8"))
        for c in d["cases"]:                                                      # 0.1.4: the full output and the predictions
            self.assertEqual(len(c["ids"]), c["tokens"])
            self.assertEqual(len(c["forced"]["nll"]), c["prompt_tokens"] + c["tokens"] - 1)
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)                 # the same output and the same predictions
        self.assertIn("its predictions are also compared", r.stdout)
        self.assertIn("same most likely id at 100.0 %", r.stdout)
        self.assertIn("self-check passed: this installation reproduces the reference output.", r.stdout)
        r = self.run_cli(g5090, "selfcheck", *args, "--reference", str(ref), stub=dict(STUB_TF_NLL_SHIFT="0.05"))
        self.assertEqual(r.returncode, 0, r.stdout)                            # the RTX 5090: the output alone decides, as before
        self.assertNotIn("reads the reference output", r.stdout)
        r = self.run_cli(g5090, "selfcheck", *args, "--reference", str(ref), stub=diff)
        self.assertEqual(r.returncode, 1, r.stdout)                            # on the RTX 5090 a difference is a failure, as before
        self.assertIn("FAIL  the output differs", r.stdout)
        self.assertIn("self-check FAILED for 3 case(s).", r.stdout)
        self.assertNotIn("reads the reference output", r.stdout)
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref), stub=diff)
        self.assertEqual(r.returncode, 3, r.stdout)                            # elsewhere: the predictions on the reference output decide
        self.assertIn("DIFF  the output differs", r.stdout)
        self.assertIn("made on an RTX 5090", r.stdout)
        n = sum(c["prompt_tokens"] + c["tokens"] - 1 for c in d["cases"])
        self.assertIn("all %d positions: same most likely id at 100.0 %%" % n, r.stdout)
        self.assertIn("(generated output differs from id #6)", r.stdout)
        self.assertIn("passed within the tolerances", r.stdout)
        self.assertIn("Exit code 3", r.stdout)
        for knob, text in ((dict(STUB_TF_NLL_SHIFT="0.05"), "differs by 0.0500 on average"),        # a GPU that computes wrongly
                           (dict(STUB_TF_FLIP_EVERY="10"), "the most likely id is the same at only 9"),
                           (dict(STUB_TF_NAN="30"), "NaN")):
            r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref), stub=dict(diff, **knob))
            self.assertEqual(r.returncode, 1, (knob, r.stdout))
            self.assertIn("self-check FAILED: ", r.stdout)
            self.assertIn(text, r.stdout)
            self.assertIn("yasakani verify", r.stdout)
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref), stub=dict(STUB_TF_NLL_SHIFT="0.05"))
        self.assertEqual(r.returncode, 1, r.stdout)                            # the same output, but wrong predictions: a failure
        self.assertIn("differs by 0.0500 on average", r.stdout)
        old = self.folder.parent / "ref_gpu_old.json"                            # a reference file written before 0.1.4
        old.write_text(json.dumps(dict(d, cases=[{k: v for k, v in c.items() if k not in ("ids", "forced")} for c in d["cases"]])),
                       encoding="utf-8")
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(old))
        self.assertEqual(r.returncode, 0, r.stdout)                            # the same output; nothing more to compare with
        self.assertNotIn("also compared", r.stdout)
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(old), stub=diff)
        self.assertEqual(r.returncode, 1, r.stdout)                            # nothing to compare with: not passed
        self.assertIn("has no full output", r.stdout)
        d["cases"][2]["prompt_sha256"] = "1" * 64
        ref.write_text(json.dumps(d), encoding="utf-8")
        r = self.run_cli(g4080, "selfcheck", *args, "--reference", str(ref), stub=diff)
        self.assertEqual(r.returncode, 1, r.stdout)                            # a tokenizer problem stays a failure
        self.assertIn("prompt ids differ", r.stdout)
        self.assertNotIn("reads the reference output", r.stdout)

    def test_selfcheck_consistency(self):
        """0.1.4: every GPU also checks its generated output against its own teacher-forced predictions (16- and 64-row passes)."""
        ref = self.folder.parent / "ref_cons.json"
        args = ["--model-dir", str(self.folder), "--dll", self.dll, "--max-ctx", "4096"]
        g5090 = [self.gpu("NVIDIA GeForce RTX 5090", "12.0", 32607, 2644)]
        g5080 = [self.gpu("NVIDIA GeForce RTX 5080", "12.0", 16303, 1000)]
        g4080 = [self.gpu("NVIDIA GeForce RTX 4080", "8.9", 16376, 1000)]
        r = self.run_cli(g4080, "selfcheck", *args, "--write", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("not known to be an RTX 5090", r.stdout)                  # --write elsewhere: a note, and the GPU in the file
        self.assertEqual(json.loads(ref.read_text(encoding="utf-8"))["gpu"], "NVIDIA GeForce RTX 4080")
        r = self.run_cli(g5090, "selfcheck", *args, "--write", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("not known to be an RTX 5090", r.stdout)
        self.assertEqual(json.loads(ref.read_text(encoding="utf-8"))["gpu"], "NVIDIA GeForce RTX 5090")
        self.assertNotIn("self-consistency", r.stdout)                          # --write checks nothing
        r = self.run_cli(g5090, "selfcheck", *args, "--reference", str(ref))
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("self-check passed: this installation reproduces the reference output.", r.stdout)
        self.assertIn("self-consistency (every GPU, no reference)", r.stdout)
        self.assertIn("every generated id is the most likely id (41 ids)", r.stdout)
        self.assertIn("note: this GPU does not use passes of more than 16 positions", r.stdout)
        self.assertIn("self-consistency passed.", r.stdout)
        wide = dict(STUB_TF_WIDE="1", STUB_GEN_MAX_ROWS="64", STUB_GEN_WIDE_PASSES="3")
        r = self.run_cli(g5090, "selfcheck", *args, "--reference", str(ref), stub=wide)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("3 of them wider than 16 positions, the widest 64", r.stdout)
        self.assertIn("the same predictions as in passes of 16, bit for bit", r.stdout)
        r = self.run_cli(g5080, "selfcheck", *args, "--reference", str(ref), stub=dict(wide, STUB_GEN_DIFF_AT="5"))
        self.assertEqual(r.returncode, 3, r.stdout)                            # another RTX 50: within the tolerances, consistent
        self.assertIn("passed within the tolerances", r.stdout)
        self.assertIn("self-consistency passed.", r.stdout)
        for gpus, knob, text in (
                (g5090, dict(STUB_TF_FLIP_EVERY="10"), "of the 41 generated ids are not the most likely id when this GPU reads"),
                (g5090, dict(wide, STUB_TF_WIDE_NLL_SHIFT="0.000001"), "passes of 64 positions do not give the same predictions"),
                (g5090, dict(STUB_GEN_MAX_ROWS="64"), "were used in the generation, but reading the text in passes of 64 failed"),
                (g5080, dict(wide, STUB_TF_WIDE_NLL_SHIFT="0.000001"), "passes of 64 positions do not give the same predictions")):
            r = self.run_cli(gpus, "selfcheck", *args, "--reference", str(ref), stub=knob)
            self.assertEqual(r.returncode, 1, (knob, r.stdout))
            self.assertIn("self-check FAILED: ", r.stdout)
            self.assertIn(text, r.stdout)
            self.assertIn("the same sums in the same order", r.stdout)


@unittest.skipUnless(support.tokenizer_dir() is not None and os.name == "posix",
                     "needs the Qwen3.8 tokenizer files (YASAKANI_TEST_MODEL_DIR) and a POSIX test machine")
class OpenStackTests(unittest.TestCase):
    """_open_stack passes the rule's context and engine setting to the Engine (in this process, the stub library)."""

    def test_head_off_reaches_the_engine(self):
        from yasakani import cli
        folder = support.make_real_folder("stack")
        self.addCleanup(shutil.rmtree, folder.parent, True)
        with open(folder / "bdraft2_k8.shead", "wb") as f:
            f.truncate(FASTEST[2])
        gpus = G.parse(smi_line(0, "NVIDIA GeForce RTX 5070 Ti", "12.0", 16303, 11300))
        args = cli.build_parser().parse_args(["chat", "--model-dir", str(folder), "--dll", str(support.stub_path())])
        saved = {k: v for k, v in os.environ.items() if k.startswith("YSKN_")}

        def restore():                                                           # the engine settings of this process as before
            for k in [k for k in os.environ if k.startswith("YSKN_")]:
                del os.environ[k]
            os.environ.update(saved)
        self.addCleanup(restore)
        with mock.patch.object(G, "query", return_value=gpus):
            pack, eng, fmt, runner = cli._open_stack(args, warm=False)
        try:
            self.assertEqual(eng.max_ctx, 98304)
            self.assertEqual(eng.profile.env["YSKN_BDR_SHEAD"], "0")
            self.assertEqual(os.environ["YSKN_BDR_SHEAD"], "0")
            self.assertTrue(args.gpu_detected.untested)
        finally:
            eng.close()

    def test_prefill_new_keeps_room(self):
        """--prefill new on a 24 GB card: _open_stack plans with gpu.prefill_reserve_mib (the plan keeps the reserve, the engine opens
        with the plan's context); auto plans without it."""
        from yasakani import cli
        folder = support.make_real_folder("reserve")
        self.addCleanup(shutil.rmtree, folder.parent, True)
        with open(folder / "bdraft2_k8.shead", "wb") as f:
            f.truncate(FASTEST[2])
        saved = {k: v for k, v in os.environ.items() if k.startswith("YSKN_") or k == "YASAKANI_PREFILL"}

        def restore():
            for k in [k for k in os.environ if k.startswith("YSKN_") or k == "YASAKANI_PREFILL"]:
                del os.environ[k]
            os.environ.update(saved)
        self.addCleanup(restore)
        os.environ.pop("YASAKANI_PREFILL", None)
        base = ["chat", "--model-dir", str(folder), "--dll", str(support.stub_path())]
        got = {}
        for mode in ("auto", "new"):
            gpus = G.parse(smi_line(0, "NVIDIA GeForce RTX 4090", "8.9", 24564, 4000))
            args = cli.build_parser().parse_args(base + ["--prefill", mode])
            with mock.patch.object(G, "query", return_value=gpus):
                pack, eng, fmt, runner = cli._open_stack(args, warm=False)
            try:
                det = G.detect(0, environ={}, gpus=gpus)
                want = G.plan_context(pack, det, G.prefill_reserve_mib(mode, det))
                got[mode] = args.context_plan.reserve_mib
                self.assertEqual(eng.max_ctx, want.max_ctx or pack.max_ctx, mode)
                self.assertEqual((args.context_plan.reserve_mib, args.context_plan.max_ctx), (want.reserve_mib, want.max_ctx), mode)
            finally:
                eng.close()
        self.assertEqual(got, {"auto": 0, "new": G.NEW_PREFILL_RESERVE_MIB})

    def test_prefill_old_reaches_the_engine(self):
        """--prefill old and YASAKANI_PREFILL=old set YSKN_Q35_PF_V2=0 for the engine; auto leaves the default."""
        from yasakani import cli
        folder = support.make_real_folder("prefill")
        self.addCleanup(shutil.rmtree, folder.parent, True)
        gpus = G.parse(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607, 1107))
        saved = {k: v for k, v in os.environ.items() if k.startswith("YSKN_") or k == "YASAKANI_PREFILL"}

        def restore():
            for k in [k for k in os.environ if k.startswith("YSKN_") or k == "YASAKANI_PREFILL"]:
                del os.environ[k]
            os.environ.update(saved)
        self.addCleanup(restore)
        base = ["chat", "--model-dir", str(folder), "--dll", str(support.stub_path())]
        for extra, env, want in (([], None, None), (["--prefill", "old"], None, "0"), ([], "old", "0"), (["--prefill", "auto"], "old", None),
                                 (["--prefill", "new"], "old", None), ([], "new", None)):
            os.environ.pop("YASAKANI_PREFILL", None)
            if env:
                os.environ["YASAKANI_PREFILL"] = env
            args = cli.build_parser().parse_args(base + extra)
            with mock.patch.object(G, "query", return_value=gpus):
                pack, eng, fmt, runner = cli._open_stack(args, warm=False)
            try:
                self.assertEqual(eng.profile.env.get("YSKN_Q35_PF_V2"), want, (extra, env))
                self.assertEqual(os.environ.get("YSKN_Q35_PF_V2"), want, (extra, env))
                self.assertEqual(eng.profile.env["YSKN_Q35_PMAX"], "1024" if want else "2048")
            finally:
                eng.close()

    def test_edit_copy_reaches_the_engine(self):
        """--edit-copy and YASAKANI_EDIT_COPY set YSKN_EDIT_COPY for the engine; neither = EDIT_COPY_DEFAULT; a YSKN_EDIT_COPY left in
        the shell does not count."""
        from yasakani import cli
        folder = support.make_real_folder("editcopy")
        self.addCleanup(shutil.rmtree, folder.parent, True)
        gpus = G.parse(smi_line(0, "NVIDIA GeForce RTX 5090", "12.0", 32607, 1107))
        saved = {k: v for k, v in os.environ.items() if k.startswith("YSKN_") or k == "YASAKANI_EDIT_COPY"}

        def restore():
            for k in [k for k in os.environ if k.startswith("YSKN_") or k == "YASAKANI_EDIT_COPY"]:
                del os.environ[k]
            os.environ.update(saved)
        self.addCleanup(restore)
        base = ["chat", "--model-dir", str(folder), "--dll", str(support.stub_path())]
        for default in ("off", "on"):
            d = "1" if default == "on" else "0"
            for extra, env, want in (([], None, d), (["--edit-copy", "on"], "off", "1"), ([], "on", "1"), ([], "off", "0"),
                                     (["--edit-copy", "off"], "on", "0")):
                os.environ.pop("YASAKANI_EDIT_COPY", None)
                os.environ["YSKN_EDIT_COPY"] = "1" if want == "0" else "0"          # left in the shell: replaced
                if env:
                    os.environ["YASAKANI_EDIT_COPY"] = env
                args = cli.build_parser().parse_args(base + extra)
                err = io.StringIO()
                with mock.patch.object(G, "query", return_value=gpus), mock.patch.object(G, "EDIT_COPY_DEFAULT", default),                         contextlib.redirect_stderr(err):
                    pack, eng, fmt, runner = cli._open_stack(args, warm=False)
                try:
                    # the shell's value is not used, and the note says so (as 0.1.7 said for the settings it removed)
                    notes = [x for x in err.getvalue().splitlines() if x.startswith("note: ignored ")]
                    self.assertEqual(len(notes), 1, err.getvalue())
                    self.assertIn("YSKN_EDIT_COPY", notes[0])
                    self.assertEqual(eng.profile.env["YSKN_EDIT_COPY"], want, (default, extra, env))
                    self.assertEqual(os.environ["YSKN_EDIT_COPY"], want, (default, extra, env))
                    self.assertEqual(args.context_plan.edit_copy, "on" if want == "1" else "off")
                finally:
                    eng.close()

    def stack(self, card, argv, parallel=1):
        """_open_stack in this process on the card ``card`` (PrefillModeTests.CARDS): the engine's settings, and the args."""
        from yasakani import cli
        folder = support.make_real_folder("card")
        self.addCleanup(shutil.rmtree, folder.parent, True)
        keys = ("YASAKANI_PREFILL", "YASAKANI_EDIT_COPY")
        saved = {k: v for k, v in os.environ.items() if k.startswith("YSKN_") or k in keys}

        def restore():
            for k in [k for k in os.environ if k.startswith("YSKN_") or k in keys]:
                del os.environ[k]
            os.environ.update(saved)
        self.addCleanup(restore)
        for k in [k for k in os.environ if k.startswith("YSKN_") or k in keys]:
            del os.environ[k]
        args = cli.build_parser().parse_args(argv + ["--model-dir", str(folder), "--dll", str(support.stub_path())])
        with mock.patch.object(G, "query", return_value=G.parse(PrefillModeTests.CARDS[card] + "\n")), \
                contextlib.redirect_stderr(io.StringIO()):
            pack, eng, fmt, runner = cli._open_stack(args, warm=False, parallel=parallel)
        try:
            env = {k: eng.profile.env.get(k) for k in ("YSKN_Q35_PF_V2", "YSKN_Q35_PMAX", "YSKN_Q35_LAYER_ROWS", "YSKN_EDIT_COPY")}
            self.assertEqual(env, {k: os.environ.get(k) for k in env}, card)                  # what the engine reads
        finally:
            eng.close()
        return env, args

    def test_settings_on_each_card(self):
        """auto and new on a 32 GB card, a 24 GB card and a card of unknown memory: the settings the engine gets."""
        from yasakani.runtime import DEFAULT_ENV
        big = (None, DEFAULT_ENV["YSKN_Q35_PMAX"], DEFAULT_ENV["YSKN_Q35_LAYER_ROWS"])
        old, small_new = ("0", "1024", "4096"), (None, "1024", "4096")
        off = "1" if G.EDIT_COPY_DEFAULT == "on" else "0"
        want = {("32 GB", "auto"): big, ("24 GB", "auto"): old, ("unknown", "auto"): old,
                ("32 GB", "new"): big, ("24 GB", "new"): small_new, ("unknown", "new"): small_new}
        self.assertEqual(big[1:], ("2048", "8192"))
        for (card, mode), (pf, pmax, rows) in want.items():
            env, args = self.stack(card, ["chat"] + (["--prefill", mode] if mode != "auto" else []))
            self.assertEqual(env, {"YSKN_Q35_PF_V2": pf, "YSKN_Q35_PMAX": pmax, "YSKN_Q35_LAYER_ROWS": rows, "YSKN_EDIT_COPY": off},
                             (card, mode))
            self.assertEqual(args.context_plan.prefill, "old" if pf == "0" else "new", (card, mode))

    def test_parallel_on_24_gb(self):
        """serve --parallel 2 on a 24 GB card: the context for two requests, and the reading of 0.1.7 as for one."""
        env, args = self.stack("24 GB", ["serve", "--edit-copy", "on"], parallel=2)
        self.assertEqual(args.parallel_planned, 2)
        self.assertEqual(args.context_plan.parallel, 2)
        self.assertEqual(env, {"YSKN_Q35_PF_V2": "0", "YSKN_Q35_PMAX": "1024", "YSKN_Q35_LAYER_ROWS": "4096", "YSKN_EDIT_COPY": "1"})

    def test_given_context_head_reaches_the_engine(self):
        """--max-ctx: the head is left out where it does not fit beside the given context, and loaded where it does."""
        from yasakani import cli
        folder = support.make_real_folder("given")
        self.addCleanup(shutil.rmtree, folder.parent, True)
        with open(folder / "bdraft2_k8.shead", "wb") as f:
            f.truncate(FASTEST[2])
        gpus = G.parse(smi_line(0, "NVIDIA GeForce RTX 5070 Ti", "12.0", 16303, 11300))     # about 5,000 MiB free
        saved = {k: v for k, v in os.environ.items() if k.startswith("YSKN_")}

        def restore():
            for k in [k for k in os.environ if k.startswith("YSKN_")]:
                del os.environ[k]
            os.environ.update(saved)
        self.addCleanup(restore)
        for ctx, head_off in (("98304", True), ("16384", False)):
            args = cli.build_parser().parse_args(["chat", "--model-dir", str(folder), "--dll", str(support.stub_path()), "--max-ctx", ctx])
            with mock.patch.object(G, "query", return_value=gpus):
                pack, eng, fmt, runner = cli._open_stack(args, warm=False)
            try:
                head = "0" if head_off else str(pack.draft_head)                     # "0" leaves it out; its path loads it
                self.assertEqual(eng.max_ctx, int(ctx))
                self.assertEqual(eng.profile.env["YSKN_BDR_SHEAD"], head)
                self.assertEqual(os.environ["YSKN_BDR_SHEAD"], head)
            finally:
                eng.close()


if __name__ == "__main__":
    unittest.main()
