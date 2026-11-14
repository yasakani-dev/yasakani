# SPDX-License-Identifier: Apache-2.0
"""Backend-specific diagnostics must not imply CUDA checks passed on Metal."""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from yasakani import cli
from yasakani.runtime import ModelPack


class MetalCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.weights = self.root / "qwen38_27b_e3x.yskn"
        self.weights.write_bytes(b"diagnostic fixture weights")
        for filename in ("tokenizer.json", "tokenizer_config.json"):
            (self.root / filename).write_text("{}", encoding="utf-8")

    def manifest(self, extra):
        sums = {self.weights.name: hashlib.sha256(self.weights.read_bytes()).hexdigest(), **extra}
        (self.root / "yasakani.json").write_text(json.dumps({"sha256": sums}), encoding="utf-8")

    def verify_cli(self):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        return subprocess.run([sys.executable, "-m", "yasakani", "verify", "--backend", "metal",
                               "--model-dir", str(self.root)], env=env, capture_output=True, text=True, timeout=10)

    def test_info_finds_metal_library_and_accepts_pack_without_auxiliary_files(self):
        library = self.root / "libyskn_metal.dylib"
        library.write_bytes(b"diagnostic fixture library")
        with mock.patch("yasakani.textio.ChatFormatter") as formatter, \
                mock.patch.object(cli, "find_library", side_effect=AssertionError("CUDA lookup must not run")), \
                mock.patch.object(cli, "_gpu_lines", side_effect=AssertionError("NVIDIA lookup must not run")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = cli.main(["info", "--backend", "metal", "--dll", str(library), "--model-dir", str(self.root)])
            self.assertEqual(rc, 0, output.getvalue())
            self.assertIn("libyskn_metal.dylib", output.getvalue())
            self.assertIn("backend  metal", output.getvalue())
            self.assertIn("tokenizer  ok", output.getvalue())
            self.assertNotIn("draft", output.getvalue())
            formatter.assert_called_once_with(self.root)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = cli.main(["info", "--backend", "metal", "--dll", str(self.root / "missing.dylib")])
            self.assertEqual(rc, 1)
            self.assertIn("engine   NOT FOUND", output.getvalue())

    def test_verify_explicitly_skips_only_missing_cuda_auxiliaries(self):
        auxiliary = ("bdraft2_k8.yskn", "bdraft2_k8.shead", "kvq_u5.bin")
        self.manifest({filename: "0" * 64 for filename in auxiliary})
        result = self.verify_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("SKIP (not used by Metal and not downloaded)"), 3)
        self.assertIn("all files match", result.stdout)
        # Downloaded auxiliary files are checked; Metal is not a checksum bypass.
        (self.root / auxiliary[0]).write_bytes(b"incorrect auxiliary")
        result = self.verify_cli()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("SKIP (not used by Metal and not downloaded)"), 2)
        self.assertIn("DIFFERENT", result.stdout)
        self.assertNotIn("all files match", result.stdout)

    def test_verify_missing_unrecognized_checksum_file_is_failure_not_skip(self):
        self.manifest({"required-extra.bin": "0" * 64})
        result = self.verify_cli()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("required-extra.bin", result.stdout + result.stderr)
        self.assertNotIn("SKIP", result.stdout)
        self.assertNotIn("all files match", result.stdout)

    def test_selfcheck_rejects_cuda_and_legacy_reference_for_metal(self):
        pack = ModelPack.load(self.root, require_aux=False)
        case = {"name": "fixture", "prompt_sha256": "a", "sha256": "b", "tokens": 1, "head": [2]}
        reference = self.root / "selfcheck-metal.json"
        for backend in ("cuda", None):
            data = {"format": cli.SELFCHECK_FORMAT, "cases": [case]}
            if backend is not None:
                data["backend"] = backend
            reference.write_text(json.dumps(data), encoding="utf-8")
            engine = mock.Mock()
            output = io.StringIO()
            with self.subTest(backend=backend), \
                    mock.patch.object(cli, "_open_stack", return_value=(pack, engine, mock.Mock(), mock.Mock())), \
                    mock.patch.object(cli, "_selfcheck_run", return_value=[case]), contextlib.redirect_stdout(output):
                result = cli.main(["selfcheck", "--backend", "metal", "--model-dir", str(self.root)])
            self.assertEqual(result, 2)
            self.assertIn("different backend", output.getvalue())
            self.assertNotIn("PASS", output.getvalue())
            engine.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
