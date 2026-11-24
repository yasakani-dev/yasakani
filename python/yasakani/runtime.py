# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""The default runtime: environment settings, speculation settings and the model package.

The values below are the configuration the published numbers were measured with.  Change them only to experiment: a different
setting is a different engine configuration and its output is not covered by the published checks.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ._native import YasakaniError

# ---- Qwen3.8 ids used by the engine settings --------------------------------------------------------------------------------------
EOS_IDS = (248046, 248044)         # <|im_end|>, <|endoftext|>
IM_START = 248045                  # <|im_start|>: the prompt snapshot is taken right after its last occurrence
VOCAB_SIZE = 248320
HIDDEN = 5120
MIN_UNSUPPRESSED_TOKENS = 20       # sampler top-k cap (Q35_SMP_K): fewer leaves placeholder entries in candidate lists
PROTECTED_CONTROL_TOKENS = (       # special: false in Qwen3.8 tokenizer.json, but suppressing them breaks thinking or tool calls
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)

# ---- environment: read by the engine library when the model is opened -------------------------------------------------------------
# Value settings.  The two path settings (YSKN_BDR_SHEAD, YSKN_TB_PROFILE) come from the model package.
DEFAULT_ENV: Dict[str, str] = {
    "YSKN_BDR_SHEAD_M": "512",
    "YSKN_Q35_G6": "1",
    "YSKN_Q35_KV8": "1",
    "YSKN_Q35_KV_FP16": "0",
    "YSKN_Q35_LAYER_OVERLAP": "1",
    "YSKN_Q35_LAYER_ROWS": "8192",
    "YSKN_Q35_PFALT": "1",
    "YSKN_Q35_PMAX": "2048",
    "YSKN_Q35_SHARED_WEIGHTS": "1",
    "YSKN_Q35_ATX": "1",
    "YSKN_TB_CAP": "1",
    "YSKN_TB_VIT2": "1",
    "YSKN_ATX_PK3_PLUT": "1",
    "YSKN_ATX_PKV": "3",
    "YSKN_ATX_VSP": "7",
    # prompt snapshots (0.1.4): up to 8 saved states in pinned host memory (about 187 MiB each); for prompts of 8,192 ids or more
    # also on the 1,024-id grid in the last 4,096 ids and every 65,536 ids. A cold read
    # can fill otherwise unused slots in long gaps for subsequent edits.
    "YSKN_SNAP_SLOTS": "8",
    "YSKN_SNAP_STEP": "1024",
    "YSKN_SNAP_TAIL": "4096",
    "YSKN_SNAP_SPARSE": "65536",
    "YSKN_SNAP_MIN": "8192",
}
PATH_ENV = ("YSKN_BDR_SHEAD", "YSKN_TB_PROFILE")
TEST_ENV_PREFIX = "YSKN_TEST_"     # read only by engine libraries built with the test exports; kept from the shell (RuntimeProfile.apply)

# ---- speculation settings: the 46 integers of the generation call, in the order the engine reads them ----------------------------
SPEC_FIELDS = ("gamma", "use_d0", "use_d1", "d0_nmin", "d0_nmax", "d0_priority_n", "extend", "d1_rejected", "auto_gamma", "use_graph",
               "use_d3", "d3_steps", "d1_bigram", "d1_static", "d3_extend", "d3_lazy", "d3_gate", "gamma_long", "long_n", "d1_trigram",
               "eos1", "eos2", "long_d0", "sel", "cost_tok_us", "cost_row_us", "cost_nd_us", "nd_auto", "prmode", "cost_row2_us",
               "lfrow", "lfrow_q_pm", "mark_tok", "cost_tok_k", "cost_row2_k", "cost_nd_k", "bdraft", "bdr_long_n", "bdr_adapt",
               "bdr_probe", "bdr_d0_len", "prefill_head", "snap", "snap_tok", "copy_rows", "copy_index")
assert len(SPEC_FIELDS) == 46

DEFAULT_SPEC: Dict[str, int] = dict(
    gamma=7, use_d0=1, use_d1=1, d0_nmin=2, d0_nmax=8, d0_priority_n=3, extend=1, d1_rejected=1, auto_gamma=0, use_graph=1,
    use_d3=0, d3_steps=0, d1_bigram=1, d1_static=0, d3_extend=0, d3_lazy=0, d3_gate=0, gamma_long=15, long_n=6, d1_trigram=1,
    eos1=EOS_IDS[0], eos2=EOS_IDS[1], long_d0=8, sel=0, cost_tok_us=0, cost_row_us=0, cost_nd_us=0, nd_auto=0, prmode=0,
    cost_row2_us=0, lfrow=0, lfrow_q_pm=300, mark_tok=-1, cost_tok_k=0, cost_row2_k=0, cost_nd_k=0, bdraft=1, bdr_long_n=4,
    bdr_adapt=1, bdr_probe=4, bdr_d0_len=15, prefill_head=1, snap=1, snap_tok=IM_START,
    copy_rows=64, copy_index=1)            # 0.1.4: copy passes up to 64 rows, copy index with cursor (copy_rows=16, copy_index=0 = 0.1.3)
assert tuple(DEFAULT_SPEC) == SPEC_FIELDS

# settings that decide how the engine keeps its state between calls; not tunable per request
STATE_FIELDS = ("prefill_head", "snap", "snap_tok")

# statistics returned by the generation call (index in the 64 doubles; 43 and later need an engine library of 0.1.4 or newer)
N_STATS = 64
STAT_INDEX = dict(passes=0, tokens=1, drafted_0=3, accepted_0=4, drafted_1=5, accepted_1=6, drafted_2=23, accepted_2=24,
                  produced=30, eos=31, snap_mode=41, cached_tokens=42, bdr_rounds=37, bdr_d0=38,
                  copy_passes=43, wide_passes=44, copy_tokens=45, index_ms=46, max_rows=47, common_prefix=48, snap_slots=49,
                  snap_saved=50, snap_ms=51, passes_48=52, passes_64=53, wide_tokens=54, cursor_hits=55, wide_full=56, passes_32=57,
                  # 0.1.7 (0 with older engine libraries): times the copy loop decided that wide passes were slower than 16 rows, its
                  # final times of copy cycles of 16 / 32 / 48 / 64 rows (ms) and the widths it could use (bit 0 = 32, 1 = 48, 2 = 64)
                  wide_backoffs=58, t_copy_16=59, t_copy_32=60, t_copy_48=61, t_copy_64=62, wide_mask=63)


def spec_array(spec: Dict[str, int]):
    import numpy as np
    unknown = set(spec) - set(SPEC_FIELDS)
    if unknown:
        raise ValueError("unknown speculation setting(s): " + ", ".join(sorted(unknown)))
    return np.array([int(spec[k]) for k in SPEC_FIELDS], np.int32)


@dataclass
class RuntimeProfile:
    """Environment variables applied before the model is opened."""
    env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def default(cls, pack: "ModelPack", overrides: Optional[Dict[str, str]] = None) -> "RuntimeProfile":
        if pack.is_hf_inkling2:                   # the engine of #28 has no drafter head or KV profile
            env: Dict[str, str] = {k: v for k, v in os.environ.items() if k.startswith("YSKN_INK2_")}
            for k, v in (overrides or {}).items():
                if not k.startswith("YSKN_"):
                    raise ValueError("engine settings start with YSKN_: " + k)
                env[k] = str(v)
            return cls(env)
        env = dict(DEFAULT_ENV)
        env["YSKN_BDR_SHEAD"] = str(pack.draft_head)
        env["YSKN_TB_PROFILE"] = str(pack.kv_profile)
        for k, v in (overrides or {}).items():
            if not k.startswith("YSKN_"):
                raise ValueError("engine settings start with YSKN_: " + k)
            env[k] = str(v)
        return cls(env)

    def apply(self, environ=None) -> List[str]:
        """Make the process environment exactly this profile: engine settings left over from the shell are removed first (a stale
        setting would silently change the engine).  Returns the names that were removed.  ``YSKN_TEST_*`` settings are kept: only an
        engine library built with the test exports reads them (``YSKN_TEST_CODE_ARCH``, 0.1.7), the released library ignores them."""
        environ = os.environ if environ is None else environ
        removed = [k for k in list(environ) if k.startswith("YSKN_") and not k.startswith(TEST_ENV_PREFIX) and k not in self.env]
        for k in removed:
            environ.pop(k)
        environ.update(self.env)
        return removed


@dataclass
class ModelPack:
    """A model folder: weights, drafter, drafter head, KV profile, tokenizer files (and ``yasakani.json`` listing them)."""
    root: Path
    weights: Path
    draft: Optional[Path] = None
    draft_head: Optional[Path] = None
    kv_profile: Optional[Path] = None
    max_ctx: int = 262144
    model_id: str = "qwen3.8-27b"
    sha256: Dict[str, str] = field(default_factory=dict)
    arch: str = "qwen35"
    vocab_size: int = VOCAB_SIZE
    padded_vocab_size: int = VOCAB_SIZE
    eos_ids: Tuple[int, ...] = EOS_IDS
    sampling_defaults: Dict[str, float] = field(default_factory=dict)    # the model's (generation_config.json); empty: Qwen3.8's

    DEFAULT_FILES = dict(weights="qwen38_27b_e3x.yskn", draft="bdraft2_k8.yskn", draft_head="bdraft2_k8.shead", kv_profile="kvq_u5.bin")

    def load_suppress_tokens(self, paths: Sequence[Path], *, allow_special: bool = False) -> Tuple[int, ...]:
        """Load and validate one or more output-token suppression JSON files against this model pack."""
        return load_suppress_tokens(paths, self, allow_special=allow_special)

    @property
    def is_hf_inkling2(self) -> bool:
        """A folder of the published Inkling-2 safetensors without yasakani.json, run by the engine of #28 inside yskn.dll /
        libyskn.so (a library built with YSKN_WITH_INK2).  Not the 0.3.0 Inkling-2 folders ("architecture": "inkling2" in
        yasakani.json, yasakani/inkling2.py), which never reach this class."""
        return self.arch == HF_INKLING2_ARCH

    @classmethod
    def load(cls, root, *, hf_inkling2: bool = False) -> "ModelPack":
        """``hf_inkling2``: the caller can run a folder of the published Inkling-2 safetensors without yasakani.json (it checks the
        engine library itself, see :func:`hf_inkling2_error`).  Without it such a folder gives the error of 0.3.0, with a hint."""
        root = Path(root)
        if not root.is_dir():
            raise YasakaniError("model folder not found: %s" % root)
        cfg = _hf_inkling2_config(root) if hf_inkling2 else None
        if cfg is not None:
            tcfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
            for need in ("tokenizer.json", "tokenizer_config.json"):
                if not (root / need).is_file():
                    raise YasakaniError("model folder %s is missing %s" % (root, need))
            st_files = sorted(
                p for p in root.glob("*.safetensors")
                if p.is_file() and p.name != "mtp.safetensors"
            )
            if not st_files:
                raise YasakaniError("model folder %s has no model .safetensors file" % root)
            padded_v = int(tcfg.get("vocab_size", 201024))
            vocab_v = int(tcfg.get("unpadded_vocab_size", padded_v))
            max_pos = int(tcfg.get("max_position_embeddings", 131072))
            eos_list: List[int] = []
            gc_file = root / "generation_config.json"
            raw_eos = cfg.get("eos_token_id")
            gc: dict = {}
            if gc_file.is_file():
                try:
                    gc = json.loads(gc_file.read_text(encoding="utf-8"))
                    if not isinstance(gc, dict):
                        gc = {}
                    if "eos_token_id" in gc:
                        raw_eos = gc["eos_token_id"]
                except (OSError, UnicodeDecodeError, ValueError):
                    gc = {}
            if isinstance(raw_eos, int) and not isinstance(raw_eos, bool):
                eos_list.append(int(raw_eos))
            elif isinstance(raw_eos, list):
                eos_list.extend(int(x) for x in raw_eos if isinstance(x, int) and not isinstance(x, bool))
            if not eos_list:
                eos_list = [200006, 199999]
            elif 199999 not in eos_list:
                eos_list.append(199999)
            return cls(
                root=root,
                weights=st_files[0],
                max_ctx=min(max_pos, 131072),
                model_id=root.name or "inkling2-mini",
                arch=HF_INKLING2_ARCH,
                vocab_size=vocab_v,
                padded_vocab_size=padded_v,
                eos_ids=tuple(eos_list),
                sampling_defaults=_model_sampling(gc),          # as the 0.3.0 Inkling-2 path: the model's own values
            )
        files = dict(cls.DEFAULT_FILES)
        meta = {}
        mf = root / "yasakani.json"
        if mf.is_file():
            meta = json.loads(mf.read_text(encoding="utf-8"))
            files.update(meta.get("files") or {})
        paths = {k: root / v for k, v in files.items() if k in cls.DEFAULT_FILES}
        missing = [str(p.name) for p in paths.values() if not p.is_file()]
        if missing:
            raise YasakaniError("model folder %s is missing: %s%s" % (root, ", ".join(missing), _hf_inkling2_hint(root)))
        for need in ("tokenizer.json", "tokenizer_config.json"):
            if not (root / need).is_file():
                raise YasakaniError("model folder %s is missing %s" % (root, need))
        return cls(root=root, weights=paths["weights"], draft=paths["draft"], draft_head=paths["draft_head"],
                   kv_profile=paths["kv_profile"], max_ctx=int(meta.get("max_ctx", 262144)),
                   model_id=str(meta.get("model_id", "qwen3.8-27b")), sha256=dict(meta.get("sha256") or {}))

    def verify(self, progress=None) -> Dict[str, bool]:
        """Compare the sha256 of every listed file with ``yasakani.json`` (reads ~15 GB: a minute or two)."""
        res = {}
        for name, want in sorted(self.sha256.items()):
            p = self.root / name
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    h.update(b)
            res[name] = h.hexdigest() == want
            if progress:
                progress(name, res[name])
        return res


# ---- Inkling-2 folders without yasakani.json (the engine of #28) ---------------------------------------------------------------------
# Two rules, kept apart: yasakani/inkling2.py takes a folder whose yasakani.json says "architecture": "inkling2" (0.3.0, its own
# library yskn_inkling2.dll); the rule here takes a folder that has NO yasakani.json and whose config.json says model_type
# "inkling2_loop" (the published safetensors as downloaded), and only where the caller runs an engine library with that engine.
HF_INKLING2_ARCH = "inkling2_loop"
HF_INKLING2_BUILD_FLAG = "ink2=1"           # in yskn_build_info of a library built with -DYSKN_WITH_INK2=ON


def _hf_inkling2_config(root) -> Optional[dict]:
    """config.json of a folder of the published Inkling-2 safetensors that has no yasakani.json; None for every other folder (also
    when config.json cannot be read: such a folder gets the usual error of a model folder)."""
    root = Path(root)
    if (root / "yasakani.json").exists():
        return None
    try:
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    tcfg = cfg.get("text_config")
    if cfg.get("model_type") == HF_INKLING2_ARCH or (isinstance(tcfg, dict) and tcfg.get("model_type") == "inkling2_loop_text"):
        return cfg
    return None


def is_hf_inkling2_folder(root) -> bool:
    """See :func:`_hf_inkling2_config` (not ``yasakani.inkling2.is_inkling2``, which reads yasakani.json)."""
    return _hf_inkling2_config(root) is not None


def lib_has_hf_inkling2(lib) -> bool:
    """The engine library has the Inkling-2 engine of #28 (built with YSKN_WITH_INK2; the released Windows yskn.dll has not)."""
    try:
        info = lib.build_info()
    except Exception:
        return False
    return HF_INKLING2_BUILD_FLAG in (info or "").split(";")


def _hf_inkling2_hint(root) -> str:
    if not is_hf_inkling2_folder(root):
        return ""
    return ("  (this is a folder of the published Inkling-2 weights without yasakani.json: make a YASAKANI model folder of it with "
            "tools/convert/inkling2_convert.py, or on Linux build the engine library with -DYSKN_WITH_INK2=ON, see BUILDING.md)")


def hf_inkling2_error(root) -> YasakaniError:
    """The error for a folder of the published Inkling-2 weights on an engine library without the engine of #28: the one of 0.3.0
    (the Qwen3.8 files are missing), with the hint."""
    root = Path(root)
    missing = [n for n in ModelPack.DEFAULT_FILES.values() if not (root / n).is_file()]
    return YasakaniError("model folder %s is missing: %s%s" % (root, ", ".join(missing), _hf_inkling2_hint(root)))


def _model_sampling(gc: dict) -> Dict[str, float]:
    """The sampling defaults of generation_config.json, as the 0.3.0 Inkling-2 folders keep them (top_k 0 = off)."""
    def num(k, d, typ):
        v = gc.get(k, d)
        try:
            return typ(v) if v is not None and not isinstance(v, bool) else d
        except (TypeError, ValueError):
            return d
    return dict(temperature=num("temperature", 1.0, float), top_p=num("top_p", 1.0, float), top_k=int(num("top_k", 0, int) or 0))


def suppress_fingerprint(token_ids: Sequence[int]) -> Optional[str]:
    if not token_ids:
        return None
    payload = ",".join(str(i) for i in sorted(set(token_ids))).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:32]


def _tokenizer_metadata(root: Path) -> Tuple[str, Dict[str, int], Set[int]]:
    tok_path = root / "tokenizer.json"
    try:
        raw = tok_path.read_bytes()
    except OSError as exc:
        raise YasakaniError("cannot read tokenizer.json: %s" % exc) from exc
    sha = hashlib.sha256(raw).hexdigest()
    special: Set[int] = set(EOS_IDS) | {IM_START}
    protected_names = frozenset(PROTECTED_CONTROL_TOKENS)
    vocab: Dict[str, int] = {}
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError):
        return sha, vocab, special
    if isinstance(data, dict):
        model = data.get("model")
        if isinstance(model, dict) and isinstance(model.get("vocab"), dict):
            for tok, tid in model["vocab"].items():
                if isinstance(tok, str) and isinstance(tid, int) and not isinstance(tid, bool):
                    vocab[tok] = tid
                    if tok in protected_names:
                        special.add(tid)
        added = data.get("added_tokens")
        if isinstance(added, list):
            for entry in added:
                if not isinstance(entry, dict):
                    continue
                tid = entry.get("id")
                content = entry.get("content")
                if isinstance(tid, int) and not isinstance(tid, bool):
                    if isinstance(content, str):
                        vocab[content] = tid
                    if entry.get("special") or content in protected_names:
                        special.add(tid)
    return sha, vocab, special


def _validate_int_ids(raw: object, path: Path) -> List[int]:
    if not isinstance(raw, list):
        raise YasakaniError("%s: token ID list must be a JSON array" % path)
    out: List[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise YasakaniError("%s: token ID must be an integer, got %r" % (path, item))
        if item < 0 or item >= VOCAB_SIZE:
            raise YasakaniError("%s: token ID %d out of range [0, %d)" % (path, item, VOCAB_SIZE))
        out.append(item)
    return out


def load_suppress_tokens(paths: Sequence[Path], pack: ModelPack, *, allow_special: bool = False) -> Tuple[int, ...]:
    """Read one or more suppression JSON files (ID array, target_token_ids/suppress_tokens/token_ids, or exact tokens)."""
    if not paths:
        return ()
    tok_sha, vocab, special_ids = _tokenizer_metadata(pack.root)
    merged: Set[int] = set()
    for p in paths:
        path = Path(p)
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise YasakaniError("cannot read suppression mask %s: %s" % (path, exc)) from exc
        if isinstance(data, list):
            merged.update(_validate_int_ids(data, path))
            continue
        if not isinstance(data, dict):
            raise YasakaniError("%s: expected a JSON array or object" % path)
        if "vocab_size" in data:
            vs = data["vocab_size"]
            if isinstance(vs, bool) or vs != VOCAB_SIZE:
                raise YasakaniError("%s: vocab_size %r does not match %d" % (path, vs, VOCAB_SIZE))
        if "tokenizer_sha256" in data:
            want = data["tokenizer_sha256"]
            allowed = [want] if isinstance(want, str) else (want if isinstance(want, list) and all(isinstance(s, str) for s in want) else None)
            if allowed is None or tok_sha not in allowed:
                raise YasakaniError("%s: tokenizer.json SHA-256 (%s) does not match expected %r" % (path, tok_sha, want))
        found = False
        for key in ("target_token_ids", "suppress_tokens", "token_ids"):
            if key in data:
                ids = _validate_int_ids(data[key], path)
                if key == "target_token_ids" and "n_target" in data:
                    nt = data["n_target"]
                    if isinstance(nt, bool) or not isinstance(nt, int) or len(ids) != nt:
                        raise YasakaniError("%s: n_target (%r) does not match target_token_ids length (%d)" % (path, nt, len(ids)))
                merged.update(ids)
                found = True
        if "tokens" in data:
            raw_toks = data["tokens"]
            if not isinstance(raw_toks, list) or not all(isinstance(t, str) for t in raw_toks):
                raise YasakaniError("%s: 'tokens' must be a JSON array of strings" % path)
            if not vocab:
                raise YasakaniError("%s: 'tokens' requires a valid tokenizer.json vocabulary" % path)
            for tok in raw_toks:
                if tok not in vocab:
                    raise YasakaniError("%s: token %r not found as a single token in tokenizer.json" % (path, tok))
                tid = vocab[tok]
                if tid < 0 or tid >= VOCAB_SIZE:
                    raise YasakaniError("%s: token %r maps to out-of-range ID %d" % (path, tok, tid))
                merged.add(tid)
            found = True
        if not found:
            raise YasakaniError(
                "%s: expected a JSON array of IDs or an object with 'target_token_ids', 'suppress_tokens', 'token_ids', or 'tokens'" % path
            )
    remaining = VOCAB_SIZE - len(merged)
    if remaining < MIN_UNSUPPRESSED_TOKENS:
        raise YasakaniError(
            "suppression mask must leave at least %d unmasked token IDs (got %d)" % (MIN_UNSUPPRESSED_TOKENS, remaining)
        )
    if not allow_special:
        blocked = sorted(merged & special_ids)
        if blocked:
            sample = ", ".join(str(i) for i in blocked[:5])
            extra = " (+%d more)" % (len(blocked) - 5) if len(blocked) > 5 else ""
            raise YasakaniError(
                "suppression mask includes special/control/EOS token ID(s) %s%s; pass --allow-suppress-special to override" % (sample, extra)
            )
    return tuple(sorted(merged))
