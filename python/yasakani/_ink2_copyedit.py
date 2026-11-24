# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Copy decoding as the main path for edits (Inkling-2 binding; greedy and seeded sampling; opt-in, "off" leaves the engine as it was).

    from yasakani import _ink2_copyedit as ce
    ce.enable(native, "auto", min_match=8)   # "off": the class spec_step again, nothing of this module on the instance
    accepted, nxt, _ = native.spec_step(anchor, pick)       # unchanged call
    ce.stats(native, reset=False)

Proposals: the 4-gram index and copy cursor of _ink2_copy, with chains up to the engine's row limit (max_step_rows - 1 copied
ids per pass; 32 rows when the flash prompt attention is on, so every copy pass stays on the kernels single steps use).

A copy pass feeds the anchor and W proposed ids in one ink2_step_many pass and keeps the longest prefix the main model itself
emits; the rest is truncated (inside the rollback).  Greedy (pick None): row r's arg-max (on the device) is compared with proposal
r.  Sampling (pick given, ink2 contract of the drafter path): row r's logits are read (ink2_get_row) and given to pick, one call per
emitted token in output order (pick seeks the seed's draw k + r), and the drawn token is kept while it equals proposal r; the first
draw that differs is the next token.  Every emitted token is the main model's own choice from the logits a plain step gives (a
pass gives every row the arithmetic of a single step), drawn with the same random number, so the output equals copy off.

When to copy (rule of this module): a proposal is a candidate when its backward match has at least `min_match` ids (a fresh
lookup, or the copy cursor while the output follows the source: its run measured up to max(8, min_match, LONG_MATCH) ids).  Then the cost model of _ink2_copy picks W from the ladder
(expected tokens per ms of a W + 1 row pass against the measured ms per token of the other steps):
  - long match (backward match >= LONG_MATCH ids): copy is the first choice, any W, margin MARGIN_LONG; the
    drafter is not called for that pass;
  - shorter match while the drafter drafts: only W >= MIN_W_DRAFTER (a short copy competes with a drafter pass of the same rows);
  - shorter match without drafting: any W, margin MARGIN.
Greedy and sampled steps learn separate acceptance rates (sampling at temperature 1.0 follows a copy less often).
"""

import os
import time
import types

from . import _ink2_copy as base

ENV = "YASAKANI_INK2_COPY"              # off | auto
MIN_ENV = "YASAKANI_INK2_COPY_MIN"      # min_match, 4 .. 64
MODES = ("off", "auto")
DEFAULT_MODE = "off"
DEFAULT_MIN = 8
LONG_MATCH = 16
FLASH_SAFE_ROWS = 32                    # with the flash prompt attention on, passes with logits of up to 32 rows run the kernel
                                        # of single steps (the engine's pass_exact rule); wider passes may run the tensor-core
                                        # kernel, whose numbers differ from single steps: copy stays within the 32
LONG_W_DRAFTING = 1                     # smallest copy pass for a long match while the drafter drafts
MIN_W_DRAFTER = base.MIN_W_DRAFTER      # 15
MARGIN, MARGIN_LONG = 1.05, 1.0


def ladder_for(max_w):
    """Copied ids per pass: the ladder of _ink2_copy, then every 8 up to max_w (one recorded graph per size)."""
    out = [w for w in base.LADDER if w <= max_w]
    w = base.LADDER[-1] + 8
    while w <= max_w:
        out.append(w)
        w += 8
    if max_w >= 1 and out and max_w > out[-1] and (max_w < base.LADDER[-1] or max_w - out[-1] >= 4):
        out.append(max_w)
    return tuple(out)


class Ctl(base.CopyCtl):
    def __init__(self, ladder, **kw):
        super().__init__(**kw)
        self.ladder = tuple(ladder)

    def choose(self, n_chain, bucket, cursor, w_min=1, margin=None):
        """-> W (0 = no copy pass) from self.ladder, W >= w_min, W <= n_chain."""
        q1 = self.q_next if cursor else self.q_first[bucket]
        qn = self.q_next
        m = self.margin if margin is None else margin
        best_w, best_rate = 0, m / self.alt_ms()
        geo, run, j = 0.0, 1.0, 0
        for W in self.ladder:
            if W > n_chain:
                break
            while j < W:
                geo += run
                run *= qn
                j += 1
            if W < w_min:
                continue
            rate = (1.0 + q1 * geo) / self.pass_ms(W + 1)
            if rate > best_rate:
                best_w, best_rate = W, rate
        return best_w


def new_stats():
    s = base.new_stats()
    s.update(sampled_steps=0, sampled_copy_passes=0, sampled_copy_accepted=0, long_copy_passes=0, skipped_short=0)
    return s


class Spec(base.CopySpec):
    """Index, cursor and two controllers (greedy, sampled) of one engine sequence."""

    def __init__(self, max_w, min_match=DEFAULT_MIN):
        lad = ladder_for(max_w)
        super().__init__(Ctl(lad, margin=MARGIN))
        self.ctl_s = Ctl(lad, margin=MARGIN)
        self.max_w = int(max_w)
        self.min_match = int(min_match)
        self.stats = new_stats()

    def lookup(self, seen, anchor):
        """_ink2_copy.CopySpec.lookup with chains of up to self.max_w ids."""
        F = len(seen)
        p, ln, cursor = -1, 0, False
        c = self.cur
        if 0 < c <= F and seen[c - 1] == anchor:
            q = c - 1
            l = 1
            cap = max(8, self.min_match, LONG_MATCH)      # the run the output has followed, up to what the rule asks
            while l < cap and q - l >= 0 and seen[q - l] == seen[F - l]:
                l += 1
            if l >= base.N:
                p, ln, cursor = q, l, True
        if p < 0:
            self.ix.index_upto(seen, F)
            after = self.last_src if (self.last_src >= 0 and F - self.last_src_at <= base.AFTER_SPAN) else -1
            p, ln = self.ix.find(seen, anchor, F, after)
            if p < 0 or ln < base.N:
                return None
        end = min(F, p + self.max_w)
        chain = base.text_chain(seen[p + 1:end + 1] if end < F else seen[p + 1:F] + [anchor])
        if not chain:
            return None
        return p + 1, chain, ln, cursor


def _copy_pass(self, anchor, chain, W, pick):
    """One copy pass of the anchor and chain[:W] -> (accepted count, next token).  The positions after the kept ones are dropped."""
    F = len(self.seen_ids)
    ids = [anchor] + chain[:W]
    _, top, _ = self.step_many(ids, logits=False)
    acc = 0
    if pick is None:
        while acc < W and int(top[acc]) == chain[acc]:
            acc += 1
        nxt = int(top[acc])
    else:
        while True:
            y = int(pick(self._row_logits(acc)))      # one draw per emitted token, in output order
            if acc < W and y == chain[acc]:
                acc += 1
                continue
            nxt = y
            break
    if acc < W:
        self.truncate(F + 1 + acc)
    return acc, nxt


def _spec_step(self, anchor, pick=None, want_logits=False):
    cp = self.__dict__.get("_copy")
    orig = type(self).spec_step
    if cp is None or want_logits:
        return orig(self, anchor, pick, want_logits)
    anchor = int(anchor)
    sampled = pick is not None
    ctl = cp.ctl_s if sampled else cp.ctl
    st = cp.stats
    st["steps"] += 1
    st["sampled_steps"] += 1 if sampled else 0
    th = time.perf_counter()
    seen = self.seen_ids
    F = len(seen)
    prop = cp.lookup(seen, anchor)
    W, long_ = 0, False
    if prop is not None:
        src, chain, ln, cursor = prop
        bucket = base.bucket_of(ln)
        st["proposals"] += 1
        st["cursor"] += 1 if cursor else 0
        if ln >= cp.min_match:                         # fresh lookup or cursor: a match of at least min_match ids
            long_ = ln >= LONG_MATCH
            count = min(len(chain), self.max_step_rows - 1, self.rollback)
            if long_:
                W = ctl.choose(count, bucket, cursor, LONG_W_DRAFTING if base._drafting_now(self) else 1, MARGIN_LONG)
            else:
                W = ctl.choose(count, bucket, cursor, MIN_W_DRAFTER if base._drafting_now(self) else 1, MARGIN)
            if W and F + W + 1 > self._copy_maxpos:
                W = 0
        else:
            st["skipped_short"] += 1
    if W:
        host = time.perf_counter() - th
        t0 = time.perf_counter()
        acc, nxt = _copy_pass(self, anchor, chain, W, pick)
        dt = time.perf_counter() - t0
        th = time.perf_counter()
        ctl.time_pass(W + 1, 1e3 * dt)
        ctl.learn(acc, W, bucket, cursor)
        cp.follow(src, self.seen_ids, nxt, F, acc + 1)
        st["copy_passes"] += 1
        st["long_copy_passes"] += 1 if long_ else 0
        st["sampled_copy_passes"] += 1 if sampled else 0
        st["sampled_copy_accepted"] += acc if sampled else 0
        st["copy_rows"] += W
        st["copy_accepted"] += acc
        st["copy_tokens"] += acc + 1
        st["copy_ms"] += 1e3 * dt
        st["w_hist"][W] = st["w_hist"].get(W, 0) + 1
        ss = getattr(self, "_sstats", None)
        if ss is not None:
            ss["passes"] += 1
            ss["tokens"] += 1 + acc
            ss["copy_passes"] = ss.get("copy_passes", 0) + 1
            ss["copy_tokens"] = ss.get("copy_tokens", 0) + 1 + acc
        st["host_ms"] += 1e3 * (host + time.perf_counter() - th)
        return chain[:acc], nxt, None
    ss = getattr(self, "_sstats", None)
    a0 = ss["active_passes"] if ss is not None else 0
    host = time.perf_counter() - th
    t0 = time.perf_counter()
    accepted, nxt, lg = orig(self, anchor, pick, False)
    dt = time.perf_counter() - t0
    th = time.perf_counter()
    ne = len(accepted) + 1
    ss = getattr(self, "_sstats", None)
    ctl.time_other(1e3 * dt, ne, ss is not None and ss["active_passes"] > a0)
    st["other_steps"] += 1
    st["other_tokens"] += ne
    st["other_ms"] += 1e3 * dt
    if prop is not None:
        emitted = list(accepted) + [nxt]
        n = min(ne, len(chain))
        m = 0
        while m < n and emitted[m] == chain[m]:
            m += 1
        ctl.learn(m, n if m == n else m + 1, bucket, cursor)
        st["shadow_trials"] += n if m == n else m + 1
        st["shadow_matched"] += m
        cp.follow(src, self.seen_ids, nxt, F, ne)
    else:
        cp.cur = -1
    st["host_ms"] += 1e3 * (host + time.perf_counter() - th)
    return accepted, nxt, lg


def _install(cls):
    """Keeps the index in step with reset / truncate (once per class)."""
    if getattr(cls, "_copyedit_hooks", False):
        return
    orig_reset, orig_truncate = cls.reset, cls.truncate

    def reset(self):
        cp = self.__dict__.get("_copy")
        if cp is not None:
            cp.reset()
        return orig_reset(self)

    def truncate(self, n):
        cp = self.__dict__.get("_copy")
        if cp is not None and 0 <= int(n) <= len(self.seen_ids):
            cp.truncate(self.seen_ids, int(n))
        return orig_truncate(self, n)

    reset.__doc__, truncate.__doc__ = orig_reset.__doc__, orig_truncate.__doc__
    cls.reset, cls.truncate = reset, truncate
    cls._copyedit_hooks = True


def mode_of(value=None, environ=None):
    import os
    env = os.environ if environ is None else environ
    v = value if value not in (None, "") else env.get(ENV, "")
    v = (v or DEFAULT_MODE).strip().lower()
    if v not in MODES:
        raise ValueError("%s / --copy must be one of %s, not %r" % (ENV, ", ".join(MODES), v))
    return v


def min_of(value=None, environ=None):
    import os
    env = os.environ if environ is None else environ
    v = value if value not in (None, "") else (env.get(MIN_ENV) or "").strip()
    if v in (None, ""):
        return DEFAULT_MIN
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = 0
    if not base.N <= n <= base.MAX_BACK:
        raise ValueError("%s / --copy-min must be a number from %d to %d, not %r" % (MIN_ENV, base.N, base.MAX_BACK, v))
    return n


def max_rows_of(native):
    """Rows of the widest copy pass: 32 when the flash prompt attention is on (``native.flash_attn``, set by the package), so every
    copy pass stays on the kernels single steps use, else no limit (the engine's own)."""
    return FLASH_SAFE_ROWS if getattr(native, "flash_attn", 0) else 1 << 20


def enable(native, mode, min_match=DEFAULT_MIN):
    """mode "auto": copy passes on this engine instance; "off": the instance's spec_step is the class method again."""
    mode = str(mode).strip().lower()
    if mode not in MODES:
        raise ValueError("copy mode must be auto or off")
    if mode == "off":
        native.__dict__.pop("_copy", None)
        native.__dict__.pop("spec_step", None)
        return {"mode": "off"}
    if int(getattr(native, "abi", 0)) < 2 or native.max_step_rows < 2:
        raise RuntimeError("copy decoding needs passes of several rows (DLL version 2)")
    _install(type(native))
    native._copy = Spec(max(1, min(native.max_step_rows - 1, native.rollback, max_rows_of(native) - 1)), min_match)
    native._copy_maxpos = int(native.lib_max_positions())
    native.spec_step = types.MethodType(_spec_step, native)
    return {"mode": "auto", "min_match": int(min_match), "ladder": native._copy.ctl.ladder}


def stats(native, reset=False):
    cp = native.__dict__.get("_copy")
    if not isinstance(cp, Spec):
        return {}
    out = dict(cp.stats)
    out["w_hist"] = dict(cp.stats["w_hist"])
    for key, c in (("ctl", cp.ctl), ("ctl_sampled", cp.ctl_s)):
        out[key] = {"q_first": [round(x, 4) for x in c.q_first], "q_next": round(c.q_next, 4), "t_alt": c.t_alt,
                    "t_plain": c.t_plain, "pass_ms": {r: round(v, 3) for r, v in sorted(c.t.items())}}
    if reset:
        cp.stats = new_stats()
    return out


def calibrate(native, reps=2):
    """Measures the pass sizes of the ladder on the current state (unchanged afterwards) for both controllers -> {R: ms}."""
    cp = native.__dict__.get("_copy")
    if not isinstance(cp, Spec):
        raise RuntimeError("enable(native, 'auto') first")
    if native.position() < 1:
        raise RuntimeError("calibrate needs a position seen")
    pos = native.position()
    out = {}
    for W in cp.ctl.ladder:
        R = W + 1
        if R > native.max_step_rows or pos + R > native._copy_maxpos:
            continue
        ts = []
        for k in range(reps + 1):
            ids = ([t if t >= 0 else 0 for t in native.seen_ids[-R:]] * R)[:R]   # image positions: any text id (timing only)
            t0 = time.perf_counter()
            native.step_many(ids, logits=False)
            native.truncate(pos)
            if k:
                ts.append(1e3 * (time.perf_counter() - t0))
        ts.sort()
        out[R] = ts[len(ts) // 2]
    for c in (cp.ctl, cp.ctl_s):
        c.t.update(out)
        c.n.update({r: max(c.n.get(r, 0), 1) for r in out})
    return out
