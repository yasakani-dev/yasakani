# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 the YASAKANI authors
"""Proposals and costs of copy decoding for edits (Inkling-2 binding; used by _ink2_copyedit, off by default).

Proposals: a 4-gram index over the ids the engine has seen (the binding's seen_ids) and the anchor (N = 4): the nearest earlier
place at or after the point where the last copy stopped, unless its backward match is shorter than both 8 and the longest one;
else the place with the longest backward match (up to 64 ids), the newest on ties; at most 256 candidates.  First the copy
cursor: while the output follows a source, the next proposal goes on from there.  The index keeps exact 4-gram keys (no hash
buckets), so a candidate never fails the 4-id check.

Image positions: the binding writes a negative number unique to the position into seen_ids for each image position (an image
vector has no token id).  Such an entry never equals another entry, so no 4-gram window and no backward match spans it, and a
proposed chain stops before it (``text_chain``): ids are never proposed for image positions and image positions never match text.

Costs (CopyCtl): copied ids per pass W from the ladder maximising expected tokens per ms, 1 + q1 (1 - qn^W) / (1 - qn) over the
pass time of W + 1 rows, against the measured ms per token of the other steps (drafter or plain) times a margin.  q1: share of
first proposed ids the output followed (by backward-match length 4-7 / 8-15 / 16-31 / 32-64; a cursor continuation uses qn),
qn: share of the later ones; both learnt from every step, also from the steps that did not copy (the output is compared with
the proposal it had), so copying starts and stops without probe passes.  Pass times: measured per row count (the first pass of a
size records its graph and is not counted), before that prior costs scaled by the measured plain step.  While the drafter
drafts, only W >= MIN_W_DRAFTER copies: a shorter copy pass competes with a drafter pass of about the same rows.
"""

import time

N = 4
MAX_CAND, MAX_BACK = 256, 64        # candidates looked at, longest backward match
AFTER_SPAN = 256                    # `after` only while the last stop is at most this far behind the frontier
LADDER = (1, 3, 7, 15, 23, 31)      # copied ids per pass: R = 2, 4, 8, 16, 24, 32 (one graph each)
MAX_W = 31
MIN_W_DRAFTER = 15                  # with the drafter drafting: copy passes of 16 rows or more only
# prior pass costs, converted folder, RTX 5090 (ms): R = 1 .. 8 measured; above 8 rows +0.9 ms per row.
PRIOR_MS = {1: 4.52, 2: 6.08, 3: 7.61, 4: 8.94, 5: 9.86, 6: 10.84, 7: 11.76, 8: 13.00}
Q_FIRST_PRIOR = (0.35, 0.5, 0.65, 0.8)
Q_NEXT_PRIOR = 0.85


def prior_ms(R):
    return PRIOR_MS[R] if R in PRIOR_MS else PRIOR_MS[8] + 0.9 * (R - 8)


def text_chain(chain):
    """The chain up to its first image position (a negative entry of seen_ids)."""
    for j, t in enumerate(chain):
        if t < 0:
            return chain[:j]
    return chain


def bucket_of(length):
    return 0 if length < 8 else (1 if length < 16 else (2 if length < 32 else 3))


class CopyIndex:
    """Exact 4-gram index of S[0 .. upto - 1]: key (S[p-3], .., S[p]) -> end positions p, ascending (newest last)."""

    def __init__(self):
        self.grams = {}
        self.upto = 0

    def clear(self):
        self.grams = {}
        self.upto = 0

    def index_upto(self, seen, end):
        g = self.grams
        p = self.upto
        if p < N - 1:
            p = min(N - 1, end)
        while p < end:
            key = (seen[p - 3], seen[p - 2], seen[p - 1], seen[p])
            lst = g.get(key)
            if lst is None:
                g[key] = [p]
            else:
                lst.append(p)
            p += 1
        if end > self.upto:
            self.upto = end

    def drop_from(self, seen, n):
        """Forget the windows ending at positions >= n (exactly; called before seen loses them)."""
        g = self.grams
        p = self.upto - 1
        while p >= n and p >= N - 1:
            key = (seen[p - 3], seen[p - 2], seen[p - 1], seen[p])
            lst = g.get(key)
            if lst and lst[-1] == p:
                lst.pop()
                if not lst:
                    del g[key]
            else:                                   # out of step (should not happen): start again
                self.clear()
                return
            p -= 1
        self.upto = min(self.upto, n)

    def find(self, seen, anchor, F, after, max_cand=MAX_CAND, max_back=MAX_BACK):
        """The source place for S = seen[:F] + [anchor] (S[F] = anchor) -> (p, backward match length) or (-1, 0)."""
        if F < N - 1:
            return -1, 0
        lst = self.grams.get((seen[F - 3], seen[F - 2], seen[F - 1], anchor))
        if not lst:
            return -1, 0
        best, best_len, cand, cand_len, c = -1, 0, -1, 0, 0
        for i in range(len(lst) - 1, -1, -1):        # newest first
            if c >= max_cand:
                break
            p = lst[i]
            if p >= F:
                continue
            l = N
            while l < max_back and p - l >= 0 and seen[p - l] == seen[F - l]:
                l += 1
            c += 1
            if l > best_len:                         # strict >: the newest on ties
                best, best_len = p, l
            if after >= 0 and p >= after:            # walking down: the last one seen is the nearest at or after `after`
                cand, cand_len = p, l
        if cand >= 0 and cand_len >= min(best_len, 8):
            return cand, cand_len
        return best, best_len


class CopyCtl:
    """When to copy and how many ids (see the module text)."""

    def __init__(self, margin=1.05, beta_first=0.15, beta_next=0.05, beta_t=0.2, beta_alt=0.1):
        self.margin = margin
        self.beta_first, self.beta_next, self.beta_t, self.beta_alt = beta_first, beta_next, beta_t, beta_alt
        self.q_first = list(Q_FIRST_PRIOR)
        self.q_next = Q_NEXT_PRIOR
        self.t = {}                 # R -> ms (EWMA of measured copy passes)
        self.n = {}                 # R -> passes seen (the first records the graph)
        self.t_alt = None           # ms per emitted token of the other steps
        self.t_plain = None         # ms of a one-token step (scales the prior pass costs)
        self.n_other = 0

    @staticmethod
    def _clamp(q):
        return 0.02 if q < 0.02 else (0.995 if q > 0.995 else q)

    def pass_ms(self, R):
        v = self.t.get(R)
        if v is not None:
            return v
        s = (self.t_plain / PRIOR_MS[1]) if self.t_plain else 1.0
        return prior_ms(R) * s

    def alt_ms(self):
        if self.t_alt is not None:
            return self.t_alt
        return self.t_plain if self.t_plain else PRIOR_MS[1]

    def choose(self, n_chain, bucket, cursor, w_min=1):
        """-> W (0 = no copy pass); only W >= w_min."""
        q1 = self.q_next if cursor else self.q_first[bucket]
        qn = self.q_next
        best_w, best_rate = 0, self.margin / self.alt_ms()
        geo, run, j = 0.0, 1.0, 0
        for W in LADDER:
            if W > n_chain:
                break
            while j < W:                             # geo = 1 + qn + .. + qn^(W-1)
                geo += run
                run *= qn
                j += 1
            if W < w_min:
                continue
            rate = (1.0 + q1 * geo) / self.pass_ms(W + 1)
            if rate > best_rate:
                best_w, best_rate = W, rate
        return best_w

    def learn(self, matched, trials, bucket, cursor):
        """`trials` ids compared with the proposal, the first `matched` equal (matched < trials: the next one differed)."""
        if trials <= 0:
            return
        ok1 = matched >= 1
        if cursor:
            self.q_next = self._clamp(self.q_next + self.beta_next * ((1.0 if ok1 else 0.0) - self.q_next))
        else:
            q = self.q_first[bucket]
            self.q_first[bucket] = self._clamp(q + self.beta_first * ((1.0 if ok1 else 0.0) - q))
        if not ok1:
            return
        b, q = self.beta_next, self.q_next
        for j in range(1, trials):                   # later ids, while the earlier ones were followed
            ok = j < matched
            q += b * ((1.0 if ok else 0.0) - q)
            if not ok:
                break
        self.q_next = self._clamp(q)

    def time_pass(self, R, ms):
        k = self.n.get(R, 0) + 1
        self.n[R] = k
        if k == 1 and R not in self.t:
            return                                    # the first pass of this size records its graph
        v = self.t.get(R)
        self.t[R] = ms if v is None else v + self.beta_t * (ms - v)

    def time_other(self, ms, tokens, drafted):
        """A step of the other path (drafter pass or plain step) that took ms and emitted tokens."""
        self.n_other += 1
        if self.n_other == 1:
            return                                    # the first step after a (re)load may record its graph
        if not drafted and (self.t_plain is None or ms < 3.0 * self.t_plain):
            self.t_plain = ms if self.t_plain is None else self.t_plain + self.beta_alt * (ms - self.t_plain)
        per = ms / max(1, tokens)
        if self.t_alt is not None and per > 3.0 * self.t_alt:
            return                                    # a graph being recorded, or a stall: not a cost of the other path
        self.t_alt = per if self.t_alt is None else self.t_alt + self.beta_alt * (per - self.t_alt)


class CopySpec:
    """Index, cursor and controller of one engine sequence (S = seen_ids + [anchor])."""

    def __init__(self, ctl=None):
        self.ix = CopyIndex()
        self.ctl = ctl or CopyCtl()
        self.cur, self.last_src, self.last_src_at = -1, -1, -1
        self.stats = new_stats()

    def reset(self):
        self.ix.clear()
        self.cur, self.last_src, self.last_src_at = -1, -1, -1

    def truncate(self, seen, n):
        if n < self.ix.upto:
            self.ix.drop_from(seen, n)
        if self.cur > n:
            self.cur = -1
        if self.last_src >= n or self.last_src_at >= n:
            self.last_src, self.last_src_at = -1, -1

    def lookup(self, seen, anchor):
        """The proposal: cursor first, else the index -> (src, chain, backward match length, cursor) or None; chain = S[src .. F][:MAX_W]."""
        F = len(seen)
        p, ln, cursor = -1, 0, False
        c = self.cur
        if 0 < c <= F and seen[c - 1] == anchor:
            q = c - 1
            l = 1
            while l < 8 and q - l >= 0 and seen[q - l] == seen[F - l]:
                l += 1
            if l >= N:
                p, ln, cursor = q, l, True
        if p < 0:
            self.ix.index_upto(seen, F)
            after = self.last_src if (self.last_src >= 0 and F - self.last_src_at <= AFTER_SPAN) else -1
            p, ln = self.ix.find(seen, anchor, F, after)
            if p < 0 or ln < N:
                return None
        end = min(F, p + MAX_W)                      # S[F] is the anchor
        chain = text_chain(seen[p + 1:end + 1] if end < F else seen[p + 1:F] + [anchor])
        if not chain:
            return None
        return p + 1, chain, ln, cursor

    def follow(self, src, seen_after, nxt, F_old, ne):
        """The cursor after a step that emitted ne ids (the last one, nxt, not fed)."""
        if src < 0:
            self.cur = -1
            return
        L = len(seen_after)
        k = 0
        while k < ne:
            i, j = src + k, F_old + 1 + k
            a = seen_after[i] if i < L else (nxt if i == L else None)
            b = seen_after[j] if j < L else nxt
            if a is None or a != b:
                break
            k += 1
        if k == ne:
            self.cur = src + ne
        else:
            self.cur, self.last_src, self.last_src_at = -1, src + k, F_old + 1 + k


def new_stats():
    return {"steps": 0, "proposals": 0, "cursor": 0, "copy_passes": 0, "copy_rows": 0, "copy_accepted": 0, "copy_tokens": 0,
            "copy_ms": 0.0, "other_steps": 0, "other_tokens": 0, "other_ms": 0.0, "host_ms": 0.0, "shadow_trials": 0,
            "shadow_matched": 0, "w_hist": {}}


# ---- the hook on the binding ------------------------------------------------------------------------------------------------------
def _drafting_now(self):
    """The drafter drafts at this step: loaded, and mode "on", or "auto" with its governor on (probes aside)."""
    mode = getattr(self, "_d_mode", None)
    if mode in (None, "off") or not self.draft_loaded():
        return False
    if mode == "on":
        return True
    return bool(self._cgov.active if getattr(self, "_cut", None) else self._gov.active)
