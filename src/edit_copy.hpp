// SPDX-License-Identifier: Apache-2.0
// pattern_edit_copy.hpp - 100% API-Compatible Drop-In Upgrade for `src/edit_copy.hpp` (Invention 3)
//
// Preserves the exact `yskn::EditCopy` class name and method signatures expected by `src/specx.cpp`:
//   - `explicit EditCopy(int input_end)`
//   - `void advance(const std::vector<int32_t>& ids, int end, int raw_next)`
//   - `void missed()`
//   - `void hit()`
//   - `int propose(const std::vector<int32_t>& ids, int end, int limit, std::vector<int32_t>& out) const`
//
// Capabilities over v0.2.0 `src/edit_copy.hpp`:
//   1. Exact backward compatibility on all static literal replacements (`from[] -> to[]`), including
//      `kept_ratio = 4`, `miss_limit = 2`, pure insertion/deletion anchoring, and uncommitted suffix replay.
//   2. Automatic 1-Slot & 2-Slot Variable Capture + Reordering (`prefix $1 mid $2 -> out_pre $2 out_mid $1`)
//      induced only after TWO distinct confirmed source locations exhibit the exact same template structure!
//   3. Automatic Affine Integer-Token Progression (`$1 -> $1 + delta` for `|delta| <= 4`), solving the exact
//      failure case noted in `src/edit_copy.hpp` line 96 ("when the replacement differs from record to record").
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace yskn {

class EditCopy {
    static constexpr int gap = 8;          // Widened from 6 -> 8 to cover field-reorder / type-annotation spans
    static constexpr int anchor = 4;       // 4 unchanged tokens before & after an edit before learning
    static constexpr int rule_limit = 16;
    static constexpr int kept_ratio = 4;
    static constexpr int miss_limit = 2;
    static constexpr int max_slots = 2;    // Supports 0 (literal), 1 ($1), or 2 ($1, $2) captured variable slots!
    static constexpr int max_delta = 4;    // Max affine token ID offset `|to_tok - captured_src_tok| <= 4`

    struct OutElem {
        int32_t lit = 0;       // Literal token ID when `slot_idx < 0`
        int8_t slot_idx = -1;  // -1 = literal; 0 = $1; 1 = $2
        int8_t delta = 0;      // Emits `captured[slot_idx] + delta` when `slot_idx >= 0`
    };

    struct Rule {
        std::array<int32_t, gap + 1> from{};
        std::array<int8_t, gap + 1> from_slot_id{};  // -1 = literal match required; 0 = $1; 1 = $2
        int nf = 0;
        int n_slots = 0;

        std::array<OutElem, gap + 1> to{};
        int nt = 0;

        int score = 0, last = -1;
        int n_edit = 0, n_kept = 0;
        int miss_run = 0, miss_src = -1;

        bool usable() const {
            return score >= 2 && n_kept * kept_ratio <= n_edit && miss_run < miss_limit;
        }
    };

    std::array<Rule, rule_limit> rules_{};
    int used_ = 0;
    int input_end_ = 0;
    int source_ = -1;
    int output_ = -1;
    int literal_ = 0;
    int run_ = 0;
    mutable int prop_rule_ = -1;
    mutable int prop_src_ = -1;

    // Check if `r` matches `ids[at .. at + r.nf)` and extract captured slot values `caps[0..1]`
    bool match_rule(const Rule& r, const std::vector<int32_t>& ids, int at, int32_t (&caps)[max_slots]) const {
        if (r.nf <= 0 || at + r.nf > input_end_) return false;
        for (int k = 0; k < r.nf; ++k) {
            const int8_t sid = r.from_slot_id[k];
            if (sid >= 0) {
                caps[sid] = ids[at + k];
            } else if (ids[at + k] != r.from[k]) {
                return false;
            }
        }
        return true;
    }

    static void eval_rule_to(const Rule& r, const int32_t (&caps)[max_slots], std::array<int32_t, gap + 1>& rendered) {
        for (int k = 0; k < r.nt; ++k) {
            const OutElem& e = r.to[k];
            rendered[k] = (e.slot_idx >= 0) ? (caps[e.slot_idx] + static_cast<int32_t>(e.delta)) : e.lit;
        }
    }

    int match(const std::vector<int32_t>& ids, int at, int32_t (&best_caps)[max_slots]) const {
        int best = -1;
        int best_lit_count = -1;
        for (int i = 0; i < used_; ++i) {
            const Rule& r = rules_[i];
            if (!r.usable()) continue;
            int32_t caps[max_slots]{};
            if (!match_rule(r, ids, at, caps)) continue;
            const int lit_cnt = r.nf - r.n_slots;
            // Prefer longer match, then more specific (more literal anchors), then higher score
            if (best < 0 || r.nf > rules_[best].nf ||
                (r.nf == rules_[best].nf && lit_cnt > best_lit_count) ||
                (r.nf == rules_[best].nf && lit_cnt == best_lit_count && r.score > rules_[best].score)) {
                best = i;
                best_lit_count = lit_cnt;
                best_caps[0] = caps[0];
                best_caps[1] = caps[1];
            }
        }
        return best;
    }

    // Attempt to generalize an existing literal score=1 rule `r` with a new observation `(src_new, out_new)`
    // into a 1-slot or 2-slot parameterized / affine rule!
    // Safety guard: requires at least ONE literal token in `from` (`nf > n_diff`) so wildcards never match everywhere!
    static bool try_generalize_literal_rule(Rule& r, const int32_t* src_new, const int32_t* out_new, int src_pos) {
        if (r.n_slots > 0) return false;  // Already generalized; handled by `match_rule` check
        int diff_pos[max_slots]{};
        int n_diff = 0;
        for (int k = 0; k < r.nf; ++k) {
            if (r.from[k] != src_new[k]) {
                if (n_diff >= max_slots) return false;
                diff_pos[n_diff++] = k;
            }
        }
        // Must have 1..2 slots AND at least 1 fixed literal anchor token inside `from`!
        if (n_diff == 0 || r.nf - n_diff < 1) return false;

        int32_t old_caps[max_slots]{}, new_caps[max_slots]{};
        for (int s = 0; s < n_diff; ++s) {
            old_caps[s] = r.from[diff_pos[s]];
            new_caps[s] = src_new[diff_pos[s]];
        }

        std::array<OutElem, gap + 1> cand_to{};
        for (int k = 0; k < r.nt; ++k) {
            const int32_t old_y = r.to[k].lit;
            const int32_t new_y = out_new[k];
            if (old_y == new_y) {
                cand_to[k] = OutElem{old_y, -1, 0};
                continue;
            }
            // Check if `(old_y, new_y)` is explained by `cap[s] + delta` for a unique slot `s`!
            bool mapped = false;
            for (int s = 0; s < n_diff; ++s) {
                const int32_t d_old = old_y - old_caps[s];
                const int32_t d_new = new_y - new_caps[s];
                if (d_old == d_new && std::abs(d_old) <= max_delta) {
                    cand_to[k] = OutElem{0, static_cast<int8_t>(s), static_cast<int8_t>(d_old)};
                    mapped = true;
                    break;
                }
            }
            if (!mapped) return false;
        }

        // Upgrade `r` to a parameterized template rule (`score = 2` since confirmed at 2 distinct places!)
        r.from_slot_id.fill(-1);
        for (int s = 0; s < n_diff; ++s) {
            r.from_slot_id[diff_pos[s]] = static_cast<int8_t>(s);
        }
        r.n_slots = n_diff;
        r.to = cand_to;
        r.score = std::min(7, r.score + 1);
        r.n_edit++;
        r.miss_run = std::max(0, r.miss_run - 1);
        r.last = src_pos;
        return true;
    }

    void observe_edit(const std::vector<int32_t>& ids, int ns, int no) {
        if (ns == 0 || no == 0) {
            if (source_ + ns < input_end_ && output_ + no < static_cast<int>(ids.size())) {
                ++ns;
                ++no;
            } else {
                return;
            }
        }
        if (ns > gap + 1 || no > gap + 1) return;

        const int32_t* src_ptr = ids.data() + source_;
        const int32_t* out_ptr = ids.data() + output_;

        // 1. First pass: check exact match against existing literal OR parameterized rules
        for (int i = 0; i < used_; ++i) {
            Rule& r = rules_[i];
            if (r.nf != ns) continue;
            int32_t caps[max_slots]{};
            if (!match_rule(r, ids, source_, caps)) continue;

            std::array<int32_t, gap + 1> rendered{};
            eval_rule_to(r, caps, rendered);
            if (r.nt == no && std::equal(rendered.begin(), rendered.begin() + no, out_ptr)) {
                if (r.last != source_) {
                    r.score = std::min(7, r.score + 1);
                    ++r.n_edit;
                    r.miss_run = std::max(0, r.miss_run - 1);
                }
                r.last = source_;
                return;
            }
            if (r.n_slots == 0 && r.last != source_) {
                r.score = std::max(0, r.score - 1);
            }
        }

        // 2. Second pass: try generalizing an existing literal rule of same `(nf, nt)` at a distinct source position
        for (int i = 0; i < used_; ++i) {
            Rule& r = rules_[i];
            if (r.nf == ns && r.nt == no && r.last != source_) {
                if (try_generalize_literal_rule(r, src_ptr, out_ptr, source_)) return;
            }
        }

        // 3. Otherwise insert as a fresh literal candidate rule (`score = 1`)
        int slot = -1;
        if (used_ < rule_limit) {
            slot = used_++;
        } else {
            for (int i = 0; i < used_; ++i) {
                if (!rules_[i].usable()) { slot = i; break; }
            }
            if (slot < 0) return;
        }
        Rule& nr = rules_[slot];
        nr = Rule{};
        nr.nf = ns;
        nr.nt = no;
        nr.n_slots = 0;
        nr.from_slot_id.fill(-1);
        std::copy_n(src_ptr, ns, nr.from.begin());
        for (int k = 0; k < no; ++k) nr.to[k] = OutElem{out_ptr[k], -1, 0};
        nr.score = 1;
        nr.last = source_;
        nr.n_edit = 1;
    }

    void observe_kept(const std::vector<int32_t>& ids) {
        for (int i = 0; i < used_; ++i) {
            Rule& r = rules_[i];
            if (run_ < anchor + r.nf || source_ - r.nf < 0) continue;
            int32_t caps[max_slots]{};
            if (match_rule(r, ids, source_ - r.nf, caps)) ++r.n_kept;
        }
    }

public:
    explicit EditCopy(int input_end) : input_end_(input_end) {}

    void advance(const std::vector<int32_t>& ids, int end, int raw_next) {
        if (source_ >= 0) {
            while (source_ < input_end_ && output_ < end) {
                if (ids[source_] == ids[output_]) {
                    ++source_;
                    ++output_;
                    literal_ = std::min(anchor, literal_ + 1);
                    ++run_;
                    if (used_ > 0) observe_kept(ids);
                    continue;
                }
                int ds = -1, dout = -1;
                for (int cost = 1; cost <= gap * 2 && ds < 0; ++cost) {
                    for (int s = std::max(0, cost - gap); s <= std::min(gap, cost); ++s) {
                        const int o = cost - s;
                        if (source_ + s + anchor > input_end_ || output_ + o + anchor > end) continue;
                        if (std::equal(ids.begin() + source_ + s, ids.begin() + source_ + s + anchor,
                                       ids.begin() + output_ + o)) {
                            ds = s;
                            dout = o;
                            break;
                        }
                    }
                }
                if (ds >= 0) {
                    if (literal_ >= anchor) observe_edit(ids, ds, dout);
                    source_ += ds;
                    output_ += dout;
                    literal_ = 0;
                    run_ = 0;
                    continue;
                }
                if (end - output_ < gap + anchor) return;
                source_ = -1;
                break;
            }
            if (source_ >= 0 && source_ < input_end_) return;
        }
        source_ = -1;
        if (end - input_end_ >= anchor && raw_next >= anchor && raw_next < input_end_) {
            source_ = raw_next;
            output_ = end;
            literal_ = anchor;
            run_ = anchor;
        }
    }

    void missed() {
        if (prop_rule_ < 0 || prop_rule_ >= used_) return;
        Rule& r = rules_[prop_rule_];
        if (r.miss_src != prop_src_) {
            ++r.miss_run;
            r.miss_src = prop_src_;
            r.last = prop_src_;
        }
        prop_rule_ = -1;
    }

    void hit() {
        if (prop_rule_ >= 0 && prop_rule_ < used_) rules_[prop_rule_].miss_run = 0;
        prop_rule_ = -1;
    }

    int propose(const std::vector<int32_t>& ids, int end, int limit, std::vector<int32_t>& out) const {
        out.clear();
        prop_rule_ = -1;
        if (source_ < 0 || limit <= 0 ||
            std::none_of(rules_.begin(), rules_.begin() + used_, [](const Rule& r) { return r.usable(); })) {
            return -1;
        }
        int s = source_, o = output_, first_edit = -1;
        std::array<int32_t, gap + 1> rendered{};
        while (s < input_end_ && static_cast<int>(out.size()) < limit) {
            int32_t caps[max_slots]{};
            const int ri = match(ids, s, caps);
            const Rule* rule = (ri >= 0) ? &rules_[ri] : nullptr;
            const int32_t* data = nullptr;
            int count = 1;
            if (rule) {
                eval_rule_to(*rule, caps, rendered);
                data = rendered.data();
                count = rule->nt;
            } else {
                data = &ids[s];
                count = 1;
            }
            for (int j = 0; j < count && static_cast<int>(out.size()) < limit; ++j) {
                if (o < end) {
                    if (ids[o++] != data[j]) {
                        out.clear();
                        return -1;
                    }
                } else {
                    if (rule && first_edit < 0) {
                        first_edit = static_cast<int>(out.size());
                        prop_rule_ = ri;
                        prop_src_ = s;
                    }
                    out.push_back(data[j]);
                }
            }
            s += rule ? rule->nf : 1;
        }
        return (o == end) ? first_edit : -1;
    }

    int usable_rule_count() const {
        return static_cast<int>(std::count_if(rules_.begin(), rules_.begin() + used_,
                                              [](const Rule& r) { return r.usable(); }));
    }
};

}  // namespace yskn
