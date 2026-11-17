/* SPDX-License-Identifier: Apache-2.0 */
/* A CPU-only stand-in for the engine library, used by the Python tests.  It is NOT the engine: it follows scripted outputs and
   mimics the call conventions (output ids written pass by pass, cancel at a pass boundary, thread-local last error, prompt reuse
   reporting) so that the Python layer can be tested without a GPU.  Build: cc -shared -fPIC -O1 -pthread yskn_stub.c -o libyskn.so */
#include <stdatomic.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define EXPORT __attribute__((visibility("default")))
#define EOS_ID 248046
#define VOCAB 248320

typedef struct { int32_t *ids; int n; int loop_from; } script_t;
typedef struct yskn_engine {
    int max_ctx, pos, draft;
    atomic_int cancel;
    int32_t *hist; int nhist;
    int32_t *snap; int nsnap;
    int opts[64];
    int tf_rows; int32_t tf_am[64]; float tf_mg[64];     /* the last teacher-forced pass */
    int wallowed, wmask, wpref, warch;                  /* 0.1.7 wide copy passes (the STUB_WIDE build only) */
    int xwoff;                                          /* set_opt 31 = 0: no exact-sum GEMV for 17..64 rows */
} yskn_engine;

static __thread char g_err[256];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static script_t g_queue[64];
static int g_qhead = 0, g_qtail = 0;
static int g_delay_us = 0;
static int g_calls = 0;
static int g_last_n_prompt = 0;
static int g_prefill_max = 0, g_prefill_calls = 0;     /* yskn_prefill: the longest read, and the reads (0.1.7) */
static int32_t g_last_cfg[46];
static int g_last_n_cfg = 0;

static void set_err(const char *m) { strncpy(g_err, m, sizeof g_err - 1); g_err[sizeof g_err - 1] = 0; }
static void sleep_us(int us) { if (us > 0) { struct timespec t = { us / 1000000, (us % 1000000) * 1000L }; nanosleep(&t, 0); } }

EXPORT void stub_push_script(const int32_t *ids, int n, int loop_from) {
    pthread_mutex_lock(&g_mu);
    script_t s; s.ids = malloc(sizeof(int32_t) * (n ? n : 1)); memcpy(s.ids, ids, sizeof(int32_t) * n); s.n = n; s.loop_from = loop_from;
    g_queue[g_qtail++ % 64] = s;
    pthread_mutex_unlock(&g_mu);
}
EXPORT void stub_set_delay_us(int us) { g_delay_us = us; }
EXPORT int stub_get_opt(yskn_engine *e, int key) { return e->opts[key & 63]; }
EXPORT int stub_calls(void) { return g_calls; }
EXPORT int stub_last_n_prompt(void) { return g_last_n_prompt; }
EXPORT int stub_nhist(yskn_engine *e) { return e->nhist; }
EXPORT int stub_last_cfg(int32_t *out) { memcpy(out, g_last_cfg, sizeof g_last_cfg); return g_last_n_cfg; }

EXPORT yskn_engine *yskn_open(const char *path, int max_ctx, int device, int flags) {
    (void)flags;
    const char *log = getenv("STUB_OPEN_LOG");               /* what the real engine would see: the device and CUDA's numbering */
    if (log && log[0]) {
        FILE *f = fopen(log, "a");
        if (f) {
            const char *order = getenv("CUDA_DEVICE_ORDER"), *vis = getenv("CUDA_VISIBLE_DEVICES");
            fprintf(f, "device=%d order=%s visible=%s\n", device, order ? order : "(unset)", vis ? vis : "(unset)");
            fclose(f);
        }
    }
    if (strstr(path, "fail")) { set_err("cudaMalloc failed: out of memory"); return 0; }
    yskn_engine *e = calloc(1, sizeof *e);
    e->max_ctx = max_ctx;
#ifdef STUB_WIDE
    {   /* as the engine: YSKN_WIDE not set = by the code architecture (STUB_WIDE_ARCH, default 86), 0 off, 1 on (8.x closed), 2 all */
        const char *w = getenv("YSKN_WIDE"), *a = getenv("STUB_WIDE_ARCH");
        e->warch = a && a[0] ? atoi(a) : 86; e->wpref = w && w[0] ? atoi(w) : -1;
        int native = e->warch >= 120;
        e->wallowed = e->wpref > 0 || (e->wpref < 0 && native);
        e->wmask = !e->wallowed ? 0 : ((native || e->wpref != 1) ? 7 : 0);
    }
#endif
    return e;
}
EXPORT void yskn_close(yskn_engine *e) { if (e) { free(e->hist); free(e->snap); free(e); } }
EXPORT int yskn_reset(yskn_engine *e) { free(e->hist); e->hist = 0; e->nhist = 0; e->pos = 0; e->tf_rows = 0; return 0; }
EXPORT int yskn_position(yskn_engine *e) { return e->pos; }
EXPORT const char *yskn_last_error(void) { return g_err; }
EXPORT int yskn_cancel(yskn_engine *e) { atomic_store(&e->cancel, 1); return 0; }
EXPORT int yskn_set_opt(yskn_engine *e, int key, int value) {
#ifdef STUB_WIDE
    if (key == 38) { if (value < 0 || value > 7) { set_err("set_opt 38: widths of copy passes must be 0..7"); return 1; } e->wmask = e->wallowed ? value : 0; }
#endif
    if (key == 31) e->xwoff = value == 0;
    e->opts[key & 63] = value; return 0;
}
EXPORT int yskn_load_bdraft(yskn_engine *e, const char *path) {
    if (strstr(path, "bad")) { set_err("bdraft: bad container"); return 1; }
    e->draft = 1; return 0;
}
EXPORT int yskn_prefill(yskn_engine *e, const int32_t *ids, int n, int32_t *first) {   /* as the engine: from the current position */
    (void)ids;
    if (n < 1 || e->pos + n > e->max_ctx) { set_err("prefill: context overflow"); return 1; }
    e->pos += n; *first = 1000; g_prefill_calls++; if (n > g_prefill_max) g_prefill_max = n; return 0;
}
EXPORT int stub_prefill_max(void) { return g_prefill_max; }
EXPORT int stub_prefill_calls(void) { return g_prefill_calls; }
EXPORT int yskn_decode(yskn_engine *e, int32_t *out, int steps) { for (int i = 0; i < steps; i++) out[i] = 1000 + i; e->pos += steps; return 0; }
EXPORT int yskn_logits(yskn_engine *e, float *out) { (void)e; (void)out; return 0; }
EXPORT int yskn_nll(yskn_engine *e, const int32_t *ids, int n, float *out) { (void)e; (void)ids; for (int i = 0; i + 1 < n; i++) out[i] = 1.0f; return 0; }
EXPORT int yskn_state_hash(yskn_engine *e, uint64_t *out, int max_out, int *n_out) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < e->nhist; i++) { h ^= (uint64_t)(uint32_t)e->hist[i]; h *= 1099511628211ULL; }
    if (max_out < 2) { set_err("state_hash: buffer too small"); return 1; }
    out[0] = (uint64_t)e->pos; out[1] = h; *n_out = 2; return 0;
}

/* Teacher forcing.  The stub "model" predicts the id that follows (argmax = target), with an NLL and a logit margin that depend on
   the target and the position.  Knobs for the tests (read at each call; names without YSKN_ so the package keeps them):
   STUB_TF_NLL_SHIFT=x  add x to every NLL;  STUB_TF_FLIP_EVERY=k  argmax = target + 1 at every position p with p % k == k - 1;
   STUB_TF_NAN=p  NaN as the NLL at position p.
   Wide passes (0.1.4): STUB_TF_WIDE=1  accept passes of up to 64 rows (else 1..16, as on GPUs without wide passes);
   STUB_TF_WIDE_NLL_SHIFT=x  add x to the NLL in passes of more than 16 rows. */
static double env_f(const char *k, double d) { const char *v = getenv(k); return v && v[0] ? atof(v) : d; }
static int env_i(const char *k, int d) { const char *v = getenv(k); return v && v[0] ? atoi(v) : d; }
static int tf_pass(yskn_engine *e, const int32_t *ids, const int32_t *targets, int M, int32_t *out_argmax, float *out_nll) {
#ifdef STUB_WIDE
    const int wide_ok = e->wallowed;
#else
    const int wide_ok = 0;
#endif
    if (M < 1 || M > ((env_i("STUB_TF_WIDE", 0) || wide_ok) ? 64 : 16)) { set_err("verify: M must be 1..16 (1..64 where wide passes are on)"); return 1; }
    if (e->pos + M > e->max_ctx) { set_err("verify: context overflow"); return 1; }
    double shift = env_f("STUB_TF_NLL_SHIFT", 0.0) + (M > 16 ? env_f("STUB_TF_WIDE_NLL_SHIFT", 0.0) : 0.0);
    int flip = env_i("STUB_TF_FLIP_EVERY", 0), nan_at = env_i("STUB_TF_NAN", -1);
    for (int i = 0; i < M; i++) {
        int p = e->pos + i, t = targets ? targets[i] : ids[i] + 1;
        int am = (flip > 0 && p % flip == flip - 1) ? t + 1 : t;
        float nll = (float)(0.05 + 0.001 * ((t * 31 + p * 7) % 97) + shift);
        if (p == nan_at) nll = NAN;
        e->tf_am[i] = am; e->tf_mg[i] = (float)(0.5 + ((t + p) % 8) * 0.5);
        if (out_argmax) out_argmax[i] = am;
        if (out_nll) out_nll[i] = nll;
    }
    e->tf_rows = M;
    return 0;
}
EXPORT int yskn_verify_nll(yskn_engine *e, const int32_t *ids, const int32_t *targets, int M, int use_graph, int32_t *out_argmax, float *out_nll) {
    (void)use_graph; g_err[0] = 0; return tf_pass(e, ids, targets, M, out_argmax, out_nll);
}
EXPORT int yskn_verify(yskn_engine *e, const int32_t *ids, int M, int use_graph, int32_t *out_argmax, int32_t *top_ids, float *top_vals) {
    (void)use_graph; (void)top_ids; (void)top_vals;
    if (e->tf_rows != M) { set_err("stub: verify without targets only repeats the last pass"); return 1; }
    for (int i = 0; i < M; i++) out_argmax[i] = e->tf_am[i];
    (void)ids; return 0;
}
EXPORT int yskn_commit(yskn_engine *e, int n_accept) {
    if (n_accept < 0 || n_accept >= e->tf_rows) { set_err("commit: n_accept out of range of the last verify pass"); return 1; }
    e->pos += n_accept + 1; return 0;
}
EXPORT int yskn_logits_rows(yskn_engine *e, int row0, int n, float *out) {
    if (row0 < 0 || row0 + n > e->tf_rows) { set_err("logits_rows: rows out of range"); return 1; }
    memset(out, 0, sizeof(float) * (size_t)n * VOCAB);
    for (int r = 0; r < n; r++) out[(size_t)r * VOCAB + e->tf_am[row0 + r]] = e->tf_mg[row0 + r];
    return 0;
}

static int lcp(const int32_t *a, int na, const int32_t *b, int nb) { int i = 0; while (i < na && i < nb && a[i] == b[i]) i++; return i; }

EXPORT int yskn_spec_generate(yskn_engine *e, const int32_t *prompt, int n_prompt, int n_gen, const int32_t *cfg, int n_cfg,
                              int32_t *out, double *stats, double *times_ms) {
    g_err[0] = 0;
    atomic_store(&e->cancel, 0);
    g_calls++; g_last_n_prompt = n_prompt;
    if (n_cfg != 41 && n_cfg != 44 && n_cfg != 46) { set_err("spec_generate: bad config length"); return 1; }
    memset(g_last_cfg, 0, sizeof g_last_cfg); memcpy(g_last_cfg, cfg, sizeof(int32_t) * n_cfg); g_last_n_cfg = n_cfg;
    if (n_prompt < 1 || n_prompt + n_gen + 1 > e->max_ctx) { set_err("spec_generate: context overflow"); return 1; }
    if (prompt[0] == 7) { set_err("stub: injected failure"); return 1; }
    int eos1 = cfg[20], eos2 = cfg[21];
    int snap_on = n_cfg >= 44 ? cfg[42] : 0, snap_tok = n_cfg >= 44 ? cfg[43] : -1;

    /* prompt reuse: live state, snapshot, or fresh */
    int mode = 0, cached = 0;
    int l = lcp(e->hist, e->nhist, prompt, n_prompt);
    if (snap_on && e->nhist > 0 && l == e->nhist) { mode = 1; cached = l; }
    else if (snap_on && e->nsnap > 0 && lcp(e->snap, e->nsnap, prompt, n_prompt) == e->nsnap) { mode = 2; cached = e->nsnap; }

    /* the scripted output of this call */
    script_t sc;
    pthread_mutex_lock(&g_mu);
    if (g_qhead != g_qtail) sc = g_queue[g_qhead++ % 64];
    else {                                                /* STUB_GEN_DIFF_AT=k: from id k on, the default output is another one */
        int diff_at = env_i("STUB_GEN_DIFF_AT", -1);
        sc.n = 40; sc.ids = malloc(sizeof(int32_t) * 40); sc.loop_from = -1;
        for (int i = 0; i < 40; i++) sc.ids[i] = (diff_at >= 0 && i >= diff_at ? 2000 : 1000) + i;
    }
    pthread_mutex_unlock(&g_mu);
    int k = 0;
    #define NEXT() (k < sc.n ? sc.ids[k++] : (sc.loop_from >= 0 && sc.n > 0 ? (k = sc.loop_from, sc.ids[k++]) : EOS_ID))

    int idx = 0, passes = 0, hit = 0;
    long dr[3] = { 0, 0, 0 }, ac[3] = { 0, 0, 0 };        /* look-ahead counts: 1 + 1 + 2 tokens proposed per pass, the first   */
    static const int pat[4] = { 1, 3, 2, 4 };            /* emitted - 1 of them confirmed, in that order (as the engine's 3..6, 23, 24) */
    out[idx++] = NEXT();                                  /* the prompt pass predicts the first id */
    hit = (out[0] == eos1 || out[0] == eos2);
    sleep_us(g_delay_us);
    while (!hit && idx < n_gen + 1 && !atomic_load(&e->cancel)) {
        int cnt = pat[passes % 4], emitted = 0;
        for (int j = 0; j < cnt && idx < n_gen + 1; j++) {
            int32_t t = NEXT();
            out[idx++] = t; emitted++;
            if (t == eos1 || t == eos2) { hit = 1; break; }
        }
        int acc = emitted > 0 ? emitted - 1 : 0;
        dr[0] += 1; dr[1] += 1; dr[2] += 2;
        ac[0] += acc > 0 ? 1 : 0; ac[1] += acc > 1 ? 1 : 0; ac[2] += acc > 2 ? acc - 2 : 0;
        passes++;
        sleep_us(g_delay_us);
    }
    free(sc.ids);

    memset(stats, 0, sizeof(double) * 48);
    stats[0] = passes + 1; stats[1] = idx; stats[30] = idx - 1; stats[31] = hit; stats[41] = mode; stats[42] = cached;
    stats[44] = env_i("STUB_GEN_WIDE_PASSES", 0); stats[47] = env_i("STUB_GEN_MAX_ROWS", 0);   /* wide passes / widest pass (0.1.4) */
    stats[3] = dr[0]; stats[4] = ac[0]; stats[5] = dr[1]; stats[6] = ac[1]; stats[23] = dr[2]; stats[24] = ac[2];
#ifdef STUB_WIDE
    if (n_cfg >= 46) { stats[58] = env_i("STUB_GEN_BACKOFFS", 0); stats[63] = e->wmask; }
#endif
    times_ms[0] = 0.01 * (n_prompt - cached); times_ms[1] = 0.1 * idx;

    /* the engine's state: prompt + all generated ids except the last one */
    int nh = n_prompt + idx - 1;
    int32_t *h = malloc(sizeof(int32_t) * (nh > 0 ? nh : 1));
    memcpy(h, prompt, sizeof(int32_t) * n_prompt); memcpy(h + n_prompt, out, sizeof(int32_t) * (idx - 1));
    free(e->hist); e->hist = h; e->nhist = nh; e->pos = nh;
    if (snap_on && snap_tok >= 0) {                       /* snapshot right after the last chat-start marker of the prompt */
        int last = -1; for (int i = 0; i < n_prompt; i++) if (prompt[i] == snap_tok) last = i;
        if (last >= 0) { free(e->snap); e->nsnap = last + 1; e->snap = malloc(sizeof(int32_t) * e->nsnap); memcpy(e->snap, prompt, sizeof(int32_t) * e->nsnap); }
    }
    return 0;
}

#ifdef STUB_WIDE
/* ---- 0.1.7: wide copy passes per GPU (built with -DSTUB_WIDE only, so the tests of older behaviour see a library without them) ----
   Knobs: STUB_WIDE_ARCH (code architecture, default 86), STUB_WIDE_T16 (ms of a 16-row cycle, default 11), STUB_WIDE_R32 / R48 / R64
   (time of a W-row cycle / 16 rows, defaults 1.39 / 1.94 / 2.27, the RTX 5090's), STUB_WIDE_SMP (factor on W-row cycles while
   sampling), STUB_WIDE_FREE (free MiB, default 4000), STUB_WIDE_JITTER (relative noise of the rounds: +x on every other call). */
static int g_prepare_calls = 0, g_cycle_calls = 0;
EXPORT int stub_prepare_calls(void) { return g_prepare_calls; }
EXPORT int stub_wide_mask(yskn_engine *e) { return e->wmask; }
EXPORT const char *yskn_build_info(void) { return "cuda=0;wide=1;stub=1"; }
EXPORT int yskn_prepare_wide(yskn_engine *e, int max_rows) { (void)e; (void)max_rows; g_prepare_calls++; return 0; }
EXPORT int yskn_time_cycle(yskn_engine *e, const int32_t *ids, int M, int iters, float *ms) {
    (void)ids;
    if (M < 2 || M > 64 || (M > 16 && !e->wallowed)) { set_err("time_cycle: M must be 2..16 (2..64 where wide passes are allowed)"); return 1; }
    if (e->pos < 1 || e->pos + M > e->max_ctx) { set_err("time_cycle: read a prompt first"); return 1; }
    double r = 1.0;
    if (M > 16) {
        const char *k = M <= 32 ? "STUB_WIDE_R32" : (M <= 48 ? "STUB_WIDE_R48" : "STUB_WIDE_R64");
        r = env_f(k, M <= 32 ? 1.39 : (M <= 48 ? 1.94 : 2.27));
        if (e->opts[10] > 0) r *= env_f("STUB_WIDE_SMP", 1.0);
        if ((g_cycle_calls++ & 1) && env_f("STUB_WIDE_JITTER", 0.0) > 0) r *= 1.0 + env_f("STUB_WIDE_JITTER", 0.0);
    }
    for (int i = 0; i < iters; i++) ms[i] = (float)(env_f("STUB_WIDE_T16", 11.0) * r);
    return 0;
}
EXPORT int yskn_wide_state(yskn_engine *e, int32_t *out, int n, int *n_out) {
    int32_t v[27] = {0};
    v[0] = 1; v[1] = e->warch; v[2] = e->warch >= 120 ? 120 : e->warch; v[3] = 82; v[4] = e->wallowed; v[5] = e->wallowed || e->warch >= 120;
    v[6] = (!e->xwoff && (e->warch >= 120 || e->wmask)) ? 64 : 16; v[7] = e->wallowed ? e->wmask : 0; v[8] = e->wpref; v[9] = e->warch >= 120;
    for (int i = 0; i < 4; i++) v[10 + i] = e->opts[39 + i];
    v[14] = env_i("STUB_WIDE_FREE", 4000); v[15] = 24576; v[16] = 0; v[17] = 1; v[18] = 0;
    v[19] = 0x12345678; v[20] = (int32_t)0x9abcdef0; v[21] = 0x0fedcba9; v[22] = (int32_t)0x87654321;
    v[24] = e->warch; v[26] = e->draft;
    int k = n < 27 ? n : 27; for (int i = 0; i < k; i++) out[i] = v[i]; *n_out = k; return 0;
}
#endif
