/* SPDX-License-Identifier: Apache-2.0 */
/* A CPU-only stand-in for the engine library, used by the Python tests.  It is NOT the engine: it follows scripted outputs and
   mimics the call conventions (output ids written pass by pass, cancel at a pass boundary, thread-local last error, prompt reuse
   reporting) so that the Python layer can be tested without a GPU.  Build: cc -shared -fPIC -O1 -pthread yskn_stub.c -o libyskn.so

   The multi-stream entries (yskn_ms_*, experimental, see src/yskn_ms.h) follow the contract of that header with a stand-in model: the
   next token is a function of the tokens before it (a hash of the whole sequence), its length and the sampling of the stream (greedy
   ignores the other values; else temperature, top-p, top-k, min-p, sampler and seed count).  So a stream gives the same ids in a pass
   of several streams as alone, whatever its drafts, as the engine does.  yskn_spec_generate uses the same model on an engine that
   stub_model turned it on for (scripts pushed with stub_push_script still come first).  The checks and refusals are those of the
   header: the frontier (the first row of a slot), commits of confirmed drafts only, empty and moved-out slots, the live sequence and the
   frontier form of yskn_spec_generate (cfg[46]).  Built with -DSTUB_NO_MS the library has no multi-stream entries (an older library).
   Image input: stub_set_visual has the arguments and the effects of yskn_set_visual (a test binds it as that entry for one engine, so
   that the stub's own table keeps the 0.1.x form without image input); while it is on, the multi-stream entries refuse as the header
   says (code 3).
   For the commands run as programs (no test API): STUB_MODEL=k or k,vocab turns the stand-in model on for every engine the process
   opens (as stub_model(e, 1, k, vocab)); STUB_DELAY_US=us sets the time of a pass (as stub_set_delay_us); STUB_MS_SALT=x (not 0)
   makes the multi-stream passes give other tokens than the same stream alone (a GPU on which they differ). */
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
#define EOS2_ID 248044
#define VOCAB 248320
#define MS_MAX 64                                         /* slots at most */
#define MS_ROWS 16                                        /* rows of a multi-stream pass at most */
#define MS_TOP 8                                          /* entries of a top list */

typedef struct { int32_t *ids; int n; int loop_from; } script_t;
typedef struct { int temp, top_p, top_k, min_p, sampler; uint32_t seed; } smp_t;
typedef struct {                                          /* a stream slot */
    int filled, out;                                      /* an ms_load filled it / ms_unload moved its stream out */
    int32_t *kv; int nkv, cap;                            /* the tokens in its KV (its position = nkv) */
    uint64_t h;                                           /* the model's hash of kv */
    int load_pos, front;                                  /* position at the load; the token it emitted last (-1 = not known) */
    int src_known, src_self, src_tick; int32_t *src_ids; int nsrc;   /* the source's live sequence at the load */
    smp_t smp;
} ms_slot_t;
typedef struct yskn_engine {
    int max_ctx, pos, draft;
    atomic_int cancel;
    int32_t *hist; int nhist;
    int32_t *snap; int nsnap;
    int opts[64];
    int tf_rows; int32_t tf_am[64]; float tf_mg[64];     /* the last teacher-forced pass */
    int live_ok, front, tick, snap_born;                  /* the live sequence (hist) and its frontier; reads so far; snapshot's read */
    int model_on, model_eos, model_vocab;                 /* the stand-in model (stub_model) */
    int ms_ns, ms_made, ms_drafter, ms_open, ms_dvalid, ms_dM, ms_draft_mode, ms_fail_in, ms_steps, ms_drafts;
    int visual_on, visual_sets;                           /* image input on; calls of stub_set_visual so far */
    int32_t ms_dlay[MS_ROWS], ms_dtoks[MS_ROWS];          /* the rows and tokens of the last ms_draft */
    int ms_last_r0[MS_MAX], ms_last_len[MS_MAX];          /* the open pass */
    int32_t ms_last_toks[MS_ROWS], ms_last_am[MS_ROWS];
    ms_slot_t ms[MS_MAX];
    int wallowed, wmask, wpref, warch;                  /* 0.1.7 wide copy passes (the STUB_WIDE build only) */
    int xwoff;                                          /* set_opt 31 = 0: no exact-sum GEMV for 17..64 rows */
    uint8_t *suppress; int suppress_n;
} yskn_engine;

static __thread char g_err[256];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static script_t g_queue[64];
static int g_qhead = 0, g_qtail = 0;
static int g_delay_us = 0;
static int g_calls = 0;
static int g_last_n_prompt = 0;
static int g_prefill_max = 0, g_prefill_calls = 0;     /* yskn_prefill: the longest read, and the reads (0.1.7) */
static int32_t g_last_cfg[64];
static int g_last_n_cfg = 0;

static void set_err(const char *m) { strncpy(g_err, m, sizeof g_err - 1); g_err[sizeof g_err - 1] = 0; }
/* a test asks for a failure by a word in the name of the model folder or of the file (the last two parts of the path): the folders
   above them (a temporary folder with a random name) do not count */
static int marked(const char *path, const char *word) {
    const char *tail = path + strlen(path);
    int parts = 0;
    while (tail > path && !((tail[-1] == '/' || tail[-1] == '\\') && ++parts == 2)) tail--;
    return strstr(tail, word) != 0;
}
static void sleep_us(int us) { if (us > 0) { struct timespec t = { us / 1000000, (us % 1000000) * 1000L }; nanosleep(&t, 0); } }
static double env_f(const char *k, double d) { const char *v = getenv(k); return v && v[0] ? atof(v) : d; }
static int env_i(const char *k, int d) { const char *v = getenv(k); return v && v[0] ? atoi(v) : d; }

/* ---- the stand-in model ------------------------------------------------------------------------------------------------------ */
static uint64_t mix64(uint64_t x) { x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27; x *= 0x94d049bb133111ebULL; x ^= x >> 31; return x; }
#define H0 0x243f6a8885a308d3ULL
static uint64_t roll(uint64_t h, int32_t t) { return mix64(h ^ ((uint64_t)(uint32_t)t * 0x9e3779b97f4a7c15ULL)); }
static uint64_t hash_ids(const int32_t *ids, int n) { uint64_t h = H0; for (int i = 0; i < n; i++) h = roll(h, ids[i]); return h; }
static smp_t carrier_smp(const yskn_engine *e) {
    smp_t s = { e->opts[10], e->opts[11], e->opts[12], e->opts[29], e->opts[16], (uint32_t)e->opts[13] }; return s;
}
/* the token after a sequence of n tokens with hash h */
static int32_t model_next(const yskn_engine *e, uint64_t h, int n, const smp_t *s) {
    uint64_t k = mix64(h + (uint64_t)n * 0xd1b54a32d192ed03ULL);
    if (s->temp > 0)
        k = mix64(k ^ mix64(((uint64_t)s->seed << 32) ^ ((uint64_t)(uint32_t)s->temp << 8) ^ ((uint64_t)(uint32_t)s->top_p << 20) ^
                            ((uint64_t)(uint32_t)s->top_k << 40) ^ ((uint64_t)(uint32_t)s->min_p << 46) ^ ((uint64_t)(uint32_t)s->sampler << 58)));
    int vocab = e->model_vocab > 0 ? e->model_vocab : 64;
    for (int attempt = 0; attempt < vocab + 2; attempt++) {
        int32_t tok;
        if (attempt == 0 && e->model_eos > 0 && (k >> 8) % (uint64_t)e->model_eos == 0)
            tok = ((k >> 3) & 1) ? EOS_ID : EOS2_ID;
        else
            tok = 1000 + (int32_t)((k >> 24) % (uint64_t)vocab);
        if (!e->suppress || e->suppress_n <= 0 || !e->suppress[tok]) return tok;
        k = mix64(k + 0x9e3779b97f4a7c15ULL);
    }
    for (int32_t v = 0; v < VOCAB; v++) if (!e->suppress[v]) return v;
    return 0;
}
static int32_t model_take(const yskn_engine *e, uint64_t *h, int *n, const smp_t *s) {
    int32_t t = model_next(e, *h, *n, s); *h = roll(*h, t); (*n)++; return t;
}
static uint64_t fnv(const int32_t *ids, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= (uint64_t)(uint32_t)ids[i]; h *= 1099511628211ULL; }
    return h;
}

/* what every single-request entry does first (as the engine's ms_gate): finish the pending pass work; the drafts are dropped */
static void ms_gate(yskn_engine *e) { e->ms_open = 0; e->ms_dvalid = 0; }

EXPORT void stub_push_script(const int32_t *ids, int n, int loop_from) {
    pthread_mutex_lock(&g_mu);
    script_t s; s.ids = malloc(sizeof(int32_t) * (n ? n : 1)); memcpy(s.ids, ids, sizeof(int32_t) * n); s.n = n; s.loop_from = loop_from;
    g_queue[g_qtail++ % 64] = s;
    pthread_mutex_unlock(&g_mu);
}
EXPORT void stub_set_delay_us(int us) { g_delay_us = us; }
/* drop the scripts nobody used (they would come before the stand-in model) */
EXPORT void stub_clear_scripts(void) {
    pthread_mutex_lock(&g_mu);
    while (g_qhead != g_qtail) free(g_queue[g_qhead++ % 64].ids);
    pthread_mutex_unlock(&g_mu);
}
EXPORT int stub_get_opt(yskn_engine *e, int key) { return e->opts[key & 63]; }
EXPORT int stub_calls(void) { return g_calls; }
EXPORT int stub_last_n_prompt(void) { return g_last_n_prompt; }
EXPORT int stub_nhist(yskn_engine *e) { return e->nhist; }
EXPORT int stub_last_cfg(int32_t *out) { memcpy(out, g_last_cfg, sizeof g_last_cfg); return g_last_n_cfg; }
/* the stand-in model for yskn_spec_generate of this engine (on = 1; the multi-stream passes always use it); eos_every = k > 0: about
   one token in k is a stop id; vocab = the number of other tokens (1000, 1001, ...) */
EXPORT void stub_model(yskn_engine *e, int on, int eos_every, int vocab) { e->model_on = on; e->model_eos = eos_every; e->model_vocab = vocab > 0 ? vocab : 64; }
/* drafts of ms_draft: 0 = right up to a place that depends on the stream, 1 = all right, 2 = the first one wrong */
EXPORT void stub_ms_draft_mode(yskn_engine *e, int mode) { e->ms_draft_mode = mode; }
/* k > 0: the k-th yskn_ms_step2 from now fails (an engine fault) */
EXPORT void stub_ms_fail_in(yskn_engine *e, int k) { e->ms_fail_in = k; }
EXPORT int stub_ms_steps(yskn_engine *e) { return e->ms_steps; }
EXPORT int stub_ms_drafts(yskn_engine *e) { return e->ms_drafts; }
EXPORT int stub_live_ok(yskn_engine *e) { return e->live_ok; }
EXPORT int stub_front(yskn_engine *e) { return e->live_ok ? e->front : -1; }
EXPORT int stub_ms_front(yskn_engine *e, int slot) { return slot >= 0 && slot < MS_MAX ? e->ms[slot].front : -2; }
EXPORT int stub_visual_on(yskn_engine *e) { return e->visual_on; }
EXPORT int stub_visual_sets(yskn_engine *e) { return e->visual_sets; }

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
    if (marked(path, "fail")) { set_err("cudaMalloc failed: out of memory"); return 0; }
    yskn_engine *e = calloc(1, sizeof *e);
    e->max_ctx = max_ctx;
    e->front = -1; e->model_vocab = 64;
    for (int s = 0; s < MS_MAX; s++) { e->ms_last_r0[s] = -1; e->ms[s].front = -1; e->ms[s].load_pos = -1; }
    const char *dl = getenv("STUB_DELAY_US");
    if (dl && dl[0]) g_delay_us = atoi(dl);
    const char *m = getenv("STUB_MODEL");
    if (m && m[0]) {
        const char *c = strchr(m, ',');
        e->model_on = 1; e->model_eos = atoi(m); e->model_vocab = c && atoi(c + 1) > 0 ? atoi(c + 1) : 64;
    }
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
EXPORT void yskn_close(yskn_engine *e) {
    if (!e) return;
    for (int s = 0; s < MS_MAX; s++) { free(e->ms[s].kv); free(e->ms[s].src_ids); }
    free(e->suppress); free(e->hist); free(e->snap); free(e);
}
EXPORT int yskn_reset(yskn_engine *e) {
    ms_gate(e); free(e->hist); e->hist = 0; e->nhist = 0; e->pos = 0; e->tf_rows = 0; e->live_ok = 0; e->front = -1; return 0;
}
EXPORT int yskn_position(yskn_engine *e) { return e->pos; }
/* the effects of yskn_set_visual: features == NULL clears the image input; setting it or clearing it resets the state and drops the
   live sequence and the saved prompt state (the engine keeps them only when the image head stays the same) */
EXPORT int stub_set_visual(yskn_engine *e, const uint16_t *features, const int32_t *slots, int n_image, const int32_t *positions,
                           int n_prompt, int delta) {
    (void)slots; (void)positions; (void)n_prompt; (void)delta;
    g_err[0] = 0;
    yskn_reset(e);
    free(e->snap); e->snap = 0; e->nsnap = 0;
    e->visual_on = features != 0 && n_image > 0;
    e->visual_sets++;
    return 0;
}
EXPORT int yskn_set_suppress(yskn_engine *e, const int32_t *ids, int n) {
    g_err[0] = 0;
    ms_gate(e);
    if (n < 0) { set_err("set_suppress: n must be >= 0"); return -1; }
    if (n > 0 && !ids) { set_err("set_suppress: missing token IDs"); return -1; }
    if (n == 0) { e->suppress_n = 0; if (e->suppress) memset(e->suppress, 0, VOCAB); return 0; }
    uint8_t *mask = calloc(VOCAB, 1);
    if (!mask) { set_err("set_suppress: out of memory"); return -1; }
    int unique = 0;
    for (int i = 0; i < n; i++) {
        int32_t id = ids[i];
        if (id < 0 || id >= VOCAB) { free(mask); set_err("set_suppress: token ID out of range"); return -1; }
        if (!mask[id]) { mask[id] = 1; unique++; }
    }
    if (VOCAB - unique < 20) { free(mask); set_err("set_suppress: at least 20 unmasked token IDs must remain"); return -1; }
    free(e->suppress);
    e->suppress = mask;
    e->suppress_n = unique;
    return 0;
}
EXPORT int stub_suppress_count(yskn_engine *e) { return e->suppress_n; }
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
    if (marked(path, "bad")) { set_err("bdraft: bad container"); return 1; }
    e->draft = 1; return 0;
}
EXPORT int yskn_prefill(yskn_engine *e, const int32_t *ids, int n, int32_t *first) {   /* as the engine: from the current position */
    (void)ids; ms_gate(e); e->live_ok = 0;
    if (n < 1 || e->pos + n > e->max_ctx) { set_err("prefill: context overflow"); return 1; }
    e->pos += n; *first = 1000; g_prefill_calls++; if (n > g_prefill_max) g_prefill_max = n; return 0;
}
EXPORT int stub_prefill_max(void) { return g_prefill_max; }
EXPORT int stub_prefill_calls(void) { return g_prefill_calls; }
EXPORT int yskn_decode(yskn_engine *e, int32_t *out, int steps) { ms_gate(e); e->live_ok = 0; for (int i = 0; i < steps; i++) out[i] = 1000 + i; e->pos += steps; return 0; }
EXPORT int yskn_logits(yskn_engine *e, float *out) { (void)e; (void)out; return 0; }
EXPORT int yskn_nll(yskn_engine *e, const int32_t *ids, int n, float *out) { (void)ids; ms_gate(e); e->live_ok = 0; for (int i = 0; i + 1 < n; i++) out[i] = 1.0f; return 0; }
EXPORT int yskn_state_hash(yskn_engine *e, uint64_t *out, int max_out, int *n_out) {
    if (max_out < 2) { set_err("state_hash: buffer too small"); return 1; }
    out[0] = (uint64_t)e->pos; out[1] = fnv(e->hist, e->nhist); *n_out = 2; return 0;
}

/* Teacher forcing.  The stub "model" predicts the id that follows (argmax = target), with an NLL and a logit margin that depend on
   the target and the position.  Knobs for the tests (read at each call; names without YSKN_ so the package keeps them):
   STUB_TF_NLL_SHIFT=x  add x to every NLL;  STUB_TF_FLIP_EVERY=k  argmax = target + 1 at every position p with p % k == k - 1;
   STUB_TF_NAN=p  NaN as the NLL at position p.
   Wide passes (0.1.4): STUB_TF_WIDE=1  accept passes of up to 64 rows (else 1..16, as on GPUs without wide passes);
   STUB_TF_WIDE_NLL_SHIFT=x  add x to the NLL in passes of more than 16 rows. */
static int tf_pass(yskn_engine *e, const int32_t *ids, const int32_t *targets, int M, int32_t *out_argmax, float *out_nll) {
    ms_gate(e); e->live_ok = 0;
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
    ms_gate(e); e->live_ok = 0;
    if (e->tf_rows != M) { set_err("stub: verify without targets only repeats the last pass"); return 1; }
    for (int i = 0; i < M; i++) out_argmax[i] = e->tf_am[i];
    (void)ids; return 0;
}
EXPORT int yskn_commit(yskn_engine *e, int n_accept) {
    e->live_ok = 0;
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

/* cfg: 41, 44, 46 or 47 values; the 47th (cfg[46]) = 1 is the frontier form of yskn_ms.h: the last prompt token is the frontier of the
   live sequence (the token the carrier's stream emitted last), it is not read, and the first id comes from the first pass */
EXPORT int yskn_spec_generate(yskn_engine *e, const int32_t *prompt, int n_prompt, int n_gen, const int32_t *cfg, int n_cfg,
                              int32_t *out, double *stats, double *times_ms) {
    g_err[0] = 0;
    atomic_store(&e->cancel, 0);
    g_calls++; g_last_n_prompt = n_prompt;
    if (n_cfg != 41 && n_cfg != 44 && n_cfg != 46 && n_cfg != 47) { set_err("spec_generate: bad config length"); return 1; }
    memset(g_last_cfg, 0, sizeof g_last_cfg); memcpy(g_last_cfg, cfg, sizeof(int32_t) * n_cfg); g_last_n_cfg = n_cfg;
    if (n_prompt < 1 || n_prompt + n_gen + 1 > e->max_ctx) { set_err("spec_generate: context overflow"); return 1; }
    if (prompt[0] == 7) { set_err("stub: injected failure"); return 1; }
    ms_gate(e);
    int eos1 = cfg[20], eos2 = cfg[21];
    int snap_on = n_cfg >= 44 ? cfg[42] : 0, snap_tok = n_cfg >= 44 ? cfg[43] : -1;
    int frontier = n_cfg >= 47 ? cfg[46] : 0;

    /* prompt reuse: live state, snapshot, or fresh */
    int mode = 0, cached = 0;
    int l = lcp(e->hist, e->nhist, prompt, n_prompt);
    if (frontier) {                                       /* refused: nothing changes, the live sequence stays */
        if (!snap_on || !e->live_ok || e->nhist != n_prompt - 1 || l != e->nhist) {
            set_err("spec_generate: the frontier form needs snap = 1 and the live sequence = the prompt without its last token"); return 1;
        }
        if (prompt[n_prompt - 1] != e->front) {
            set_err("spec_generate: the frontier form: the last prompt token is not the token the stream emitted last"); return 1;
        }
        mode = 1; cached = n_prompt - 1;
    }
    else if (snap_on && e->live_ok && e->nhist > 0 && l == e->nhist) { mode = 1; cached = l; }
    else if (snap_on && e->nsnap > 0 && lcp(e->snap, e->nsnap, prompt, n_prompt) == e->nsnap) { mode = 2; cached = e->nsnap; }

    /* the output of this call: a scripted one, the stand-in model (stub_model), or the default */
    script_t sc = { 0, 0, -1 };
    int use_model = 0;
    pthread_mutex_lock(&g_mu);
    if (g_qhead != g_qtail) sc = g_queue[g_qhead++ % 64];
    else if (e->model_on) use_model = 1;
    else {                                                /* STUB_GEN_DIFF_AT=k: from id k on, the default output is another one */
        int diff_at = env_i("STUB_GEN_DIFF_AT", -1);
        sc.n = 40; sc.ids = malloc(sizeof(int32_t) * 40); sc.loop_from = -1;
        for (int i = 0; i < 40; i++) sc.ids[i] = (diff_at >= 0 && i >= diff_at ? 2000 : 1000) + i;
    }
    pthread_mutex_unlock(&g_mu);
    smp_t sp = carrier_smp(e);
    uint64_t mh = use_model ? hash_ids(prompt, n_prompt) : 0;
    int mn = n_prompt;
    int k = 0;
    #define NEXT() (use_model ? model_take(e, &mh, &mn, &sp) : \
                    (k < sc.n ? sc.ids[k++] : (sc.loop_from >= 0 && sc.n > 0 ? (k = sc.loop_from, sc.ids[k++]) : EOS_ID)))

    int idx = 0, passes = 0, hit = 0;
    long dr[3] = { 0, 0, 0 }, ac[3] = { 0, 0, 0 };        /* look-ahead counts: 1 + 1 + 2 tokens proposed per pass, the first   */
    static const int pat[4] = { 1, 3, 2, 4 };            /* emitted - 1 of them confirmed, in that order (as the engine's 3..6, 23, 24) */
    if (!frontier) {
        out[idx++] = NEXT();                              /* the prompt pass predicts the first id */
        hit = (out[0] == eos1 || out[0] == eos2);
    }
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
    #undef NEXT
    free(sc.ids);

    memset(stats, 0, sizeof(double) * 48);
    stats[0] = passes + 1; stats[1] = idx; stats[30] = idx - 1; stats[31] = hit; stats[41] = mode; stats[42] = cached;
    stats[44] = env_i("STUB_GEN_WIDE_PASSES", 0); stats[47] = env_i("STUB_GEN_MAX_ROWS", 0);   /* wide passes / widest pass (0.1.4) */
    stats[3] = dr[0]; stats[4] = ac[0]; stats[5] = dr[1]; stats[6] = ac[1]; stats[23] = dr[2]; stats[24] = ac[2];
#ifdef STUB_WIDE
    if (n_cfg >= 46) { stats[58] = env_i("STUB_GEN_BACKOFFS", 0); stats[63] = e->wmask; }
#endif
    times_ms[0] = 0.01 * (n_prompt - cached); times_ms[1] = 0.1 * idx;

    /* the engine's state: prompt + all generated ids except the last one (the frontier; with the frontier form and no id written, the
       last prompt token stays the frontier) */
    int nh = n_prompt + idx - 1;
    int32_t *h = malloc(sizeof(int32_t) * (nh > 0 ? nh : 1));
    memcpy(h, prompt, sizeof(int32_t) * (idx > 0 ? n_prompt : n_prompt - 1));
    if (idx > 1) memcpy(h + n_prompt, out, sizeof(int32_t) * (idx - 1));
    free(e->hist); e->hist = h; e->nhist = nh; e->pos = nh;
    e->front = idx > 0 ? out[idx - 1] : prompt[n_prompt - 1];
    e->live_ok = 1; e->tick++;
    if (snap_on && snap_tok >= 0) {                       /* snapshot right after the last chat-start marker of the prompt */
        int last = -1; for (int i = 0; i < n_prompt; i++) if (prompt[i] == snap_tok) last = i;
        if (last >= 0) {
            free(e->snap); e->nsnap = last + 1; e->snap = malloc(sizeof(int32_t) * e->nsnap); memcpy(e->snap, prompt, sizeof(int32_t) * e->nsnap);
            e->snap_born = e->tick;
        }
    }
    return 0;
}

/* the carrier's live sequence: out4 = position, saved prompt state (its length), live (0/1), live length; ids = the live tokens */
EXPORT int yskn_snap_info(yskn_engine *e, int32_t *out4, int32_t *ids, int max_ids) {
    out4[0] = e->pos; out4[1] = e->nsnap; out4[2] = e->live_ok; out4[3] = e->live_ok ? e->nhist : 0;
    if (ids && e->live_ok) for (int i = 0; i < e->nhist && i < max_ids; i++) ids[i] = e->hist[i];
    return 0;
}

#ifndef STUB_NO_MS
/* ---- multi-stream pass (yskn_ms.h) ------------------------------------------------------------------------------------------- */
static int ms_err(const char *m) { set_err(m); return -1; }
static long long env_ll(const char *k, long long d) { const char *v = getenv(k); return v && v[0] ? atoll(v) : d; }
static void slot_reserve(ms_slot_t *X, int n) {
    if (n > X->cap) { int c = X->cap ? X->cap : 256; while (c < n) c *= 2; X->kv = realloc(X->kv, sizeof(int32_t) * c); X->cap = c; }
}
static void slot_push(ms_slot_t *X, int32_t t) { slot_reserve(X, X->nkv + 1); X->kv[X->nkv++] = t; X->h = roll(X->h, t); }
static void drop_drafts_of(yskn_engine *e, int slot) {
    if (!e->ms_dvalid) return;
    for (int r = 0; r < e->ms_dM; r++) if (e->ms_dlay[r] == slot) { e->ms_dvalid = 0; return; }
}
/* why the pass does not take these settings (the codes of yskn_ms_info out[1]); STUB_MS_UNSUPPORTED=k sets one */
static int ms_reason(void) {
    int o = env_i("STUB_MS_UNSUPPORTED", -1);
    if (o >= 0) return o;
    const char *fp = getenv("YSKN_Q35_KV_FP16"), *k8 = getenv("YSKN_Q35_KV8"), *tb = getenv("YSKN_TB_CAP");
    if ((fp && fp[0] && atoi(fp) == 1) || (k8 && k8[0] && atoi(k8) == 0)) return 1;
    if (tb && tb[0] && atoi(tb) == 0) return 2;
    return 0;
}
static int ms_reason_of(const yskn_engine *e) { int r = ms_reason(); return r ? r : (e->visual_on ? 3 : 0); }
static const char *ms_why(int r) {
    switch (r) {
    case 1: return "the KV is kept in 16 bits; the multi-stream pass is built for KV5 only";
    case 2: return "the KV is kept in 8 bits (YSKN_TB_CAP=0); the multi-stream pass is built for KV5 (YSKN_TB_CAP=1) only";
    case 3: return "image input is on";
    case 4: return "the built-in draft head is loaded";
    default: return "these settings are not the ones the multi-stream pass is built for";
    }
}
/* the rows of a pass: slots in range, filled, not moved out, next to each other, the first row of a slot = its frontier when known */
static int ms_layout(const char *what, yskn_engine *e, int M, const int32_t *slots, const int32_t *toks, int *r0, int *len) {
    for (int s = 0; s < MS_MAX; s++) { r0[s] = -1; len[s] = 0; }
    for (int r = 0; r < M; r++) {
        int s = slots[r];
        if (s < 0 || s >= e->ms_ns) { snprintf(g_err, sizeof g_err, "%s: no such stream slot", what); return -1; }
        ms_slot_t *X = &e->ms[s];
        if (X->out) { snprintf(g_err, sizeof g_err, "%s: slot %d was moved to the carrier (ms_unload); ms_load a stream into it first", what, s); return -1; }
        if (r > 0 && slots[r - 1] == s) { len[s]++; continue; }
        if (r0[s] >= 0) { snprintf(g_err, sizeof g_err, "%s: the rows of a stream must be next to each other", what); return -1; }
        r0[s] = r; len[s] = 1;
        if (!X->filled) { snprintf(g_err, sizeof g_err, "%s: slot %d holds no stream (ms_load one into it first)", what, s); return -1; }
        if (toks && X->front >= 0 && toks[r] != X->front) {
            snprintf(g_err, sizeof g_err, "%s: the first row of slot %d is token %d, but its stream emitted %d last (the first row must be that token)",
                     what, s, toks[r], X->front);
            return -1;
        }
    }
    for (int s = 0; s < e->ms_ns; s++)
        if (r0[s] >= 0 && e->ms[s].nkv + len[s] > e->max_ctx) { snprintf(g_err, sizeof g_err, "%s: context overflow", what); return -1; }
    return 0;
}

EXPORT int yskn_ms_info(yskn_engine *e, int64_t *out, int max_out, int *n_out) {
    if (!n_out || (max_out > 0 && !out)) return ms_err("ms_info: no output array");
    int r = ms_reason_of(e), fmt = r == 1 ? 16 : r == 2 ? 8 : 5;
    long long kvb = env_ll("STUB_MS_KV_BYTES", fmt == 5 ? 22400 : fmt == 8 ? 33280 : 65536);
    long long slotb = env_ll("STUB_MS_SLOT_BYTES", 230LL << 20);
    int64_t v[15] = { 1, r, fmt, fmt == 5 ? 5 : 0, kvb, slotb, e->max_ctx, MS_ROWS, MS_MAX, e->draft, e->draft, e->ms_made, e->ms_ns,
                      196083712, 8 };
    int k = max_out < 0 ? 0 : max_out > 15 ? 15 : max_out;
    for (int i = 0; i < k; i++) out[i] = v[i];
    *n_out = k;
    return 0;
}
/* STUB_MS_BIND_FAIL=1: out of memory after the checks (no binding afterwards) */
EXPORT int yskn_ms_bind2(yskn_engine *e, yskn_engine **eng, int n, int n_light) {
    (void)eng;
    if (n != 0) return ms_err("ms_bind: engines as stream slots are a test-only form (n must be 0; use light slots)");
    if (n_light < 1 || n_light > MS_MAX) return ms_err("ms_bind: 1..64 streams");
    int r = ms_reason_of(e);
    if (r) { snprintf(g_err, sizeof g_err, "ms_bind: %s", ms_why(r)); return -1; }
    ms_gate(e);
    e->live_ok = 0;
    if (env_i("STUB_MS_BIND_FAIL", 0)) { e->ms_ns = 0; return ms_err("ms_bind: out of memory"); }
    smp_t cs = carrier_smp(e);                            /* a new slot starts with the carrier's sampling */
    for (int i = e->ms_made; i < n_light; i++) e->ms[i].smp = cs;
    if (n_light > e->ms_made) e->ms_made = n_light;
    e->ms_ns = n_light;
    for (int s = 0; s < MS_MAX; s++) { e->ms_last_r0[s] = -1; e->ms_last_len[s] = 0; }
    return 0;
}
EXPORT int yskn_ms_opt(yskn_engine *e, int key, int value) {
    if (key != 1) return ms_err("ms_opt: unknown key");
    if (value && (!e->draft || e->ms_ns < 1)) return ms_err("ms_opt 1: needs the drafter loaded (for every engine of the binding) before ms_bind");
    e->ms_drafter = value ? 1 : 0;
    return 0;
}
EXPORT int yskn_ms_load(yskn_engine *e, int slot, yskn_engine *src) {
    if (!src) return ms_err("ms_load: no source engine");
    if (slot < 0 || slot >= e->ms_ns) return ms_err("ms_load: not a light stream slot of the current binding");
    if (src->max_ctx != e->max_ctx) return ms_err("ms_load: the source must be an engine of the same model file, context size and settings in this process");
    if (src->nhist != src->pos) return ms_err("ms_load: stub: the source's tokens are not known (read the request with yskn_spec_generate)");
    if (src->visual_on) return ms_err("ms_load: the source has image input on");
    ms_slot_t *X = &e->ms[slot];
    drop_drafts_of(e, slot);
    slot_reserve(X, src->nhist);
    if (src->nhist) memcpy(X->kv, src->hist, sizeof(int32_t) * src->nhist);
    X->nkv = src->nhist; X->h = hash_ids(X->kv, X->nkv);
    X->smp = carrier_smp(src);                            /* the stream goes on with the sampling it was read with */
    e->ms_last_r0[slot] = -1; e->ms_last_len[slot] = 0;
    X->out = 0; X->src_self = src == e; X->src_tick = e->tick; X->src_known = src->live_ok;
    free(X->src_ids); X->src_ids = 0; X->nsrc = 0;
    if (X->src_known) {
        X->src_ids = malloc(sizeof(int32_t) * (src->nhist > 0 ? src->nhist : 1));
        if (src->nhist) memcpy(X->src_ids, src->hist, sizeof(int32_t) * src->nhist);
        X->nsrc = src->nhist;
    }
    X->load_pos = X->nkv; X->front = X->src_known ? src->front : -1; X->filled = 1;
    e->live_ok = 0; src->live_ok = 0;
    return 0;
}
EXPORT int yskn_ms_unload(yskn_engine *e, int slot, const int32_t *ids, int n) {
    if (e->ms_ns < 1) return ms_err("ms_unload: ms_bind first");
    if (slot < 0 || slot >= e->ms_ns) return ms_err("ms_unload: not a light stream slot of the current binding");
    ms_slot_t *X = &e->ms[slot];
    if (X->out) return ms_err("ms_unload: the slot's stream was moved to this engine already (ms_load a stream into the slot first)");
    if (e->ms_open) return ms_err("ms_unload: the last pass is not committed (call ms_commit first)");
    if (e->visual_on) return ms_err("ms_unload: the carrier has image input on");
    if (!X->filled || X->nkv < 1) return ms_err("ms_unload: the slot holds no tokens");
    if (!ids || n != X->nkv) { snprintf(g_err, sizeof g_err, "ms_unload: ids must be the %d tokens the slot holds (n = its position)", X->nkv); return -1; }
    if (X->src_known && (X->nsrc != X->load_pos || memcmp(X->src_ids, ids, sizeof(int32_t) * X->nsrc)))
        return ms_err("ms_unload: ids do not start with the tokens the slot was loaded with");
    if (memcmp(X->kv + X->load_pos, ids + X->load_pos, sizeof(int32_t) * (n - X->load_pos)))
        return ms_err("ms_unload: ids do not end with the tokens of the slot's passes (ids must be the tokens the slot holds)");
    drop_drafts_of(e, slot);
    e->live_ok = 0;
    if (e->nsnap > 0 && !(X->src_self && X->src_known && e->snap_born <= X->src_tick && e->nsnap <= X->nsrc)) e->nsnap = 0;
    int32_t *h = malloc(sizeof(int32_t) * n); memcpy(h, ids, sizeof(int32_t) * n);
    free(e->hist); e->hist = h; e->nhist = n; e->pos = n; e->tf_rows = 0;
    e->opts[10] = X->smp.temp; e->opts[11] = X->smp.top_p; e->opts[12] = X->smp.top_k; e->opts[13] = (int)X->smp.seed;
    e->opts[16] = X->smp.sampler; e->opts[29] = X->smp.min_p;
    e->ms_last_r0[slot] = -1; e->ms_last_len[slot] = 0;
    X->out = 1;
    if (X->src_known) { e->live_ok = 1; e->front = X->front; } else e->front = -1;
    return 0;
}
EXPORT int yskn_live_drop(yskn_engine *e) { e->live_ok = 0; e->front = -1; return 0; }
EXPORT int yskn_ms_draft(yskn_engine *e, int M, const int32_t *slots, const int32_t *toks, const int32_t *dev, int use_graph) {
    (void)use_graph;
    if (e->ms_ns < 1) return ms_err("ms_draft: ms_bind first");
    if (!e->draft || !e->ms_drafter) return ms_err("ms_draft: needs the drafter (load it before ms_bind, then ms_opt(1, 1))");
    if (M < 1 || M > MS_ROWS) return ms_err("ms_draft: 1..16 rows");
    if (!slots || !toks) return ms_err("ms_draft: no slot or token array");
    if (e->visual_on) return ms_err("ms_draft: image input is on");
    int r0[MS_MAX], len[MS_MAX];
    if (ms_layout("ms_draft", e, M, slots, toks, r0, len)) return -1;
    e->ms_open = 0;                                       /* the pass before can no longer be committed */
    e->live_ok = 0;
    memcpy(e->ms_dtoks, toks, sizeof(int32_t) * M);
    for (int s = 0; s < e->ms_ns; s++) {
        int a = r0[s], L = len[s];
        if (a < 0 || L < 2 || (dev && !dev[a])) continue;
        ms_slot_t *X = &e->ms[s];                         /* the stream's own continuation, wrong from a place on (see stub_ms_draft_mode) */
        int c = e->ms_draft_mode == 1 ? L - 1 : e->ms_draft_mode == 2 ? 0 : (int)(mix64(X->h ^ ((uint64_t)X->nkv * 31u)) % (uint64_t)L);
        uint64_t h = roll(X->h, toks[a]);
        int n = X->nkv + 1;
        for (int i = 0; i < L - 1; i++) {
            int32_t t = model_next(e, h, n, &X->smp);
            if (i == c) t += 1;
            e->ms_dtoks[a + 1 + i] = t; h = roll(h, t); n++;
        }
    }
    memcpy(e->ms_dlay, slots, sizeof(int32_t) * M); e->ms_dM = M; e->ms_dvalid = 1; e->ms_drafts++;
    return 0;
}
EXPORT int yskn_ms_step2(yskn_engine *e, int M, const int32_t *slots, const int32_t *toks, int use_graph, int32_t *out_argmax, int32_t *out_toks,
                         int32_t *top_ids, float *top_vals) {
    (void)use_graph;
    if (e->ms_ns < 1) return ms_err("ms_step: ms_bind first");
    if (M < 1 || M > MS_ROWS) return ms_err("ms_step: 1..16 rows");
    if (!slots) return ms_err("ms_step: no slot array");
    if (e->visual_on) return ms_err("ms_step: image input is on");
    int r0[MS_MAX], len[MS_MAX];
    if (ms_layout("ms_step", e, M, slots, toks, r0, len)) return -1;
    if (!toks && (!e->ms_dvalid || e->ms_dM != M || memcmp(e->ms_dlay, slots, sizeof(int32_t) * M)))
        return ms_err("ms_step: no tokens (give toks, or call ms_draft with the same rows first)");
    if (e->ms_fail_in > 0 && --e->ms_fail_in == 0) return ms_err("ms_step: stub: injected failure");
    int32_t tk[MS_ROWS], am[MS_ROWS];
    memcpy(tk, toks ? toks : e->ms_dtoks, sizeof(int32_t) * M);
    uint64_t salt = (uint64_t)env_ll("STUB_MS_SALT", 0);
    for (int s = 0; s < e->ms_ns; s++) {
        if (r0[s] < 0) continue;
        ms_slot_t *X = &e->ms[s];
        uint64_t h = X->h; int n = X->nkv;
        for (int i = 0; i < len[s]; i++) { h = roll(h, tk[r0[s] + i]); n++; am[r0[s] + i] = model_next(e, h ^ salt, n, &X->smp); }
    }
    sleep_us(g_delay_us);
    if (out_argmax) memcpy(out_argmax, am, sizeof(int32_t) * M);
    if (out_toks) memcpy(out_toks, tk, sizeof(int32_t) * M);
    for (int r = 0; r < M; r++) for (int j = 0; j < MS_TOP; j++) {
        if (top_ids) top_ids[r * MS_TOP + j] = am[r] + j;
        if (top_vals) top_vals[r * MS_TOP + j] = 10.0f - (float)j;
    }
    for (int s = 0; s < MS_MAX; s++) { e->ms_last_r0[s] = r0[s]; e->ms_last_len[s] = len[s]; }
    for (int s = 0; s < e->ms_ns; s++) if (r0[s] >= 0) { slot_push(&e->ms[s], tk[r0[s]]); e->ms[s].front = am[r0[s]]; }   /* the first row */
    memcpy(e->ms_last_toks, tk, sizeof(int32_t) * M); memcpy(e->ms_last_am, am, sizeof(int32_t) * M);
    e->ms_dvalid = 0; e->ms_open = 1; e->live_ok = 0; e->ms_steps++;
    return 0;
}
EXPORT int yskn_ms_commit(yskn_engine *e, int n, const int32_t *slots, const int32_t *n_accept) {
    if (!e->ms_open) return ms_err("ms_commit: no pass to commit (call it once, right after ms_step)");
    if (n < 0 || n > e->ms_ns || (n > 0 && (!slots || !n_accept))) return ms_err("ms_commit: bad stream list");
    for (int i = 0; i < n; i++) {
        int s = slots[i], a = n_accept[i];
        if (s < 0 || s >= e->ms_ns || e->ms_last_r0[s] < 0) return ms_err("ms_commit: the stream had no row in the last pass");
        for (int q = 0; q < i; q++) if (slots[q] == s) return ms_err("ms_commit: a stream is listed twice");
        if (a < 0 || a >= e->ms_last_len[s]) return ms_err("ms_commit: more accepted rows than the stream had drafts");
        int a0 = e->ms_last_r0[s];
        for (int j = 0; j < a; j++) if (e->ms_last_toks[a0 + j + 1] != e->ms_last_am[a0 + j]) return ms_err("ms_commit: a draft the pass did not confirm");
    }
    for (int i = 0; i < n; i++) {
        int s = slots[i], a = n_accept[i], a0 = e->ms_last_r0[s];
        if (a == 0) continue;
        for (int j = 1; j <= a; j++) slot_push(&e->ms[s], e->ms_last_toks[a0 + j]);
        e->ms[s].front = e->ms_last_am[a0 + a];
    }
    e->ms_open = 0;
    return 0;
}
EXPORT int yskn_ms_flush(yskn_engine *e) { ms_gate(e); return 0; }
EXPORT int yskn_ms_set_sampling(yskn_engine *e, int slot, int temperature, int top_p, int top_k, int min_p, int sampler, uint32_t seed) {
    if (e->ms_ns < 1) return ms_err("ms_set_sampling: ms_bind first");
    if (slot < 0 || slot >= e->ms_ns) return ms_err("ms_set_sampling: no such stream slot");
    if (top_p <= 0 || top_p > 1000) return ms_err("ms_set_sampling: top-p x 1000 must be 1..1000");
    if (top_k < 1 || top_k > 20) return ms_err("ms_set_sampling: top-k must be 1..20");
    if (min_p < 0 || min_p > 1000) return ms_err("ms_set_sampling: min-p x 1000 must be 0..1000");
    if (sampler != 2 && sampler != 3) return ms_err("ms_set_sampling: the sampler must be 2 or 3");
    smp_t s = { temperature, top_p, top_k, min_p, sampler, seed };
    e->ms[slot].smp = s;
    return 0;
}
EXPORT int yskn_ms_slot_pos(yskn_engine *e, int slot, int *pos) {
    if (slot < 0 || slot >= e->ms_ns) return ms_err("ms_slot_pos: no such stream slot");
    if (!pos) return ms_err("ms_slot_pos: no output");
    *pos = e->ms[slot].nkv;
    return 0;
}
/* the values yskn_state_hash gives for the carrier holding the slot's tokens */
EXPORT int yskn_ms_slot_hash(yskn_engine *e, int slot, uint64_t *out, int max_out, int *n_out) {
    if (slot < 0 || slot >= e->ms_ns) return ms_err("ms_slot_hash: no such stream slot");
    if (!n_out || (max_out > 0 && !out)) return ms_err("ms_slot_hash: no output array");
    if (max_out < 2) return ms_err("ms_slot_hash: output array too small");
    out[0] = (uint64_t)e->ms[slot].nkv; out[1] = fnv(e->ms[slot].kv, e->ms[slot].nkv); *n_out = 2;
    return 0;
}
#endif

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
