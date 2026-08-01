// mz_generate: verify runtime KV chunk growth and chunk-boundary-crossing
// decodes.
//
//   mz_generate <model.gguf> [n_prompt=300] [n_gen=600] [swap_ratio=0.5]
//
// Phases:
//   A. Prefill n_prompt tokens in one batch that crosses a 256-token chunk
//      boundary (decode_chunk_aligned splits it; apply_ubatch materializes
//      chunk 1+ on demand). Cross-checked by re-decoding the tail token by
//      token and comparing logits.
//   B. Greedy-generate n_gen tokens, crossing several more boundaries; each
//      crossing must grow the chunk set. Re-generates the same range a second
//      time (chunks already grown) and requires the identical token sequence.
//   C. (swap_ratio >= 0) swapout to the given ratio — the grown chunks are
//      part of the plan via on_kv_chunks_grown — then swapin_generate with a
//      follow batch, and compare a K-row snapshot of a grown chunk across the
//      swapout/swapin round trip.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include "llama-model.h"
#include "llama-kv-cache-unified.h"

#include "mzcache_types.h"
#include "mzcache_globals.h"
#include "mzcache_kv_state.h"
#include "mzcache_core.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...)                                          \
    do {                                                          \
        if (cond) {                                               \
            printf("[OK]   " __VA_ARGS__);                        \
            printf("\n");                                         \
        } else {                                                  \
            printf("[FAIL] " __VA_ARGS__);                        \
            printf("\n");                                         \
            g_failures++;                                         \
        }                                                         \
    } while (0)

static std::vector<float> logits_snapshot(llama_context * ctx, int n_vocab) {
    const float * l = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(l, l + n_vocab);
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static int argmax(const std::vector<float> & v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) best = (int) i;
    }
    return best;
}

// Sum of |x| over one K row (n_embd_k_gqa elements) of chunk `ci` at
// chunk-local row `row`, read straight from the SVM chunk table.
static double k_row_abs_sum(int il, int ci, int row, int hdim) {
    const __fp16 * base = (const __fp16 *) g_svm_chunk_ptrs[2 * il + 0]->c[ci];
    if (!base) return -1.0;
    double s = 0.0;
    for (int e = 0; e < hdim; ++e) {
        s += std::fabs((double) base[(size_t) row * hdim + e]);
    }
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [n_prompt=300] [n_gen=600] [swap_ratio=0.5 (-1: skip)]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const int   n_prompt   = (argc > 2) ? atoi(argv[2]) : 300;
    const int   n_gen      = (argc > 3) ? atoi(argv[3]) : 600;
    const float swap_ratio = (argc > 4) ? (float) atof(argv[4]) : 0.5f;

    common_params params;
    params.sampling.seed = 1234;
    params.use_mmap      = false;
    params.n_ctx         = n_prompt + n_gen + 64; // generous headroom: lazy chunks make it free
    params.n_batch       = std::max(512, n_prompt);
    params.n_ubatch      = 256; // FA kernel tiles queries in its launch grid; 256 = one KV chunk (write ceiling)
    params.warmup        = false;
    params.n_gpu_layers  = 100;
    params.flash_attn    = true;
    params.model.path    = model_path;

    common_init();
    auto init_res = common_init_from_params(params);
    llama_model *   model = init_res.model.get();
    llama_context * ctx   = init_res.context.get();
    if (!model || !ctx) {
        fprintf(stderr, "ERROR: model/context init failed\n");
        return 1;
    }

    const llama_vocab * vocab  = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto * kv_cache = static_cast<llama_kv_cache_unified *>(llama_get_memory(ctx));
    const int n_layers = (int) kv_cache->layers_k_chunks.size();
    const int hdim     = model->hparams.n_embd_k_gqa(n_layers - 1);

    printf("=== mz_generate: n_prompt=%d n_gen=%d n_ctx=%u swap_ratio=%.2f ===\n",
           n_prompt, n_gen, llama_n_ctx(ctx), (double) swap_ratio);

    CHECK(kv_cache->n_chunks_alloc == 1,
          "lazy init: only chunk 0 materialized at startup (n_chunks_alloc=%u)", kv_cache->n_chunks_alloc);

    // ---- build an n_prompt-token prompt (BOS + repeated content) ----
    std::string text = "The quick brown fox jumps over the lazy dog while the curious cat watches from the window. ";
    std::vector<llama_token> seed_tok(512);
    int n_seed = llama_tokenize(vocab, text.c_str(), text.size(), seed_tok.data(), seed_tok.size(), true, true);
    if (n_seed <= 1) { fprintf(stderr, "tokenize failed\n"); return 1; }

    std::vector<llama_token> prompt;
    prompt.push_back(seed_tok[0]); // BOS
    for (int i = 1; (int) prompt.size() < n_prompt; i = (i + 1 <= n_seed - 1) ? i + 1 : 1) {
        prompt.push_back(seed_tok[i]);
    }

    // =====================================================================
    // Phase A: single-batch prefill crossing a chunk boundary
    // =====================================================================
    printf("\n--- Phase A: batch prefill crossing the %d-token boundary ---\n", TOKENS_PER_CHUNK);

    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        common_batch_add(batch, prompt[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    int ret = mzcache::decode_chunk_aligned(ctx, batch);
    CHECK(ret == 0, "batch prefill of %d tokens decoded (ret=%d)", n_prompt, ret);

    const uint32_t chunks_after_prefill = (n_prompt + TOKENS_PER_CHUNK - 1) / TOKENS_PER_CHUNK;
    CHECK(kv_cache->n_chunks_alloc == chunks_after_prefill,
          "chunks grew to cover the prompt (n_chunks_alloc=%u, expected %u)",
          kv_cache->n_chunks_alloc, chunks_after_prefill);

    // written rows are nonzero, unwritten rows still zero (chunks are memset)
    const int last_ci  = (n_prompt - 1) / TOKENS_PER_CHUNK;
    const int last_row = (n_prompt - 1) % TOKENS_PER_CHUNK;
    CHECK(k_row_abs_sum(n_layers - 1, last_ci, last_row, hdim) > 0.0,
          "K row of the last prompt token (chunk %d row %d) is populated", last_ci, last_row);
    if (last_row + 1 < TOKENS_PER_CHUNK) {
        CHECK(k_row_abs_sum(n_layers - 1, last_ci, last_row + 1, hdim) == 0.0,
              "K row past the prompt (chunk %d row %d) is still zero", last_ci, last_row + 1);
    }

    std::vector<float> logits_batch = logits_snapshot(ctx, n_vocab);

    // cross-check: drop the tokens of the crossing chunk and re-decode them
    // one by one — same cells, same logits expected (up to FA batching FP noise)
    {
        const int redo_from = TOKENS_PER_CHUNK * ((n_prompt - 1) / TOKENS_PER_CHUNK);
        llama_memory_seq_rm(llama_get_memory(ctx), 0, redo_from, -1);

        llama_batch one = llama_batch_init(1, 0, 1);
        for (int i = redo_from; i < n_prompt; ++i) {
            common_batch_clear(one);
            common_batch_add(one, prompt[i], i, {0}, true);
            ret = mzcache::decode_chunk_aligned(ctx, one);
            if (ret != 0) break;
        }
        llama_batch_free(one);
        CHECK(ret == 0, "token-wise re-decode of positions %d..%d (ret=%d)", redo_from, n_prompt - 1, ret);

        std::vector<float> logits_tokenwise = logits_snapshot(ctx, n_vocab);
        const float diff = max_abs_diff(logits_batch, logits_tokenwise);
        CHECK(argmax(logits_batch) == argmax(logits_tokenwise) && diff < 0.5f,
              "batch vs token-wise logits agree (max_abs_diff=%.4f, same argmax=%d)",
              (double) diff, argmax(logits_batch));
    }

    // =====================================================================
    // create the mzcache core NOW so phase B growth exercises the
    // on_kv_chunks_grown bookkeeping path
    // =====================================================================
    mzcache_core * core = new mzcache_core(n_layers, hdim, (int) kv_cache->n_chunks_alloc, model);

    // =====================================================================
    // Phase B: long greedy generation across several chunk boundaries
    // =====================================================================
    printf("\n--- Phase B: greedy generation of %d tokens ---\n", n_gen);

    llama_sampler * sampler = llama_sampler_init_greedy();

    std::vector<llama_token> gen_tokens;
    gen_tokens.reserve(n_gen);
    int n_past = n_prompt;

    llama_batch one = llama_batch_init(1, 0, 1);
    llama_token tok = llama_sampler_sample(sampler, ctx, -1);
    std::string gen_text;

    for (int i = 0; i < n_gen; ++i) {
        gen_tokens.push_back(tok);
        gen_text += common_token_to_piece(ctx, tok);

        common_batch_clear(one);
        common_batch_add(one, tok, n_past, {0}, true);
        ret = mzcache::decode_chunk_aligned(ctx, one);
        if (ret != 0) {
            fprintf(stderr, "decode failed at gen step %d (pos %d), ret=%d\n", i, n_past, ret);
            break;
        }
        n_past++;
        tok = llama_sampler_sample(sampler, ctx, -1);
    }

    CHECK(ret == 0, "generated %d tokens without decode errors (final n_past=%d)", (int) gen_tokens.size(), n_past);

    const uint32_t chunks_after_gen = (n_past + TOKENS_PER_CHUNK - 1) / TOKENS_PER_CHUNK;
    CHECK(kv_cache->n_chunks_alloc == chunks_after_gen,
          "chunks grew with generation (n_chunks_alloc=%u, expected %u)",
          kv_cache->n_chunks_alloc, chunks_after_gen);

    // a row written right after a boundary crossing lives at row 0 of a grown chunk
    if (chunks_after_gen > chunks_after_prefill) {
        const int ci = (int) chunks_after_prefill; // first chunk grown during generation
        CHECK(k_row_abs_sum(n_layers - 1, ci, 0, hdim) > 0.0,
              "first row of generation-grown chunk %d is populated", ci);
    }

    printf("generated text (first 200 chars): %.200s%s\n",
           gen_text.c_str(), gen_text.size() > 200 ? "..." : "");

    // determinism cross-check: rewind to n_prompt and regenerate — the chunks
    // are already grown this time, so the non-growth path must reproduce the
    // exact same tokens
    {
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prompt, -1);

        // recompute the last prompt token's logits to re-seed sampling
        common_batch_clear(one);
        common_batch_add(one, prompt[n_prompt - 1], n_prompt - 1, {0}, true);
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prompt - 1, -1);
        ret = mzcache::decode_chunk_aligned(ctx, one);

        int mismatch = -1;
        int pos = n_prompt;
        llama_token t2 = llama_sampler_sample(sampler, ctx, -1);
        for (size_t i = 0; i < gen_tokens.size() && ret == 0; ++i) {
            if (t2 != gen_tokens[i]) { mismatch = (int) i; break; }
            common_batch_clear(one);
            common_batch_add(one, t2, pos, {0}, true);
            ret = mzcache::decode_chunk_aligned(ctx, one);
            pos++;
            t2 = llama_sampler_sample(sampler, ctx, -1);
        }
        CHECK(ret == 0 && mismatch < 0,
              "regeneration over pre-grown chunks reproduces the sequence (mismatch at %d)", mismatch);
        n_past = pos;
    }

    // =====================================================================
    // Phase C: swapout / swapin with the grown chunks in the plan
    // =====================================================================
    if (swap_ratio >= 0.0f) {
        printf("\n--- Phase C: swapout(%.2f) + swapin_generate over grown chunks ---\n", (double) swap_ratio);

        // snapshot a K row of a generation-grown chunk before swapout
        const int probe_ci = (int) chunks_after_prefill;
        std::vector<__fp16> k_before(hdim);
        {
            const __fp16 * base = (const __fp16 *) g_svm_chunk_ptrs[2 * (n_layers - 1) + 0]->c[probe_ci];
            memcpy(k_before.data(), base, hdim * sizeof(__fp16));
        }

        auto [achieved, comp, store, unload] = core->swapout(swap_ratio);
        printf("swapout achieved %.3f (+comp %d, +store %d, +unload %d)\n",
               (double) achieved, comp, store, unload);
        CHECK(achieved <= swap_ratio + 0.05f, "swapout reached the target ratio (achieved %.3f)", (double) achieved);

        // follow batch: 8 tokens continuing the generation; if it crosses a
        // boundary, swapin_generate's internal chunk-aligned split handles it
        llama_batch follow = llama_batch_init(8, 0, 1);
        int pos = n_past;
        for (int i = 0; i < 8; ++i, ++pos) {
            common_batch_add(follow, gen_tokens[i % gen_tokens.size()], pos, {0}, false);
        }
        follow.logits[follow.n_tokens - 1] = true;
        printf("follow batch spans positions %d..%d (chunk %d..%d)\n",
               n_past, pos - 1, n_past / TOKENS_PER_CHUNK, (pos - 1) / TOKENS_PER_CHUNK);

        long ttft_us = core->swapin_generate(ctx, follow, false);
        CHECK(ttft_us > 0, "swapin_generate completed (TTFT %.1f ms)", ttft_us / 1000.0);

        const float * l = llama_get_logits_ith(ctx, -1);
        bool finite = true;
        for (int i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(l[i])) { finite = false; break; }
        }
        CHECK(finite, "post-swapin logits are finite");

        // restored K row must match the pre-swapout snapshot bit-exactly if the
        // probe chunk was stored/kept raw; quantization error bound if compressed
        {
            const __fp16 * base = (const __fp16 *) g_svm_chunk_ptrs[2 * (n_layers - 1) + 0]->c[probe_ci];
            CHECK(base != nullptr, "grown chunk %d is resident again after swapin", probe_ci);
            if (base) {
                double max_err = 0.0;
                for (int e = 0; e < hdim; ++e) {
                    max_err = std::max(max_err, std::fabs((double) base[e] - (double) k_before[e]));
                }
                CHECK(max_err < 0.25,
                      "restored K row of grown chunk %d matches (max_err=%.4f)", probe_ci, max_err);
            }
        }

        llama_batch_free(follow);
    }

    llama_batch_free(one);
    llama_batch_free(batch);
    llama_sampler_free(sampler);

    printf("\n=== mz_generate: %s (%d failure%s) ===\n",
           g_failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
