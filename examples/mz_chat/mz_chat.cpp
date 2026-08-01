// mz_chat: interactive multi-turn chat with mid-conversation mzcache swapout.
//
//   mz_chat <model.gguf> [n_ctx=32768]
//
// Chat as usual; two inputs are special:
//   /swap <delta>   trim the resident ratio by <delta> (e.g. "/swap 0.15"
//                   goes 1.0 -> 0.85; a second "/swap 0.15" -> 0.70). The
//                   conversation KV/weights are compressed/stored/unloaded
//                   in place — nothing is decoded yet.
//   (empty line)    quit
//
// The first chat turn after one or more /swap commands is decoded through
// swapin_generate: the swapped-out state is restored concurrently with the
// prompt prefill, so the conversation continues seamlessly and the reply
// prints a TTFT that includes the restore.
//
// Prompt and generation decodes go through mzcache::decode_chunk_aligned, so
// turns of any length are split at 256-token KV-chunk boundaries and chunks
// grow on demand as the conversation gets longer (n_ubatch=256 — the chunked
// flash-attn kernel tiles queries over its launch grid).
//
// Run from a directory that has the per-layer weight dumps in ./layers/
// (<model>_layer_<n>.bin) and is writable: the weight reload workers read
// them and the store path writes KV_<model>_L<n>.bin, both relative to the
// cwd (on device: /data/local/tmp/mzcache). Without readable dumps swapout
// refuses to unload weights and stops at the ratio reached by compression
// alone — the chat keeps working, just with a shallower trim.

#include "common.h"
#include "llama.h"

#include "llama-model.h"
#include "llama-kv-cache-unified.h"

#include "mzcache_types.h"
#include "mzcache_core.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Returns true if `line` is a swapout command; on success stores the parsed
// ratio delta in `delta`, on parse failure stores a negative value.
static bool parse_swap_command(const std::string & line, float & delta) {
    if (line.rfind("/swap", 0) != 0) {
        return false;
    }
    delta = -1.0f;
    const char * arg = line.c_str() + strlen("/swap");
    char * end = nullptr;
    const float v = strtof(arg, &end);
    if (end != arg && v > 0.0f && v <= 1.0f) {
        delta = v;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [n_ctx=32768]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const int n_ctx = (argc > 2) ? atoi(argv[2]) : 32768; // lazy chunks: unused headroom is free

    common_params params;
    params.use_mmap     = false;
    params.n_ctx        = n_ctx;   // lazy chunks: unused headroom is free
    params.n_batch      = n_ctx;
    params.n_ubatch     = 256;     // one KV chunk — the write ceiling (see docs/INTEGRATION.md #5)
    params.warmup       = false;
    params.n_gpu_layers = 100;
    params.flash_attn   = true;
    params.model.path   = model_path;

    common_init();
    auto init_res = common_init_from_params(params);
    llama_model *   model = init_res.model.get();
    llama_context * ctx   = init_res.context.get();
    if (!model || !ctx) {
        fprintf(stderr, "ERROR: model/context init failed\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);
    if (!tmpl) {
        fprintf(stderr, "ERROR: model has no chat template\n");
        return 1;
    }

    auto * kv_cache = static_cast<llama_kv_cache_unified *>(llama_get_memory(ctx));
    const int n_layers = (int) kv_cache->layers_k_chunks.size();
    const int hdim     = model->hparams.n_embd_k_gqa(n_layers - 1);

    // chunks grown by later turns are folded in via on_kv_chunks_grown
    mzcache_core * core = new mzcache_core(n_layers, hdim, (int) kv_cache->n_chunks_alloc, model);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    printf("\n=== mz_chat: n_ctx=%u, ubatch=%d ===\n", llama_n_ctx(ctx), params.n_ubatch);
    printf("chat normally; \"/swap <delta>\" trims the resident ratio (next turn restores via swapin);\n");
    printf("empty line quits\n\n");

    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(llama_n_ctx(ctx));
    int   prev_len     = 0;
    int   n_past       = 0;
    float cur_ratio    = 1.0f; // resident ratio, trimmed by /swap, back to 1.0 after swapin
    bool  swap_pending = false;
    int   ret          = 0;

    while (true) {
        printf("\033[32m> \033[0m");
        std::string user;
        if (!std::getline(std::cin, user) || user.empty()) {
            break;
        }

        float delta;
        if (parse_swap_command(user, delta)) {
            if (delta <= 0.0f) {
                printf("usage: /swap <delta in (0,1]>, e.g. /swap 0.15\n");
                continue;
            }
            if (n_past == 0) {
                printf("nothing to swap out yet — chat first\n");
                continue;
            }
            const float target = std::max(0.0f, cur_ratio - delta);
            auto [achieved, comp, store, unload] = core->swapout(target);
            cur_ratio    = achieved;
            swap_pending = true;
            printf("swapout: target %.3f, achieved %.3f (+comp %d, +store %d, +unload %d)\n",
                   (double) target, (double) achieved, comp, store, unload);
            continue;
        }

        // format the new user turn; the template diff vs prev_len is the prompt
        messages.push_back({"user", strdup(user.c_str())});
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                                formatted.data(), formatted.size());
        if (new_len > (int) formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                                formatted.data(), formatted.size());
        }
        if (new_len < 0) {
            fprintf(stderr, "ERROR: failed to apply the chat template\n");
            break;
        }
        const std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

        const bool is_first = n_past == 0;
        const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                                    nullptr, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(),
                           prompt_tokens.size(), is_first, true) < 0) {
            fprintf(stderr, "ERROR: failed to tokenize the prompt\n");
            break;
        }

        if (n_past + n_prompt_tokens + 1 > (int) llama_n_ctx(ctx)) {
            fprintf(stderr, "ERROR: context size exceeded\n");
            break;
        }

        llama_batch batch = llama_batch_init(n_prompt_tokens, 0, 1);
        for (int i = 0; i < n_prompt_tokens; ++i) {
            common_batch_add(batch, prompt_tokens[i], n_past + i, {0}, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        if (swap_pending) {
            // restore the swapped-out KV/weights while this prompt prefills
            const long ttft_us = core->swapin_generate(ctx, batch, false);
            swap_pending = false;
            cur_ratio    = 1.0f; // everything resident again
            printf("swapin_generate: TTFT %.1f ms (%d prompt tokens)\n",
                   ttft_us / 1000.0, n_prompt_tokens);
        } else {
            ret = mzcache::decode_chunk_aligned(ctx, batch);
            if (ret != 0) {
                fprintf(stderr, "ERROR: prompt decode failed, ret=%d\n", ret);
                llama_batch_free(batch);
                break;
            }
        }
        llama_batch_free(batch);
        n_past += n_prompt_tokens;

        // token-by-token generation; single-token decodes never cross a
        // chunk boundary, and boundary crossings grow the chunk set
        std::string response;
        llama_batch one = llama_batch_init(1, 0, 1);
        printf("\033[33m");
        while (true) {
            const llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, tok)) {
                break;
            }

            const std::string piece = common_token_to_piece(ctx, tok);
            printf("%s", piece.c_str());
            fflush(stdout);
            response += piece;

            if (n_past + 1 > (int) llama_n_ctx(ctx)) {
                printf("\033[0m\n");
                fprintf(stderr, "ERROR: context size exceeded\n");
                ret = -1;
                break;
            }

            common_batch_clear(one);
            common_batch_add(one, tok, n_past, {0}, true);
            ret = mzcache::decode_chunk_aligned(ctx, one);
            if (ret != 0) {
                printf("\033[0m\n");
                fprintf(stderr, "ERROR: decode failed at pos %d, ret=%d\n", n_past, ret);
                break;
            }
            n_past++;
        }
        printf("\033[0m\n");
        llama_batch_free(one);
        if (ret != 0) {
            break;
        }

        messages.push_back({"assistant", strdup(response.c_str())});
        prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false,
                                             nullptr, 0);
        if (prev_len < 0) {
            fprintf(stderr, "ERROR: failed to apply the chat template\n");
            break;
        }
    }

    for (auto & msg : messages) {
        free(const_cast<char *>(msg.content));
    }
    llama_sampler_free(smpl);
    delete core;
    return ret == 0 ? 0 : 1;
}
