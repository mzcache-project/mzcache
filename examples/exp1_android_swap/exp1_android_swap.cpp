// exp1_android_swap — CPU-only llama.cpp baseline ("android swap").
//
// Reproduces the baseline-test `mz_load_state_and_inference` measurement using
// only stock llama.cpp public APIs: load a saved KV state (stock llama_state
// blob), then process a follow-up prompt and greedily generate, reporting the
// prefill / TTFT timing. Built CPU-only (GGML_OPENCL=OFF) with the token
// embedding pinned in RAM via -DMZ_MLOCK_TOKEN_EMBD=ON (see cpu_build.sh); the
// rest of the model/KV swaps naturally under Android memory pressure.
//
// The .kv is a stock `llama_state` session blob and is reused as-is. The
// build's flash_attn must match the file's `_fa`-ness (fa file <-> flash_attn
// ON), cache dtype F16/F16, and n_ctx >= the file's token count. By default we
// load states/qwen3_0.6B_8193.kv (non-fa, flash_attn OFF).

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <cstdio>
#include <cstdlib>

// Deterministic state path (Qwen3-0.6B reproduction). Uses only common_params +
// public GGML types — no internal llama headers. Override with --prompt-cache.
static std::string make_state_path(const common_params & params, int ctx_size) {
    std::string n = "states/qwen3_0.6B";
    if (params.cache_type_k == GGML_TYPE_BF16) n += "_bf16";
    else if (params.cache_type_k == GGML_TYPE_Q8_0) n += "_q8_0";
    else if (params.cache_type_k == GGML_TYPE_Q4_0) n += "_q4_0";
    if (params.flash_attn) n += "_fa";
    n += "_" + std::to_string(ctx_size) + ".kv";
    return n;
}

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed = 1234;
    params.n_ctx     = 8250;
    params.n_predict  = 8;
    params.n_batch   = 8250;
    params.n_ubatch  = 8;
    params.warmup    = true;

    llama_log_set([](enum ggml_log_level level, const char * text, void * /*u*/) {
        if (level >= GGML_LOG_LEVEL_DEBUG) fprintf(stderr, "%s", text);
    }, nullptr);

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    // A saved full-context state restores an output id at the last token position;
    // llama_state_load_file validates that id against n_batch, so the logical batch
    // must span the whole context — otherwise loading a >n_batch-token state fails
    // with "invalid output id ... does not fit in batch size". (n_ubatch stays small.)
    if (params.n_batch < params.n_ctx) {
        params.n_batch = params.n_ctx;
    }
    common_init();

    auto init_res = common_init_from_params(params);
    llama_model *   model = init_res.model.get();
    llama_context * ctx   = init_res.context.get();
    if (!model || !ctx) {
        fprintf(stderr, "ERROR: model/context initialization failed\n");
        return 1;
    }

    // State file selection, in order of precedence:
    //   1. ANDSW_STATE=<path>  — env override used by the ctx-size sweep scripts,
    //   2. --prompt-cache <path> — only if the build registers it for this example,
    //   3. the default non-fa qwen3_0.6B_8193 path.
    std::string state_name;
    if (const char * env = std::getenv("ANDSW_STATE"); env && *env) {
        state_name = env;
    } else if (!params.path_prompt_cache.empty()) {
        state_name = params.path_prompt_cache;
    } else {
        state_name = make_state_path(params, 8193);
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    llama_sampler *     sampler = llama_sampler_init_greedy();

    std::vector<llama_token> session_tokens;
    session_tokens.reserve(params.n_ctx);
    int    n_past        = 0;
    size_t n_token_count = 0;

    if (!std::filesystem::exists(state_name)) {
        fprintf(stderr, "ERROR: no state file at %s\n", state_name.c_str());
        fprintf(stderr, "Pass one via --prompt-cache <path>, or create it first.\n");
        return 1;
    }

    std::vector<llama_token> tmp(params.n_ctx);
    auto t_load_start = std::chrono::high_resolution_clock::now();
    if (!llama_state_load_file(ctx, state_name.c_str(), tmp.data(), tmp.size(), &n_token_count)) {
        fprintf(stderr, "ERROR: failed to load state %s\n", state_name.c_str());
        return 1;
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();
    printf("[Timing] Loaded %s in %.2f ms (%zu tokens)\n", state_name.c_str(),
           std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count(),
           n_token_count);

    session_tokens.assign(tmp.begin(), tmp.begin() + n_token_count);
    n_past = (int) n_token_count;

    std::string user_input;
    while (true) {
        std::cout << "[User]> ";
        std::getline(std::cin, user_input);
        if (!std::cin || user_input == "exit") break;
        if (user_input == "ss") {
            if (!llama_state_save_file(ctx, state_name.c_str(), session_tokens.data(), n_past)) {
                fprintf(stderr, "ERROR: failed to save session\n");
            } else {
                printf("[Info] session saved with %d tokens\n", n_past);
            }
            continue;
        }

        auto t_input_done = std::chrono::high_resolution_clock::now();

        int n_tok = -llama_tokenize(vocab, user_input.c_str(), user_input.size(), nullptr, 0, true, true);
        std::vector<llama_token> follow_tokens(n_tok);
        llama_tokenize(vocab, user_input.c_str(), user_input.size(),
                       follow_tokens.data(), follow_tokens.size(), true, true);

        llama_batch batch = llama_batch_init(follow_tokens.size(), 0, 1);
        int32_t pos = n_past;
        for (size_t i = 0; i < follow_tokens.size(); ++i, ++pos) {
            common_batch_add(batch, follow_tokens[i], pos, {0}, false);
        }
        const int n_ctx = llama_n_ctx(ctx);
        const int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            fprintf(stderr, "context size exceeded\n");
            llama_batch_free(batch);
            break;
        }
        batch.logits[batch.n_tokens - 1] = true;

        auto t_decode_start = std::chrono::high_resolution_clock::now();
        llama_decode(ctx, batch);
        auto t_decode_done = std::chrono::high_resolution_clock::now();
        printf("[Timing] TTFT1 %.2f ms\n",
               std::chrono::duration<double, std::milli>(t_decode_done - t_decode_start).count());
        printf("[Timing] TTFT2 %.2f ms\n",
               std::chrono::duration<double, std::milli>(t_decode_done - t_input_done).count());

        int tok = llama_sampler_sample(sampler, ctx, -1);
        std::cout << common_token_to_piece(ctx, tok) << std::endl;
        const float * logits = llama_get_logits(ctx);
        for (int i = 0; i < 20; ++i) printf("logits[%d] = %f\n", i, logits[i]);

        n_past += batch.n_tokens;
        session_tokens.insert(session_tokens.end(), follow_tokens.begin(), follow_tokens.end());

        for (int i = 0; i < params.n_predict; ++i) {
            tok = llama_sampler_sample(sampler, ctx, -1);
            session_tokens.push_back(tok);
            if (tok == llama_vocab_eos(vocab)) break;
            common_batch_clear(batch);
            common_batch_add(batch, tok, n_past, {0}, true);
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s: decode failed\n", __func__);
                break;
            }
            n_past += 1;
            std::cout << common_token_to_piece(ctx, tok);
        }
        std::cout << std::endl;
        llama_batch_free(batch);
    }

    llama_sampler_free(sampler);
    return 0;
}
