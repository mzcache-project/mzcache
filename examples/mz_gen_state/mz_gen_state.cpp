// mz_gen_state — server-side prefill-state (.kv) generator.
//
// Port of new_llama's test_store_n_tokens, made non-interactive: decodes the
// first <n_tokens> tokens of a text corpus and saves the llama_state to
// ./states/<model_tag>[_fa]_<n_tokens>.kv (e.g. qwen3_0.6B_fa_8193.kv), the
// exact files the on-device runners consume (see EVALUATION.md §0/§1.3).
//
// Decoding parameters are pinned to the original tool (n_ctx/n_batch 65536,
// n_ubatch 512, seed 1234) so states are bit-identical to those produced by
// test_store_n_tokens on the same GPU + corpus.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-model.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

static char * copy_cstr(const std::string & s) {
    char * p = (char *) std::malloc(s.size() + 1);
    std::memcpy(p, s.data(), s.size());
    p[s.size()] = '\0';
    return p;
}

static std::string set_state_name(llama_model * model, const common_params & params, int token_count) {
    std::string state_name;

    switch (model->arch) {
        case LLM_ARCH_LLAMA:   state_name += "llama3";   break;
        case LLM_ARCH_QWEN3:   state_name += "qwen3";    break;
        case LLM_ARCH_GEMMA3:  state_name += "gemma3";   break;
        case LLM_ARCH_EXAONE4: state_name += "exaone4";  break;
        default:               state_name += "unknown";  break;
    }

    state_name += "_" + model->type_name();

    if (params.cache_type_k == GGML_TYPE_BF16) {
        state_name += "_bf16";
    } else if (params.cache_type_k == GGML_TYPE_Q8_0) {
        state_name += "_q8_0";
    } else if (params.cache_type_k == GGML_TYPE_Q4_0) {
        state_name += "_q4_0";
    }

    if (params.flash_attn) {
        state_name += "_fa";
    }

    if (token_count > 0) {
        state_name += "_" + std::to_string(token_count);
    }

    state_name += ".kv";
    return state_name;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "Usage: %s <model.gguf> <corpus.txt> <n_tokens> [fa]\n"
            "Example: %s Qwen3-0.6B-FP16.gguf wikitext.txt 8193 fa\n"
            "Writes ./states/<model_tag>[_fa]_<n_tokens>.kv\n",
            argv[0], argv[0]);
        return 1;
    }

    const std::string model_path  = argv[1];
    const std::string corpus_path = argv[2];
    const int n_tokens = std::atoi(argv[3]);
    const bool use_fa = (argc >= 5) && (std::string(argv[4]) == "fa");

    if (n_tokens <= 0) {
        std::fprintf(stderr, "ERROR: n_tokens must be > 0 (got %s)\n", argv[3]);
        return 1;
    }

    // Same decode configuration as new_llama's test_store_n_tokens.
    common_params params;
    params.sampling.seed = 1234;
    params.n_ctx    = 65536;
    params.n_predict = 32;
    params.n_batch  = 65536;
    params.n_ubatch = 512;

    {
        std::vector<char *> fixed_argv;
        fixed_argv.push_back(copy_cstr(argv[0]));
        fixed_argv.push_back(copy_cstr(std::string("--model")));
        fixed_argv.push_back(copy_cstr(model_path));
        if (use_fa) {
            fixed_argv.push_back(copy_cstr(std::string("-fa")));
        }

        int fixed_argc = (int) fixed_argv.size();
        if (!common_params_parse(fixed_argc, fixed_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
            std::fprintf(stderr, "ERROR: common_params_parse failed\n");
            return 1;
        }
        for (char * p : fixed_argv) std::free(p);
    }

    common_init();

    auto init_res = common_init_from_params(params);
    llama_model *   model = init_res.model.get();
    llama_context * ctx   = init_res.context.get();
    if (!model || !ctx) {
        std::fprintf(stderr, "ERROR: model/context initialization failed\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::ifstream fin(corpus_path);
    std::string all_text{
        std::istreambuf_iterator<char>(fin),
        std::istreambuf_iterator<char>()
    };
    if (all_text.empty()) {
        std::cerr << "ERROR: " << corpus_path << " is empty or not found\n";
        return 1;
    }

    int n_all = -llama_tokenize(vocab, all_text.c_str(), all_text.size(),
                                nullptr, 0, /*add_bos*/ true, /*special*/ true);
    std::cout << "[Info] Total tokens in " << corpus_path << ": " << n_all << std::endl;
    if (n_tokens > n_all) {
        std::fprintf(stderr, "ERROR: corpus has only %d tokens (requested %d)\n", n_all, n_tokens);
        return 1;
    }

    std::vector<llama_token> all_tokens(n_all);
    llama_tokenize(vocab, all_text.c_str(), all_text.size(),
                   all_tokens.data(), n_all, true, true);

    std::vector<llama_token> session_tokens(all_tokens.begin(), all_tokens.begin() + n_tokens);

    std::filesystem::create_directories("./states");
    const std::string state_path = "./states/" + set_state_name(model, params, n_tokens);

    llama_batch batch = llama_batch_init(session_tokens.size(), 0, /*n_seq_max*/ 1);
    for (size_t i = 0; i < session_tokens.size(); ++i) {
        common_batch_add(batch, session_tokens[i], /*pos*/ i, {0}, /*logits*/ false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    auto t0 = std::chrono::high_resolution_clock::now();
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "ERROR: decode failed\n";
        llama_batch_free(batch);
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[Info] Decoded %d tokens in %.2f ms\n", n_tokens, elapsed_ms);

    const float * logits = llama_get_logits(ctx);
    for (int i = 0; i < 5; ++i) {
        printf("logits[%d] = %f\n", i, logits[i]);
    }
    std::cout << "--------------------------------\n";

    int ret = 0;
    if (!llama_state_save_file(ctx, state_path.c_str(),
                               session_tokens.data(), session_tokens.size())) {
        std::cerr << "ERROR: failed to save state to " << state_path << "\n";
        ret = 1;
    } else {
        std::cout << "[Info] Session saved with " << n_tokens
                  << " tokens -> " << state_path << "\n";
    }

    llama_batch_free(batch);
    return ret;
}
