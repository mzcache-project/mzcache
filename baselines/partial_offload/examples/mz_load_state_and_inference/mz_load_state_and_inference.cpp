#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-kv-cache-unified.h"
#include "ggml.h"
#include "mzCache_types.h"

#include <chrono>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

const size_t arena_size = 524288; // 512 KB
char * arena = (char *) malloc(arena_size);
size_t arena_offset = 0;

// Change the state_name definition to use states directory
static const std::string STATES_DIR = "states/";
static std::string state_name_prefix = STATES_DIR;

static char * copy_from_arena(const std::string & s, char * arena, size_t & offset, size_t arena_size) {
    size_t len = s.size();
    if (offset + len + 1 > arena_size) {
        fprintf(stderr, "Arena overflow!\n");
        exit(1);
    }
    char * dest = arena + offset;
    memcpy(dest, s.data(), len);
    dest[len] = '\0';
    offset += len + 1;
    return dest;
}

static std::string set_state_name(llama_model* model, common_params& params) {

    std::string state_name = state_name_prefix;
    std::string arch = model->arch_name();

    switch (model->arch) {
        case LLM_ARCH_LLAMA:
            state_name += "llama3";
            break;
        case LLM_ARCH_QWEN3:
            state_name += "qwen3";
            break;
        case LLM_ARCH_GEMMA3:
            state_name += "gemma3";
            break;
        default:
            state_name += "unknown";
            break;
    }

    std::string type = model->type_name();

    state_name += "_" + type;


    // Modify state name based on cache type
    if (params.cache_type_k == GGML_TYPE_BF16) {
        state_name += "_bf16";
    }
    else if (params.cache_type_k == GGML_TYPE_Q8_0) {
        state_name += "_q8_0";
    }
    else if (params.cache_type_k == GGML_TYPE_Q4_0) {
        state_name += "_q4_0";
    }


    if (params.flash_attn) {
        state_name += "_fa";
    }

    state_name += ".kv";

    return state_name;
}

int main(int argc, char** argv) {
    // 1. Configure parameters
    common_params params;
    params.sampling.seed  = 1234;
    params.n_ctx         = 32000;  // Very large context size
    params.n_predict     = 2048;   // Maximum number of tokens to generate
    params.n_batch       = 32000;
    params.n_ubatch      = 16;
    params.warmup        = false;

    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_DEBUG) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();

    // 2. Initialize model and context
    auto init_res = common_init_from_params(params);
    llama_model*   model = init_res.model.get();
    llama_context* ctx   = init_res.context.get();
    if (!model || !ctx) {
        std::fprintf(stderr, "ERROR: model/context initialization failed\n");
        return 1;
    }

    std::string state_name = set_state_name(model, params);


    // 3. Prepare ChatML template and vocabulary
    const char* tmpl             = llama_model_chat_template(model, nullptr);
    const llama_vocab* vocab     = llama_model_get_vocab(model);

    // 4. Initialize sampler (greedy)
    llama_sampler* sampler = llama_sampler_init_greedy();

    // 5. Prepare session storage
    std::vector<llama_token> session_tokens;
    session_tokens.reserve(params.n_ctx);

    int    n_past        = 0;
    size_t n_token_count = 0;
    std::string user_input;
    std::vector<char> buf(524288);
    llama_batch batch;

    // 6. Load existing session if available
    if (std::filesystem::exists(state_name)) {
        // Load tokens into temporary buffer
        std::vector<llama_token> tmp(params.n_ctx);

        auto t_load_start = std::chrono::high_resolution_clock::now();

        // Use original single file loading
        if (!llama_state_load_file(ctx,
                                    state_name.c_str(),
                                    tmp.data(),
                                    tmp.size(),
                                    &n_token_count)) {
            std::cerr << "ERROR: failed to load state.bin\n";
            return 1;
        }

        auto t_load_end = std::chrono::high_resolution_clock::now();
        double t_load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
        printf("[Timing] Loaded state.bin in %.2f ms\n", t_load_ms);

        // Assign loaded tokens to session
        session_tokens.assign(tmp.begin(), tmp.begin() + n_token_count);
        n_past = static_cast<int>(n_token_count);

        std::cout << "[Info] Loaded state.bin with " << n_token_count << " tokens\n";

        // Print previous tokens
        // std::cout << "[Info] Previous tokens: ";
        // for (size_t i = 0; i < n_token_count; ++i) {
        //     char buf_char[128];
        //     int  n = llama_token_to_piece(vocab,
        //                                   session_tokens[i],
        //                                   buf_char,
        //                                   sizeof(buf_char),
        //                                   0,
        //                                   true);
        //     if (n < 0) {
        //         std::fprintf(stderr,
        //                      "ERROR: failed to convert token to piece\n");
        //         return 1;
        //     }
        //     std::cout << std::string(buf_char, n);
        // }
        // std::cout << std::endl;

        // Get KV

        auto * kv_cache = static_cast<llama_kv_cache_unified *>(llama_get_memory(ctx));
        const int   layer   = 27;

#ifdef MZCACHE_SVM_KV_CHUNK

    struct ggml_tensor_ptrs k_tensors = kv_cache->layers_k_chunks[layer];
    const int   hdim   = model->hparams.n_embd_k_gqa(layer);
    const size_t stride = ggml_row_size(GGML_TYPE_F16, hdim);

    // For chunk 0..9 (i.e. tokens 0–255, 256–511, …)
    for (int chunk_idx = 91; chunk_idx < 93; ++chunk_idx) {
        ggml_tensor *chunk = k_tensors.c[chunk_idx];

        uint16_t halfs[10];

        // We only need the first element of the first 10 tokens
        std::vector<uint16_t> raw(10);
        for (int t = 100; t < 130; ++t) {
            // token index within the entire stream
            uint32_t tok = chunk_idx*256 + t;
            // byte offset to that token’s first element
            size_t off = (size_t)t * stride;

            std::cout << "offset " << off << " for token " << tok << ": ";
            ggml_backend_tensor_get(
                chunk,
                &halfs[0],
                off,
                sizeof(uint16_t) * 10
            );
            // convert to float
            std::cout << "[K token]" << " ";
            for (int i = 0; i < 10; ++i) {
                float f = ggml_fp16_to_fp32(halfs[i]);
                std::cout << f << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    
#else

    const int   step    = 256;   // tokens per chunk
    const int   groups  = 10;    // number of chunks to inspect
    const int   hdim    = model->hparams.n_embd_k_gqa(layer);
    const size_t stride = ggml_row_size(GGML_TYPE_F16, hdim);

    // 1) grab one continuous view of the entire K cache for layer 0
    ggml_tensor *tk = kv_cache->get_k_tensor(layer);

    // 2) buffer to hold one half per token
    uint16_t halfs[10];
    
    // 3) iterate chunks 0..9, printing tokens chunk*256 + [0..9]
    for (int chunk_idx = 91; chunk_idx < 93; ++chunk_idx) {
        uint32_t base_tok = chunk_idx * step;
        std::cout << "\n-- tokens " << base_tok
                  << " .. " << (base_tok + 9)
                  << " first element --\n";

        for (int j = 100; j < 130; ++j) {
            uint32_t tok = base_tok + j;
            size_t   off = (size_t)tok * stride;  // byte offset to first element

            std::cout << "offset " << off << " for token " << tok << ": ";

            // copy exactly one half (16 bits)
            ggml_backend_tensor_get(
                tk,
                &halfs[0],
                off,
                sizeof(uint16_t) * 10
            );
            // convert to float
            std::cout << "[K token]" << " ";
            for (int i = 0; i < 10; ++i) {
                float f = ggml_fp16_to_fp32(halfs[i]);
                std::cout << f << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

#endif


    } else {
        // return with error if no state file exists
        std::cerr << "ERROR: No state file found at " << state_name << "\n";
        std::cerr << "Please create state file first in 3970-server:/home/nxclab/hongseung/new_llama/llama.cpp\n";
        return 1;
    }

    // 8. Interactive loop
    while (true) {
        std::cout << "[User]> ";
        std::getline(std::cin, user_input);
        if (!std::cin || user_input == "exit") {
            break;
        }
        if (user_input == "ss") {
            // Save session state

            if (!llama_state_save_file(
                ctx,
                state_name.c_str(),
                session_tokens.data(),
                n_past
            )) {
                std::cerr << "ERROR: failed to save session to state.bin\n";
            } else {
                std::cout << "[Info] Session saved with "
                          << n_past << " tokens\n";
            }
            continue;
        }

        auto t_input_done = std::chrono::high_resolution_clock::now();

        // Process user message
        llama_chat_message umsg{ "user", copy_from_arena(user_input, arena, arena_offset, arena_size)};

        // Apply ChatML template for follow-up
        int32_t r = llama_chat_apply_template(
            tmpl,
            &umsg,
            1,
            true,
            buf.data(),
            buf.size()
        );
        std::string follow_prompt(buf.data(), r);

        // Tokenize follow-up prompt
        int n_tok = -llama_tokenize(
            vocab,
            follow_prompt.c_str(),
            follow_prompt.size(),
            nullptr,
            0,
            true,
            true
        );
        std::vector<llama_token> follow_tokens(n_tok);
        llama_tokenize(
            vocab,
            follow_prompt.c_str(),
            follow_prompt.size(),
            follow_tokens.data(),
            follow_tokens.size(),
            true,
            true
        );

        llama_batch batch2 = llama_batch_init(follow_tokens.size(), 0, 1);
        int32_t pos = n_past;
        for (size_t i = 0; i < follow_tokens.size(); ++i, ++pos) {
            common_batch_add(batch2,
                            follow_tokens[i],
                            pos,
                            {0},
                            false);
        }

        int n_ctx = llama_n_ctx(ctx);
        int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
        if (n_ctx_used + batch2.n_tokens > n_ctx) {
            printf("\033[0m\n");
            fprintf(stderr, "context size exceeded\n");
            exit(0);
        }

        batch2.logits[batch2.n_tokens - 1] = true;
        auto t_decode_start = std::chrono::high_resolution_clock::now();
        llama_decode(ctx, batch2);
        auto t_decode_done = std::chrono::high_resolution_clock::now();

        double t_TTFT1_ms = std::chrono::duration<double, std::milli>(t_decode_done - t_decode_start).count();
        double t_TTFT2_ms = std::chrono::duration<double, std::milli>(t_decode_done - t_input_done).count();
        printf("[Timing] TTFT1 %.2f ms\n", t_TTFT1_ms);
        printf("[Timing] TTFT2 %.2f ms\n", t_TTFT2_ms);

        
        
        
        int tok = llama_sampler_sample(sampler, ctx, -1);
        std::cout << common_token_to_piece(ctx, tok) << std::endl;

        const float * logits = llama_get_logits(ctx);

        for (int i = 0; i < 20; ++i) {
            printf("logits[%d] = %f\n", i, logits[i]);
        }

        auto * kv_cache = static_cast<llama_kv_cache_unified *>(llama_get_memory(ctx));
        const int   layer   = 1;

#ifdef MZCACHE_SVM_KV_CHUNK

    struct ggml_tensor_ptrs k_tensors = kv_cache->layers_k_chunks[layer];
    const int   hdim   = model->hparams.n_embd_k_gqa(layer);
    const size_t stride = ggml_row_size(GGML_TYPE_F16, hdim);

    // For chunk 0..9 (i.e. tokens 0–255, 256–511, …)
    for (int chunk_idx = 91; chunk_idx < 93; ++chunk_idx) {
        ggml_tensor *chunk = k_tensors.c[chunk_idx];

        uint16_t halfs[10];

        // We only need the first element of the first 10 tokens
        std::vector<uint16_t> raw(10);
        for (int t = 100; t < 130; ++t) {
            // token index within the entire stream
            uint32_t tok = chunk_idx*256 + t;
            // byte offset to that token’s first element
            size_t off = (size_t)t * stride;

            std::cout << "offset " << off << " for token " << tok << ": ";
            ggml_backend_tensor_get(
                chunk,
                &halfs[0],
                off,
                sizeof(uint16_t) * 10
            );
            // convert to float
            std::cout << "[K token]" << " ";
            for (int i = 0; i < 10; ++i) {
                float f = ggml_fp16_to_fp32(halfs[i]);
                std::cout << f << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    
#else

    const int   step    = 256;   // tokens per chunk
    const int   groups  = 10;    // number of chunks to inspect
    const int   hdim    = model->hparams.n_embd_k_gqa(layer);
    const size_t stride = ggml_row_size(GGML_TYPE_F16, hdim);

    // 1) grab one continuous view of the entire K cache for layer 0
    ggml_tensor *tk = kv_cache->get_k_tensor(layer);

    // 2) buffer to hold one half per token
    uint16_t halfs[10];
    
    // 3) iterate chunks 0..9, printing tokens chunk*256 + [0..9]
    for (int chunk_idx = 91; chunk_idx < 93; ++chunk_idx) {
        uint32_t base_tok = chunk_idx * step;
        std::cout << "\n-- tokens " << base_tok
                  << " .. " << (base_tok + 9)
                  << " first element --\n";

        for (int j = 100; j < 130; ++j) {
            uint32_t tok = base_tok + j;
            size_t   off = (size_t)tok * stride;  // byte offset to first element

            std::cout << "offset " << off << " for token " << tok << ": ";

            // copy exactly one half (16 bits)
            ggml_backend_tensor_get(
                tk,
                &halfs[0],
                off,
                sizeof(uint16_t) * 10
            );
            // convert to float
            std::cout << "[K token]" << " ";
            for (int i = 0; i < 10; ++i) {
                float f = ggml_fp16_to_fp32(halfs[i]);
                std::cout << f << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

#endif

        
        n_past += batch2.n_tokens;

        // Add user tokens to session
        session_tokens.insert(
            session_tokens.end(),
            follow_tokens.begin(),
            follow_tokens.end()
        );


        // Generation loop for follow-up
        for (int i = 0; i < params.n_predict; ++i) {
            auto tok = llama_sampler_sample(sampler, ctx, -1);
            session_tokens.push_back(tok);

            if (tok == llama_vocab_eos(vocab))
                break;

            common_batch_clear(batch2);
            common_batch_add(batch2, tok, n_past, {0}, true);
            if (llama_decode(ctx, batch2)) {
                fprintf(stderr, "\n%s : failed to evaluate\n", __func__);
                llama_batch_free(batch2);

                llama_batch_free(batch);
                llama_sampler_free(sampler);
                return 1;
            }
            n_past += 1;

            std::cout << common_token_to_piece(ctx, tok);
            if (tok == llama_vocab_eos(vocab))
                break;
        }
        std::cout << std::endl;
        llama_batch_free(batch2);
    }
    
    // 9. Cleanup
    llama_batch_free(batch);
    llama_sampler_free(sampler);

    free(arena);

    
    return 0;
}
