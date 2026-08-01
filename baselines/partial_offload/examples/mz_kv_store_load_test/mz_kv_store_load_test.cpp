#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-kv-cache-unified.h"
#include "ggml.h"
#include "mzCache_types.h"
#include "mzCache_kvchunk.h"

#include <chrono>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

extern std::vector<chunk_ptrs *> g_svm_chunk_ptrs;

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

    } else {
        // return with error if no state file exists
        std::cerr << "ERROR: No state file found at " << state_name << "\n";
        std::cerr << "Please create state file first in 3970-server:/home/nxclab/hongseung/new_llama/llama.cpp\n";
        return 1;
    }


        // Get KV

        auto * kv_cache = static_cast<llama_kv_cache_unified *>(llama_get_memory(ctx));

#ifdef MZCACHE_SVM_KV_CHUNK
    const int   n_layers = kv_cache->layers_k_chunks.size();
    const int   layer    = n_layers - 1;

    struct ggml_tensor_ptrs k_tensors = kv_cache->layers_k_chunks[layer];
    const int   hdim   = model->hparams.n_embd_k_gqa(layer);
    const int   n_chunks_per_tensor = (n_past + TOKENS_PER_CHUNK - 1) / TOKENS_PER_CHUNK;
    const size_t stride = ggml_row_size(GGML_TYPE_F16, hdim);

    /////// Compress and Decompress Example ///////
    mzcache_kv_cache * kv_cache_ptr = new mzcache_kv_cache(n_layers, hdim, n_chunks_per_tensor, model->name);

    // for(int i = 50; i < 800; i += 50) {
    //     std::cout << "chunk num: " << i << std::endl;
    //     kv_cache_ptr->mzcache_store_chunks(i);
    //     kv_cache_ptr->mzcache_load_chunks(i);
    // }

    __fp16* last_key_chunk_data = (__fp16*)g_svm_chunk_ptrs[2 * kv_cache_ptr->cur_layer_idx]->c[kv_cache_ptr->cur_chunk_idx_to_store];

    std::cout << "current layer index: " << kv_cache_ptr->cur_layer_idx << "\n";
    std::cout << "current chunk index to store: " << kv_cache_ptr->cur_chunk_idx_to_store << "\n";
    std::cout << "Last key chunk before loading:\n";
    for (int i = 0; i < 20; ++i) {
        float f = (float)(last_key_chunk_data[i]);
        std::cout << f << " ";
    }
    std::cout << "\n";
    
    kv_cache_ptr->mzcache_store_chunks(500);
    
    printf("Stored layer %d, chunk %d\n", kv_cache_ptr->cur_layer_idx, kv_cache_ptr->cur_chunk_idx_to_store - 1);

    kv_cache_ptr->mzcache_load_chunks_sequential(500);



    last_key_chunk_data = (__fp16*)g_svm_chunk_ptrs[2 * kv_cache_ptr->cur_layer_idx]->c[kv_cache_ptr->cur_chunk_idx_to_store];

    std::cout << "current layer index: " << kv_cache_ptr->cur_layer_idx << "\n";
    std::cout << "current chunk index to store: " << kv_cache_ptr->cur_chunk_idx_to_store << "\n";
    std::cout << "Last key chunk after loading:\n";
    for (int i = 0; i < 20; ++i) {
        float f = (float)(last_key_chunk_data[i]);
        std::cout << f << " ";
    }
    std::cout << "\n";


    delete kv_cache_ptr;

#else

    printf("Error: MZCACHE_SVM_KV_CHUNK is not defined. Please define it to use chunked KV cache.\n");
    return 1;

#endif

    // 9. Cleanup

    printf("Freeing sampler...\n");

    llama_sampler_free(sampler);

    printf("Freeing arena memory...\n");

    free(arena);

    
    return 0;
}
