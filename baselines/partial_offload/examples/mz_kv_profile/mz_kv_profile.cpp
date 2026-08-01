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
#include <map>
#include <utility>

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


// Helper: concatenate two vectors
template<typename T>
std::vector<T> concat(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> out = a;
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

// holds benchmark results
struct BenchResult {
    int small_A;     // number of small cores assigned to ALLOC
    int big_A;       // number of big cores assigned to ALLOC
    double ms_alloc;
    double ms_decomp;
};


int main(int argc, char** argv) {
    // 1. Configure parameters
    common_params params;
    params.sampling.seed  = 1234;
    params.n_ctx         = 29000;  // Very large context size
    params.n_predict     = 2048;   // Maximum number of tokens to generate
    params.n_batch       = 512;
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

        auto * kv_cache = static_cast<llama_kv_cache_unified *>(llama_get_memory(ctx));

#ifdef MZCACHE_SVM_KV_CHUNK
    const int   n_layers = kv_cache->layers_k_chunks.size();
    const int   layer    = n_layers - 1;

    struct ggml_tensor_ptrs k_tensors = kv_cache->layers_k_chunks[layer];
    const int   hdim   = model->hparams.n_embd_k_gqa(layer);
    const int   n_chunks_per_tensor = (n_past + TOKENS_PER_CHUNK - 1) / TOKENS_PER_CHUNK;
    const size_t stride = ggml_row_size(GGML_TYPE_F16, hdim);

    // For chunk 0..9 (i.e. tokens 0–255, 256–511, …)
    for (int chunk_idx = n_chunks_per_tensor - 2; chunk_idx < n_chunks_per_tensor; ++chunk_idx) {
        ggml_tensor *chunk = k_tensors.c[chunk_idx];

        __fp16 halfs[64];
        std::cout << "layer " << layer << ", chunk " << chunk_idx << ": " << std::endl;


        // We only need the first element of the first 10 tokens
        for (int t = 0; t < 1; ++t) {
            // token index within the entire stream
            uint32_t tok = chunk_idx*256 + t;
            // byte offset to that token’s first element
            size_t off = (size_t)t * stride;

            std::cout << "offset " << off << " for token " << tok << ": ";
            ggml_backend_tensor_get(
                chunk,
                &halfs[0],
                off,
                sizeof(__fp16) * 64
            );
            // convert to float
            std::cout << "[K token]" << " ";

            __fp16 max = -1000.0f;
            __fp16 min = 1000.0f;
            for (int i = 0; i < 64; ++i) {
                float f = (float)(halfs[i]);
                std::cout << f << " ";
                max = std::max(max, halfs[i]);
                min = std::min(min, halfs[i]);
            }
            std::cout << "\n";
            std::cout << "Max: " << (float)max << ", Min: " << (float)min << "\n";

        }
        std::cout << "\n";
    }


    /////// Compress and Decompress Example ///////
    mzcache_kv_cache * kv_cache_ptr = new mzcache_kv_cache(n_layers, hdim, n_chunks_per_tensor, model->name);

    const std::vector<int> SMALL = {0,1,2,3,4,5};  // 6 cores
    const std::vector<int> BIG   = {6,7};          // 2 cores
    const int N = 500;                    // as many as desired

    std::vector<BenchResult> results;

    for (int nSmallA = 0; nSmallA <= (int)SMALL.size(); ++nSmallA) {
        for (int nBigA = 0; nBigA <= (int)BIG.size(); ++nBigA) {
            /* 1) build core sets ------------------------------------- */
            std::vector<int> small_A(SMALL.begin(), SMALL.begin() + nSmallA);
            std::vector<int> big_A  (BIG.begin(),   BIG.begin()   + nBigA);
            std::vector<int> small_B(SMALL.begin() + nSmallA, SMALL.end());
            std::vector<int> big_B  (BIG.begin()   + nBigA,   BIG.end());

            std::map<CoreType,std::vector<int>> cfg = {
                { CoreType::ALLOC,  concat(small_A, big_A) },
                { CoreType::DECOMP, concat(small_B, big_B) }
            };

            if (cfg[CoreType::ALLOC].empty() || cfg[CoreType::DECOMP].empty())
                continue;   // skip this combination

            std::cout << "Configuration: " 
                      << "small_A = " << nSmallA << ", "
                      << "big_A = "   << nBigA   << "\n";
            /* 2) init cache ----------------------------------------- */

            /* 2) init pool/cache --------------------------------------- */
            kv_cache_ptr->reconfigure_thread_pool(cfg);

            kv_cache_ptr->mzcache_compress_chunks(N);


            /* 3) run & measure ----------------------------------------- */
            auto [msAlloc, msDecomp] =
                kv_cache_ptr->mzcache_decompress_chunks_profile(N);

            results.push_back({nSmallA, nBigA, msAlloc, msDecomp});
        }
    }

    /* ---------- print summary ---------- */
    std::cout << "\n===== SUMMARY =====\n"
              << "small_A  big_A  alloc(ms)  decomp(ms)  total(ms)\n";
    for (const auto& r : results) {
        std::cout << "   " << r.small_A << "       "
                  << r.big_A   << "      "
                  << r.ms_alloc << "      "
                  << r.ms_decomp << "      "
                  << (r.ms_alloc + r.ms_decomp) << '\n';
    }

    std::map<CoreType,std::vector<int>> cfg = {
        { CoreType::ALLOC, {0,1,2} },
        { CoreType::READ,   {0,1} } // two middle cores are sufficient to utilize the i/o bandwidth
    };
    kv_cache_ptr->reconfigure_thread_pool(cfg);

    kv_cache_ptr->mzcache_store_chunks(N);


    double t_store_ms = kv_cache_ptr->mzcache_load_chunks_sequential(N);

    std::cout << "Loaded " << N << " chunks in " << t_store_ms << " ms\n";

    /* ===== 1. compute C-worker (fixed speed) throughput ===== */
    const double sC = static_cast<double>(N) / t_store_ms;   // task/ms

    /* ===== 2. compute k, k/N, makespan per (small_A,big_A) combination ===== */
    struct OptResult {
        BenchResult r;           // resource split (small_A, big_A, individual times)
        int    kB, kC;           // number of tasks handled by B·C
        double k_ratio_B;        // kB/N
        double makespan_ms;      // total elapsed time
    };
    std::vector<OptResult> opt;                 // all feasible cases
    OptResult best;                             // the single min-makespan entry
    best.makespan_ms = std::numeric_limits<double>::max();

    for (const auto& r : results)
    {
        const double sA = static_cast<double>(N) / r.ms_alloc;   // Stage 1
        const double sB = static_cast<double>(N) / r.ms_decomp;  // B speed
        const double sBC = sB + sC;                              // Stage 2 combined

        /* optimal B·C split (proportional to speed) */
        int    kB = static_cast<int>(std::round(N * sB / sBC));
        int    kC = N - kB;
        double k_ratio_B = static_cast<double>(kB) / N;

        /* pipeline makespan                                      */
        double makespan = N / std::min(sA, sBC);  // see formula ↑

        opt.push_back({r, kB, kC, k_ratio_B, makespan});
        if (makespan < best.makespan_ms) best = opt.back();
    }

    std::cout << "\n===== PIPELINE ANALYSIS (parallel Stage 2) =====\n"
            << "small_A big_A  kB   kC   kB/N  makespan(ms)\n";
    for (const auto& o : opt) {
        std::cout << "   " << o.r.small_A
                << "       " << o.r.big_A
                << "   " << o.kB
                << "   " << o.kC
                << "   " << o.k_ratio_B
                << "   " << o.makespan_ms << '\n';
    }

    /* ---------- best result summary ---------- */
    std::cout << "\n=== BEST CONFIG (min makespan) ===\n"
            << "A cores : small=" << best.r.small_A
            << ", big="           << best.r.big_A  << '\n'
            << "B cores : small=" << 6 - best.r.small_A
            << ", big="           << 2 - best.r.big_A  << '\n'
            << "Tasks   : B=" << best.kB << ", C=" << best.kC
            << "  (kB/N = " << best.k_ratio_B << ")\n"
            << "Makespan : " << best.makespan_ms << " ms\n";

    delete kv_cache_ptr;

#else

    printf("Error: MZCACHE_SVM_KV_CHUNK is not defined. Please define it to use chunked KV cache.\n");
    return 1;

#endif


    } else {
        // return with error if no state file exists
        std::cerr << "ERROR: No state file found at " << state_name << "\n";
        std::cerr << "Please create state file first in 3970-server:/home/nxclab/hongseung/new_llama/llama.cpp\n";
        return 1;
    }


    // 9. Cleanup

    printf("Freeing sampler...\n");

    llama_sampler_free(sampler);

    printf("Freeing arena memory...\n");

    free(arena);

    
    return 0;
}








/////
