#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-model.h"
#include "llama-kv-cache-unified.h"
#include "ggml.h"
#include "mzcache_types.h"
#include "mzcache_kv_state.h"
#include "mzcache_weight.h"
#include "mzcache_core.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <filesystem>

// mz_save_state_v2 — on-device KV compression roundtrip for the accuracy
// experiment. For state index <idx>:
//   1. load the raw (uncompressed) KV state saved by mz_accuracy
//   2. full-evict it (swap-out: compress every chunk, offload arenas to storage)
//   3. restore it (swap-in: reload + decompress) -> compressed->decompressed KV
//   4. save that reconstructed state for mz_load_state_accuracy to score.
// The compression backend is chosen at compile time (MZCACHE_COMPRESSION); the
// reconstructed states go to that backend's output dir.

static const std::string RAW_STATES_DIR = "/data/local/tmp/mzcache/accuracy_test/raw/";
#if defined(MZCACHE_COMPRESS_FLEXGEN)
static const std::string OUTPUT_STATES_DIR = "/data/local/tmp/mzcache/accuracy_test/full_swap_fg/";
#elif defined(MZCACHE_COMPRESS_FLEXGEN_8BIT)
static const std::string OUTPUT_STATES_DIR = "/data/local/tmp/mzcache/accuracy_test/full_swap_fg8bit/";
#elif defined(MZCACHE_COMPRESS_CACHEGEN)
static const std::string OUTPUT_STATES_DIR = "/data/local/tmp/mzcache/accuracy_test/full_swap_cg/";
#else
static const std::string OUTPUT_STATES_DIR = "/data/local/tmp/mzcache/accuracy_test/full_swap_default/";
#warning "No compression algorithm specified, using default output directory"
#endif

// State filename: <dir><arch>_<type>[_fa]_<idx>.kv — matches mz_accuracy and
// mz_load_state_accuracy so a document is addressed by its id end to end.
static std::string state_path(const std::string& dir, llama_model* model,
                              const common_params& params, int idx) {
    std::string name = dir;
    switch (model->arch) {
        case LLM_ARCH_LLAMA:   name += "llama3";  break;
        case LLM_ARCH_QWEN3:   name += "qwen3";   break;
        case LLM_ARCH_EXAONE4: name += "exaone4"; break;
        default:               name += "unknown"; break;
    }
    name += "_" + model->type_name();
    if (params.flash_attn) name += "_fa";
    name += "_" + std::to_string(idx) + ".kv";
    return name;
}

// Log MemAvailable (MB) at each phase, flushed, so an OOM/reboot points at the
// last phase printed.
static void log_mem(const char* phase) {
    long avail_kb = -1;
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        char line[128];
        while (fgets(line, sizeof line, f)) {
            if (sscanf(line, "MemAvailable: %ld kB", &avail_kb) == 1) break;
        }
        fclose(f);
    }
    std::printf("[phase] %-16s MemAvailable=%ld MB\n", phase, avail_kb / 1024);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "Usage: %s <model.gguf> <idx> <remaining_in_mem>\n"
            "  Load raw/<...>_<idx>.kv, evict to <remaining> (0 = full evict),\n"
            "  restore (decompress) and save the reconstructed .kv. remaining=0\n"
            "  measures the full compression roundtrip for the accuracy experiment.\n",
            argv[0]);
        return 1;
    }
    const std::string model_name = argv[1];
    const int   idx       = atoi(argv[2]);
    const float remaining = atof(argv[3]);

    common_params params;
    params.use_mmap     = false;
    params.n_ctx        = 32768;
    params.n_batch      = 32768;
    params.n_ubatch     = 16;
    params.warmup       = false;
    params.n_gpu_layers = 100;
    params.flash_attn   = true;
    params.model.path   = model_name;
    common_init();

    auto init_res = common_init_from_params(params);
    llama_model*   model = init_res.model.get();
    llama_context* ctx   = init_res.context.get();
    if (!model || !ctx) { std::fprintf(stderr, "ERROR: model/context init failed\n"); return 1; }

#ifndef MZCACHE_SVM_KV_CHUNK
    std::fprintf(stderr, "Error: build with MZCACHE_SVM_KV_CHUNK.\n");
    return 1;
#else
    const std::string in_path  = state_path(RAW_STATES_DIR,    model, params, idx);
    const std::string out_path = state_path(OUTPUT_STATES_DIR, model, params, idx);
    std::filesystem::create_directories(OUTPUT_STATES_DIR);

    // 1) Load the raw (uncompressed) KV state saved by mz_accuracy.
    if (!std::filesystem::exists(in_path)) {
        std::fprintf(stderr, "ERROR: no state file at %s\n", in_path.c_str());
        return 1;
    }
    std::vector<llama_token> tokens(params.n_ctx);
    size_t n_tok = 0;
    if (!llama_state_load_file(ctx, in_path.c_str(), tokens.data(), tokens.size(), &n_tok)) {
        std::fprintf(stderr, "ERROR: failed to load %s\n", in_path.c_str());
        return 1;
    }
    tokens.resize(n_tok);
    const int n_past = (int) n_tok;
    log_mem("loaded");

    // 2) Wrap the loaded KV in an mzcache_core.
    auto* kv_cache = static_cast<llama_kv_cache_unified*>(llama_get_memory(ctx));
    const int n_layers = (int) kv_cache->layers_k_chunks.size();
    const int hdim     = model->hparams.n_embd_k_gqa(n_layers - 1);
    const int n_chunks_per_tensor = (n_past + TOKENS_PER_CHUNK - 1) / TOKENS_PER_CHUNK;
    mzcache_core* core = new mzcache_core(n_layers, hdim, n_chunks_per_tensor, model);
    mzcache_kv_state* kv_state = core->get_kv_state();
    mzcache_weight*   weight   = core->get_weight();
    ThreadPool*       pool     = core->get_thread_pool();

    // 3) Full evict: compress every KV chunk and offload the arenas to storage.
    auto [ratio, n_comp, n_store, n_unload] = core->swapout(remaining);
    log_mem("swapout");

    // 4) Restore (the same swap-in steps mz_prefill runs): reload arenas +
    //    stored chunks + weights, then decompress back into the KV cache.
    kv_state->mzcache_reload_all_arenas();
    kv_state->mzcache_enqueue_load_chunks(n_store);
    weight->mzcache_reload_w_layers();
    pool->wait_idle(CoreType::READ);
    kv_state->mzcache_enqueue_decomp_chunks(n_comp);
    pool->wait_idle(CoreType::DECOMP);
    kv_state->clean_after_decompress(n_comp);
    kv_state->clean_after_read(n_store);
    log_mem("restored");

    // 5) Save the reconstructed (roundtripped) state for scoring.
    if (!llama_state_save_file(ctx, out_path.c_str(), tokens.data(), n_past)) {
        std::fprintf(stderr, "ERROR: failed to save %s\n", out_path.c_str());
        return 1;
    }
    log_mem("saved");
    std::printf("[Done] idx=%d ratio=%.4f comp=%d store=%d unload=%d -> %s\n",
                idx, ratio, n_comp, n_store, n_unload, out_path.c_str());
    delete core;
    return 0;
#endif
}
