#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "llama-kv-cache-unified.h"
#include "llama-model.h"
#include "mzcache_core.h"
#include "mzcache_kv_state.h"
#include "mzcache_types.h"
#include "mzcache_weight.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

extern std::vector<chunk_ptrs *> g_svm_chunk_ptrs;

static const std::string STATES_DIR = "/data/local/tmp/mzcache/states/";

static std::string set_state_name(llama_model * model, common_params & params, int context_size_of_states) {
    std::string state_name = STATES_DIR;

    switch (model->arch) {
        case LLM_ARCH_LLAMA:
            state_name += "llama3";
            break;
        case LLM_ARCH_QWEN3:
            state_name += "qwen3";
            break;
        case LLM_ARCH_EXAONE4:
            state_name += "exaone4";
            break;
        default:
            state_name += "unknown";
            break;
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

    if (context_size_of_states == 2049) {
        state_name += "_2049";
    } else if (context_size_of_states == 4097) {
        state_name += "_4097";
    } else if (context_size_of_states == 8193) {
        state_name += "_8193";
    } else if (context_size_of_states == 16385) {
        state_name += "_16385";
    } else if (context_size_of_states == 32700) {
        state_name += "_32700";
    } else if (context_size_of_states == 65500) {
        state_name += "_65500";
    }

    state_name += ".kv";
    return state_name;
}

static void append_log(
        const std::string & log_path,
        const std::string & log_file_name,
        const std::string & model_name,
        int context_size_of_states,
        float remaining_ratio,
        int repeats,
        int tokens_per_step,
        const std::vector<long> & prefill_us) {
    if (log_path.empty() || log_file_name.empty()) {
        std::cout << "[mzCache repeated log]\n"
                  << "  Model Name              : " << model_name << "\n"
                  << "  Context Size (states)   : " << context_size_of_states << "\n"
                  << "  Context Remaining Ratio : " << remaining_ratio << "\n"
                  << "  Repeats                 : " << repeats << "\n"
                  << "  Tokens per step         : " << tokens_per_step << "\n";
        for (size_t i = 0; i < prefill_us.size(); ++i) {
            std::cout << "  Cycle " << i << " Prefill Time (ms): "
                      << (double)prefill_us[i] / 1000.0 << "\n";
        }
        std::cout << std::endl;
        return;
    }

    const std::string full_path = log_path + "/" + log_file_name;
    const bool file_exists = std::filesystem::exists(full_path);
    std::ofstream log_file(full_path, std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Error: Cannot open log file at " << full_path << std::endl;
        return;
    }

    if (!file_exists) {
        log_file << "# mzCache repeated prefill log\n";
        log_file << "# model context remaining_ratio repeats tokens_per_step cycle prefill_us\n";
    }

    for (size_t i = 0; i < prefill_us.size(); ++i) {
        log_file << model_name << " "
                 << context_size_of_states << " "
                 << remaining_ratio << " "
                 << repeats << " "
                 << tokens_per_step << " "
                 << i << " "
                 << prefill_us[i] << "\n";
    }
}

static std::vector<llama_token> make_cycle_tokens(
        const llama_vocab * vocab,
        int cycle,
        int tokens_per_step) {
    static const std::array<const char *, 8> prompts = {
        "Mercury", "Venus", "Earth", "Mars",
        "Jupiter", "Saturn", "Uranus", "Neptune",
    };

    std::string prompt = prompts[cycle % prompts.size()];
    std::vector<llama_token> tokens;

    while ((int) tokens.size() < tokens_per_step) {
        const int n_tok = -llama_tokenize(
            vocab,
            prompt.c_str(),
            prompt.size(),
            nullptr,
            0,
            true,
            true);

        if (n_tok <= 0) {
            throw std::runtime_error("failed to count prompt tokens");
        }

        tokens.assign(n_tok, 0);
        const int got = llama_tokenize(
            vocab,
            prompt.c_str(),
            prompt.size(),
            tokens.data(),
            tokens.size(),
            true,
            true);

        if (got < 0) {
            throw std::runtime_error("failed to tokenize prompt");
        }
        tokens.resize(got);

        if ((int) tokens.size() < tokens_per_step) {
            prompt += " ";
            prompt += prompts[(cycle + (int) tokens.size()) % prompts.size()];
        }
    }

    tokens.resize(tokens_per_step);
    return tokens;
}

static llama_batch make_batch(const std::vector<llama_token> & tokens, int n_past) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    int32_t pos = n_past;
    for (llama_token token : tokens) {
        common_batch_add(batch, token, pos++, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    return batch;
}

struct LogitSnapshot {
    std::vector<float> logits;
    llama_token token = -1;
};

struct ChunkSnapshot {
    std::string label;
    int layer = -1;
    int chunk = -1;
    bool is_value = false;
    std::vector<uint16_t> bits;
};

static float half_bits_to_float(uint16_t bits) {
    __fp16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return (float) value;
}

static LogitSnapshot capture_logits(llama_context * ctx, llama_sampler * sampler, int n_vocab) {
    const float * logits = llama_get_logits(ctx);
    LogitSnapshot snapshot;
    snapshot.logits.assign(logits, logits + n_vocab);
    snapshot.token = llama_sampler_sample(sampler, ctx, -1);
    return snapshot;
}

static bool print_logit_comparison(
        llama_context * ctx,
        const LogitSnapshot & baseline,
        const LogitSnapshot & restored) {
    if (baseline.logits.size() != restored.logits.size()) {
        std::cerr << "ERROR: logit size mismatch\n";
        return false;
    }

    double sum_abs = 0.0;
    float max_abs = 0.0f;
    int max_idx = -1;
    bool finite = true;
    for (size_t i = 0; i < baseline.logits.size(); ++i) {
        const float a = baseline.logits[i];
        const float b = restored.logits[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            finite = false;
        }
        const float diff = std::fabs(a - b);
        sum_abs += diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_idx = (int) i;
        }
    }

    std::cout << "Baseline token: " << common_token_to_piece(ctx, baseline.token)
              << " | mzCache token: " << common_token_to_piece(ctx, restored.token) << "\n"
              << "Logit diff: max_abs=" << max_abs
              << " at vocab_idx=" << max_idx
              << ", mean_abs=" << sum_abs / (double) baseline.logits.size() << "\n";

    if (!finite) {
        std::cerr << "ERROR: non-finite logits detected\n";
        return false;
    }

    return true;
}

static ChunkSnapshot capture_chunk_snapshot(
        int layer,
        int chunk,
        bool is_value,
        int sample_count,
        const std::string & label) {
    ChunkSnapshot snapshot;
    snapshot.label = label;
    snapshot.layer = layer;
    snapshot.chunk = chunk;
    snapshot.is_value = is_value;

    const int slot = 2 * layer + (is_value ? 1 : 0);
    if (slot < 0 || slot >= (int) g_svm_chunk_ptrs.size() ||
        chunk < 0 || chunk >= MAX_CHUNKS_PER_TENSOR ||
        g_svm_chunk_ptrs[slot]->c[chunk] == nullptr) {
        throw std::runtime_error("cannot capture chunk snapshot for " + label);
    }

    const auto * data = (const uint16_t *) g_svm_chunk_ptrs[slot]->c[chunk];
    snapshot.bits.assign(data, data + sample_count);
    return snapshot;
}

static bool compare_chunk_snapshot(const ChunkSnapshot & before, bool require_exact) {
    const int slot = 2 * before.layer + (before.is_value ? 1 : 0);
    if (slot < 0 || slot >= (int) g_svm_chunk_ptrs.size() ||
        before.chunk < 0 || before.chunk >= MAX_CHUNKS_PER_TENSOR ||
        g_svm_chunk_ptrs[slot]->c[before.chunk] == nullptr) {
        std::cerr << "ERROR: restored chunk pointer is null for " << before.label << "\n";
        return false;
    }

    const auto * after = (const uint16_t *) g_svm_chunk_ptrs[slot]->c[before.chunk];
    int mismatches = 0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < before.bits.size(); ++i) {
        if (before.bits[i] != after[i]) {
            mismatches++;
        }
        const float diff = std::fabs(half_bits_to_float(before.bits[i]) - half_bits_to_float(after[i]));
        sum_abs += diff;
        max_abs = std::max(max_abs, diff);
    }

    std::cout << before.label << " first 10 before/after:";
    const size_t n_print = std::min<size_t>(10, before.bits.size());
    for (size_t i = 0; i < n_print; ++i) {
        std::cout << " " << half_bits_to_float(before.bits[i])
                  << "/" << half_bits_to_float(after[i]);
    }
    std::cout << "\n"
              << before.label << " compare: samples=" << before.bits.size()
              << ", bit_mismatches=" << mismatches
              << ", max_abs=" << max_abs
              << ", mean_abs=" << sum_abs / (double) before.bits.size() << "\n";

    if (require_exact && mismatches != 0) {
        std::cerr << "ERROR: expected exact chunk restore for " << before.label << "\n";
        return false;
    }

    return true;
}

static bool cleanup_appended_range(llama_context * ctx, int n_past) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, n_past, -1);

    const llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
    if (pos_max != (llama_pos) n_past - 1) {
        std::cerr << "ERROR: unexpected seq_pos_max after cleanup. expected "
                  << (n_past - 1) << ", got " << pos_max << std::endl;
        return false;
    }

    return true;
}

static void print_usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0
        << " <model_name> <context_size_of_states> <context_remaining_in_mem>"
        << " [repeats=3] [prefill_type=0] [tokens_per_step=8] [log_path] [log_file_name] [verify_restore=0]\n\n"
        << "  prefill_type: 0=prefill, 1=no_alloc_overlap, 2=no_prefill_overlap, 3=no_overlap\n"
        << "  verify_restore can also be enabled with MZCACHE_VERIFY_RESTORE=1\n";
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string model_name = argv[1];
    const int context_size_of_states = std::atoi(argv[2]);
    const float context_remaining_in_mem = std::atof(argv[3]);
    const int repeats = (argc > 4) ? std::atoi(argv[4]) : 3;
    const int prefill_type = (argc > 5) ? std::atoi(argv[5]) : 0;
    const int tokens_per_step = (argc > 6) ? std::atoi(argv[6]) : 8;
    const std::string log_path = (argc > 7) ? argv[7] : "";
    const std::string log_file_name = (argc > 8) ? argv[8] : "";
    const char * verify_env = std::getenv("MZCACHE_VERIFY_RESTORE");
    bool verify_restore = verify_env != nullptr && std::atoi(verify_env) != 0;
    if (argc > 9) {
        verify_restore = std::atoi(argv[9]) != 0;
    }

    if (repeats <= 0 || tokens_per_step <= 0) {
        std::cerr << "ERROR: repeats and tokens_per_step must be positive\n";
        return 1;
    }

    common_params params;
    params.sampling.seed = 1234;
    params.use_mmap = false;
    params.n_ctx = context_size_of_states + 32;
    params.n_predict = 32;
    params.n_batch = params.n_ctx;
    params.n_ubatch = 16;
    params.warmup = false;
    params.n_gpu_layers = 100;
    params.flash_attn = true;
    params.model.path = model_name;

    std::cout << "Model: " << model_name << "\n"
              << "Context size: " << context_size_of_states << "\n"
              << "Context remaining ratio: " << context_remaining_in_mem << "\n"
              << "Repeats: " << repeats << "\n"
              << "Prefill type: " << prefill_type << "\n"
              << "Tokens per step: " << tokens_per_step << "\n"
              << "Verify restore: " << (verify_restore ? "on" : "off") << "\n";

    common_init();
    auto init_res = common_init_from_params(params);
    llama_model * model = init_res.model.get();
    llama_context * ctx = init_res.context.get();
    if (!model || !ctx) {
        std::fprintf(stderr, "ERROR: model/context initialization failed\n");
        return 1;
    }

    const std::string state_name = set_state_name(model, params, context_size_of_states);
    int n_past = 0;
    size_t n_token_count = 0;

    if (!std::filesystem::exists(state_name)) {
        std::cerr << "ERROR: No state file found at " << state_name << "\n";
        return 1;
    }

    std::vector<llama_token> tmp(params.n_ctx);
    auto t_load_start = std::chrono::high_resolution_clock::now();
    if (!llama_state_load_file(ctx, state_name.c_str(), tmp.data(), tmp.size(), &n_token_count)) {
        std::cerr << "ERROR: failed to load state file: " << state_name << "\n";
        return 1;
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();
    const double t_load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    n_past = (int) n_token_count;
    std::cout << "[Info] Loaded state with " << n_token_count
              << " tokens in " << t_load_ms << " ms\n";

    if (!cleanup_appended_range(ctx, n_past)) {
        return 1;
    }

    if ((n_past % TOKENS_PER_CHUNK) + tokens_per_step > TOKENS_PER_CHUNK) {
        std::cerr << "ERROR: this repeated test keeps n_past fixed and does not cross chunk boundaries yet. "
                  << "n_past % TOKENS_PER_CHUNK = " << (n_past % TOKENS_PER_CHUNK)
                  << ", tokens_per_step = " << tokens_per_step
                  << ", TOKENS_PER_CHUNK = " << TOKENS_PER_CHUNK << "\n";
        return 1;
    }

    const int n_ctx = llama_n_ctx(ctx);
    if (n_past + tokens_per_step > n_ctx) {
        std::cerr << "ERROR: context size exceeded: n_past=" << n_past
                  << ", tokens_per_step=" << tokens_per_step
                  << ", n_ctx=" << n_ctx << "\n";
        return 1;
    }

    llama_sampler * sampler = llama_sampler_init_greedy();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

#ifdef MZCACHE_SVM_KV_CHUNK
    auto * kv_cache = static_cast<llama_kv_cache_unified *>(llama_get_memory(ctx));
    const int n_layers = kv_cache->layers_k_chunks.size();
    const int layer = n_layers - 1;
    const int hdim = model->hparams.n_embd_k_gqa(layer);
    const int n_chunks_per_tensor = (n_past + TOKENS_PER_CHUNK - 1) / TOKENS_PER_CHUNK;

    std::cout << "Initializing mzcache_core with " << n_layers
              << " layers, hidden dimension " << hdim
              << ", and " << n_chunks_per_tensor << " active chunks per tensor.\n";

    mzcache_core * core = new mzcache_core(n_layers, hdim, n_chunks_per_tensor, model);
    mzcache_kv_state * kv_state = core->get_kv_state();
    mzcache_weight * weight = core->get_weight();

    int max_comp_chunks = 0;
    int max_store_chunks = 0;
    for (int i = 0; i < n_layers; ++i) {
        max_comp_chunks += kv_state->chunk_bound[i];
        max_store_chunks += n_chunks_per_tensor - kv_state->chunk_bound[i];
    }

    std::vector<long> prefill_times;
    prefill_times.reserve(repeats);

    for (int cycle = 0; cycle < repeats; ++cycle) {
        std::cout << "\n========== Repeated cycle " << cycle << " ==========\n";

        if (!cleanup_appended_range(ctx, n_past)) {
            llama_sampler_free(sampler);
            delete core;
            return 1;
        }

        auto tokens = make_cycle_tokens(vocab, cycle, tokens_per_step);
        LogitSnapshot baseline_logits;
        std::vector<ChunkSnapshot> chunk_snapshots;

        if (verify_restore) {
            llama_batch baseline_batch = make_batch(tokens, n_past);
            const int baseline_ret = llama_decode(ctx, baseline_batch);
            if (baseline_ret != 0 && baseline_ret != 1) {
                std::cerr << "ERROR: baseline decode failed, ret=" << baseline_ret << "\n";
                llama_batch_free(baseline_batch);
                llama_sampler_free(sampler);
                delete core;
                return 1;
            }
            baseline_logits = capture_logits(ctx, sampler, n_vocab);
            llama_batch_free(baseline_batch);

            if (!cleanup_appended_range(ctx, n_past)) {
                llama_sampler_free(sampler);
                delete core;
                return 1;
            }

            const int snapshot_layer = n_layers - 1;
            const int sample_count = std::min(1024, TOKENS_PER_CHUNK * hdim);
            const int stored_chunk = n_chunks_per_tensor - 1;
            chunk_snapshots.push_back(capture_chunk_snapshot(
                snapshot_layer, stored_chunk, false, sample_count, "stored K last-layer/chunk"));
            chunk_snapshots.push_back(capture_chunk_snapshot(
                snapshot_layer, stored_chunk, true, sample_count, "stored V last-layer/chunk"));

            const int compressed_chunk = kv_state->chunk_bound[snapshot_layer] - 1;
            if (compressed_chunk >= 0) {
                chunk_snapshots.push_back(capture_chunk_snapshot(
                    snapshot_layer, compressed_chunk, false, sample_count, "compressed K last-layer/boundary-chunk"));
                chunk_snapshots.push_back(capture_chunk_snapshot(
                    snapshot_layer, compressed_chunk, true, sample_count, "compressed V last-layer/boundary-chunk"));
            }
        }

        llama_batch batch = make_batch(tokens, n_past);

        auto [cur_ratio, n_comp_chunks, n_store_chunks, n_unload_layers] =
            core->swapout(context_remaining_in_mem);

        // swapout bottoms out at a small residual floor: the 8-bit quantization
        // metadata pool (mins/maxs, ~1/32 of the KV cache) is kept resident for
        // decompression and is never offloaded, so full eviction (target 0)
        // settles near ~2% rather than exactly 0. Treat the achievable floor as
        // success (mz_prefill does the same); only a large shortfall indicates a
        // real swapout failure (e.g. an I/O or weight-unload error).
        constexpr float swapout_floor_tol = 0.05f;
        if (cur_ratio > context_remaining_in_mem + swapout_floor_tol) {
            std::cerr << "ERROR: swapout fell short of target ratio. target="
                      << context_remaining_in_mem << ", actual=" << cur_ratio << "\n";
            llama_batch_free(batch);
            llama_sampler_free(sampler);
            delete core;
            return 1;
        } else if (cur_ratio > context_remaining_in_mem) {
            std::cerr << "[warn] swapout reached " << cur_ratio << " (target "
                      << context_remaining_in_mem
                      << "); residual is the resident quantization-metadata floor, proceeding.\n";
        }

        bool offload_compressed_kv = false;
        if (n_comp_chunks == max_comp_chunks &&
            n_store_chunks == max_store_chunks &&
            n_unload_layers == n_layers) {
            offload_compressed_kv = true;
        }

        long alloc_time = 0;
        long restore_time = 0;
        long prefill_time = 0;

        switch (prefill_type) {
            case 0:
                prefill_time = core->swapin_generate(ctx, batch, offload_compressed_kv);
                break;
            case 1:
            {
                auto ret = core->swapin_generate_no_alloc_overlap(ctx, batch, offload_compressed_kv);
                alloc_time = std::get<0>(ret);
                prefill_time = std::get<1>(ret);
                break;
            }
            case 2:
            {
                auto ret = core->swapin_generate_no_prefill_overlap(ctx, batch, offload_compressed_kv);
                restore_time = std::get<0>(ret);
                prefill_time = std::get<1>(ret);
                break;
            }
            case 3:
            {
                auto ret = core->swapin_generate_no_overlap(ctx, batch, offload_compressed_kv);
                alloc_time = std::get<0>(ret);
                restore_time = std::get<1>(ret);
                prefill_time = std::get<2>(ret);
                break;
            }
            default:
                std::cerr << "ERROR: invalid prefill_type: " << prefill_type << "\n";
                llama_batch_free(batch);
                llama_sampler_free(sampler);
                delete core;
                return 1;
        }

        prefill_times.push_back(prefill_time);

        std::cout << "Cycle " << cycle << " ratio after swapout: " << cur_ratio << "\n"
                  << "Restore time: " << restore_time / 1000 << " ms\n"
                  << "Alloc time: " << alloc_time / 1000 << " ms\n"
                  << "Prefill time: " << prefill_time / 1000 << " ms\n"
                  << "comp_chunk size: " << n_comp_chunks << "\n"
                  << "store_chunk size: " << n_store_chunks << "\n"
                  << "unload_layers size: " << n_unload_layers << "\n";

        const float * logits = llama_get_logits(ctx);
        for (int i = 0; i < 5; ++i) {
            std::printf("logits[%d] = %f\n", i, logits[i]);
        }

        if (verify_restore) {
            LogitSnapshot restored_logits = capture_logits(ctx, sampler, n_vocab);
            if (!print_logit_comparison(ctx, baseline_logits, restored_logits)) {
                llama_batch_free(batch);
                llama_sampler_free(sampler);
                delete core;
                return 1;
            }

            for (const ChunkSnapshot & snapshot : chunk_snapshots) {
                const bool require_exact = snapshot.label.find("stored") == 0;
                if (!compare_chunk_snapshot(snapshot, require_exact)) {
                    llama_batch_free(batch);
                    llama_sampler_free(sampler);
                    delete core;
                    return 1;
                }
            }
        } else {
            const int tok = llama_sampler_sample(sampler, ctx, -1);
            std::cout << "Sampled token: " << common_token_to_piece(ctx, tok) << "\n";
        }

        llama_batch_free(batch);

        if (!cleanup_appended_range(ctx, n_past)) {
            llama_sampler_free(sampler);
            delete core;
            return 1;
        }

        if (kv_state->n_cur_comp_chunks != 0 ||
            kv_state->n_cur_store_chunks != 0 ||
            weight->n_cur_unload_layers != 0) {
            std::cerr << "ERROR: mzcache state was not reset after cycle " << cycle
                      << " (comp=" << kv_state->n_cur_comp_chunks
                      << ", store=" << kv_state->n_cur_store_chunks
                      << ", unload=" << weight->n_cur_unload_layers << ")\n";
            llama_sampler_free(sampler);
            delete core;
            return 1;
        }

        std::cout << "Cycle " << cycle << " cleanup OK\n";
    }

    append_log(log_path, log_file_name, model_name, context_size_of_states,
               context_remaining_in_mem, repeats, tokens_per_step, prefill_times);

    llama_sampler_free(sampler);
    delete core;
    return 0;
#else
    llama_sampler_free(sampler);
    std::fprintf(stderr, "ERROR: MZCACHE_SVM_KV_CHUNK is not defined\n");
    return 1;
#endif
}
