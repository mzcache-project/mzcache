#include "arg.h"
#include "common.h"
#include "llama.h"
#include "json.hpp"
#include "llama-model-loader.h"
#include "mzCache_common.h"
#include "mzCache_profiler.h"
#include "mzCache_unloader.h"
#include "mzCache_loader.h"
#include "mzCache_threadpool.h"

#include <chrono>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include <sys/types.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <sched.h>
#include <iomanip>

#define INFERENCE_TEST
#define MODEL_ARG   "-m"
#define NGL         "-ngl"
#define NGL_NUM     "100"
#define FLASHATTENTION false

/*
    ====== How to use ======
    ./android-build/bin/mz_baseline_swap <model_path> <context_size> <weight_layers> <kv_layers>
    
    Example:
    ./android-build/bin/mz_baseline_swap /data/local/tmp/mzcache/Qwen3-0.6B-FP16.gguf 32700 28 28
*/

static const std::string STATES_DIR = "/data/local/tmp/mzcache/states/";
static std::string state_name_prefix = STATES_DIR;
int state_size;

// Power logging variables
static std::atomic<bool> power_logging_active{false};

// Power measurement data structure
struct PowerMeasurement {
    int64_t timestamp_us;
    long voltage_uV;
    long current_uA;
    double power_W;
};

static std::vector<PowerMeasurement> power_measurements;
static std::mutex power_mutex;

// Power logging function
void power_logger_thread() {
    // pin CPU affinity to CPU 5
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(5, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

    const std::string voltage_path = "/sys/class/power_supply/battery/voltage_now";
    const std::string current_path = "/sys/class/power_supply/battery/current_now";

    auto read_sysfs_value = [](const std::string& path) -> long {
        std::ifstream file(path);
        if (!file.is_open()) {
            return 0;
        }
        long value = 0;
        file >> value;
        return value;
    };

    std::cout << "Power logging started (CPU 5, 100ms interval)...\n";

    while (power_logging_active.load()) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // read voltage (uV)
        long voltage_uV = read_sysfs_value(voltage_path);
        
        // read current (uA)
        long current_uA = read_sysfs_value(current_path);
        
        // compute power (W)
        double voltage_V = voltage_uV / 1000000.0;
        double current_A = std::abs(current_uA) / 1000000.0;
        double power_W = voltage_V * current_A;
        
        // timestamp (microseconds since epoch)
        auto timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            start_time.time_since_epoch()
        ).count();
        
        // record in memory (thread-safe)
        {
            std::lock_guard<std::mutex> lock(power_mutex);
            power_measurements.push_back({timestamp_us, voltage_uV, current_uA, power_W});
        }
        
        // keep 100ms interval
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto sleep_duration = std::chrono::milliseconds(100) - elapsed;
        
        if (sleep_duration.count() > 0) {
            std::this_thread::sleep_for(sleep_duration);
        }
    }

    std::cout << "Power logging stopped. Total measurements: " << power_measurements.size() << "\n";
}

// function that saves power data to a CSV file
void save_power_log(const std::string& log_filename) {
    std::ofstream log_file(log_filename);
    if (!log_file.is_open()) {
        std::cerr << "ERROR: Failed to open power log file: " << log_filename << "\n";
        return;
    }

    // write CSV header
    log_file << "Timestamp_us,Voltage_uV,Current_uA,Power_W\n";
    
    // write data rows
    std::lock_guard<std::mutex> lock(power_mutex);
    for (const auto& measurement : power_measurements) {
        log_file << measurement.timestamp_us << ","
                 << measurement.voltage_uV << ","
                 << measurement.current_uA << ","
                 << std::fixed << std::setprecision(6) << measurement.power_W << "\n";
    }
    
    log_file.close();
    std::cout << "Power log saved to: " << log_filename << "\n";
}

static std::string set_state_name(llama_model* model, common_params& params, int context_size_of_states_) {
    std::string state_name = state_name_prefix;
    std::string arch = model->arch_name();

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

    std::string type = model->type_name();
    state_name += "_" + type;

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

    if(context_size_of_states_ == 2049){
        state_name += "_2049";
        state_size = 2081;
    }else if(context_size_of_states_ == 4097){
        state_name += "_4097";
        state_size = 4129;
    }else if(context_size_of_states_ == 8193){
        state_name += "_8193";
        state_size = 9225;
    }else if(context_size_of_states_ == 16385){
        state_name += "_16385";
        state_size = 16417;
    }else if(context_size_of_states_ == 32700){
        state_name += "_32700";
        state_size = 32732;
    }else if(context_size_of_states_ == 65500){
        state_name += "_65500";
        state_size = 65500;
    }

    state_name += ".kv";
    return state_name;
}

struct MemInfo {
    size_t mem_available_bytes = 0;
};

timespec get_monotonic_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

long diff_microseconds(const timespec &start, const timespec &end) {
    return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
}

MemInfo get_meminfo() {
    std::ifstream meminfo_file("/proc/meminfo");
    std::string line;
    std::unordered_map<std::string, size_t> meminfo;

    while (std::getline(meminfo_file, line)) {
        std::istringstream iss(line);
        std::string key;
        size_t value;
        std::string unit;

        if (!(iss >> key >> value >> unit))
            continue;

        if (!key.empty() && key.back() == ':') {
            key.pop_back();
        }

        meminfo[key] = value * 1024;
    }

    MemInfo result;
    if (meminfo.count("MemAvailable")) result.mem_available_bytes = meminfo["MemAvailable"];
    return result;
}

auto copy_str = [](const std::string& s) {
    char* result = new char[s.size() + 1];
    strcpy(result, s.c_str());
    return result;
};

size_t get_free_mem_bytes() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return (size_t)info.freeram * info.mem_unit;
    }
    return 0;
}

// (weight_layers, kv_layers) to reload for a target footprint (memory kept
// resident). load_ratio == footprint: 0 = full offload = drop/reload every layer
// (slowest TTFT); 1.0 = nothing offloaded = (0,0) (fastest). The reproduced
// metric is TTFT (includes KV + weight reload), reported on the "TTFT:" line.
//
// Qwen3-0.6B (28 layers) uses the author's hand-tuned "size to load %" table.
// Other models (n_layers != 28, e.g. EXAONE-4.0-1.2B with 30 layers) have no
// hand-tuned table, so we reconstruct one: evict the same fraction (1 - ratio)
// of both the weight and the KV layers. Because resident weight and resident KV
// then each equal `ratio` of their totals, the resident footprint equals `ratio`
// exactly for any per-layer weight/KV sizes — a clean, model-agnostic mapping.
static bool lookup_layers(int context_size, double ratio, int n_layers, int & w, int & kv) {
    if (n_layers == 28) {   // Qwen3-0.6B — author-tuned table
        struct Row { int size; double r; int w; int kv; };
        static const Row table[] = {
            {8193, 0.0, 28, 28},  {8193, 0.25, 21, 21}, {8193, 0.5, 14, 14}, {8193, 0.75, 2, 12}, {8193, 1.0, 0, 0},
            {16385, 0.0, 28, 28}, {16385, 0.25, 6, 28}, {16385, 0.5, 1, 20}, {16385, 0.75, 0, 10}, {16385, 1.0, 0, 0},
            {32700, 0.0, 28, 28}, {32700, 0.25, 20, 21}, {32700, 0.5, 14, 14}, {32700, 0.75, 2, 8}, {32700, 1.0, 0, 0},
        };
        for (const auto & row : table) {
            if (row.size == context_size && row.r > ratio - 1e-6 && row.r < ratio + 1e-6) {
                w = row.w; kv = row.kv; return true;
            }
        }
        return false;
    }
    // Computed equal-fraction reconstruction (context_size independent).
    for (double r : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        if (ratio > r - 1e-6 && ratio < r + 1e-6) {
            w = kv = (int) ((1.0 - r) * n_layers + 0.5);   // round half up
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    auto to_mb = [](size_t bytes) {
        return static_cast<double>(bytes) / (1024 * 1024);
    };

    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <context_size> <load_ratio>\n";
        std::cerr << "  context_size in {8193,16385,32700}; load_ratio in {0,0.25,0.5,0.75,1}\n";
        return 1;
    }

    std::string model_path = argv[1];
    int context_size_of_states = atoi(argv[2]);
    double load_ratio = atof(argv[3]);

    // Layer count per known model (the model is loaded later, so it cannot be
    // read from the GGUF here). Qwen3-0.6B = 28 (hand-tuned table),
    // EXAONE-4.0-1.2B = 30 (computed reconstruction). Detected from the path.
    int n_layers = 0;
    if (model_path.find("Qwen3-0.6B") != std::string::npos)        n_layers = 28;
    else if (model_path.find("EXAONE-4.0-1.2B") != std::string::npos) n_layers = 30;
    else {
        std::cerr << "ERROR: unknown model in path '" << model_path
                  << "'; add its layer count to exp1_partial_offload.cpp\n";
        return 1;
    }

    int number_of_layers_to_test = 0;
    int number_of_kv_layers_to_test = 0;
    if (!lookup_layers(context_size_of_states, load_ratio, n_layers, number_of_layers_to_test, number_of_kv_layers_to_test)) {
        std::cerr << "ERROR: no (weight,kv) entry for context_size=" << context_size_of_states
                  << " ratio=" << load_ratio << "\n";
        return 1;
    }

    // Extract model name from path
    std::filesystem::path p(model_path);
    std::string model_name = p.stem().string();

    // build the power log file name
    std::string power_log_filename = STATES_DIR + "power_log_" + model_name + "_" 
                                   + std::to_string(context_size_of_states) + "_"
                                   + std::to_string(number_of_layers_to_test) + "w_"
                                   + std::to_string(number_of_kv_layers_to_test) + "kv.csv";

    bool mmap_usage = false;
    
    if(context_size_of_states == 2049){
        state_size = 2081;
    }else if(context_size_of_states == 4097){
        state_size = 4129;
    }else if(context_size_of_states == 8193){
        state_size = 9225;
    }else if(context_size_of_states == 16385){
        state_size = 16417;
    }else if(context_size_of_states == 32700){
        state_size = 32732;
    }else if(context_size_of_states == 65500){
        state_size = 65532;
    }

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "     mzCache Baseline Swap Test\n";
    std::cout << "========================================\n";
    std::cout << "Model: " << model_name << "\n";
    std::cout << "Model Path: " << model_path << "\n";
    std::cout << "Context Size: " << context_size_of_states << "\n";
    std::cout << "Weight Layers to Test: " << number_of_layers_to_test << "\n";
    std::cout << "KV Layers to Test: " << number_of_kv_layers_to_test << "\n";
    std::cout << "========================================\n\n";

    common_params params;
    params.sampling.seed  = 1234;
    params.use_mmap      = mmap_usage;
    params.n_ctx         = state_size;
    params.n_predict     = 32;
    params.n_batch       = state_size;
    params.n_ubatch      = 16;
    params.warmup        = false;
    params.flash_attn    = FLASHATTENTION;
    params.layers_to_load = number_of_kv_layers_to_test;

    if(params.layers_to_load < 0){
        std::cout << "ERROR: Invalid layers_to_load: " << params.layers_to_load << "\n";
        return 1;
    }

    int fixed_argc = 5;
    char ** fixed_argv = new char*[fixed_argc];
    fixed_argv[0] = copy_str(argv[0]);
    fixed_argv[1] = copy_str(MODEL_ARG);
    fixed_argv[2] = copy_str(model_path);
    fixed_argv[3] = copy_str(NGL);
    fixed_argv[4] = copy_str(NGL_NUM);

    if (!common_params_parse(fixed_argc, fixed_argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();

    std::cout << "[1/7] Initializing model...\n";
    auto init_res = mzCache::mz_common_init_from_params(params);
    
    llama_model*   model = init_res.model.get();
    llama_context* ctx   = init_res.context.get();
    llama_model_loader* model_loader = init_res.model_loader;

    mzCache::mzProfiler profiler(model);
    mzCache::mzCacheUnloader unloader(model);
    mzCache::mzCacheLoader loader(model);

    if (!model || !ctx) {
        std::fprintf(stderr, "ERROR: model/context initialization failed\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto * kv_cache = static_cast<llama_kv_cache_unified *>(llama_get_memory(ctx));

    std::cout << "[2/7] Loading KV cache state...\n";
    std::string state_name = set_state_name(model, params, context_size_of_states);
    std::cout << "State file: " << state_name << "\n";
    
    size_t n_token_count = 0;
    if (std::filesystem::exists(state_name)) {
        std::vector<llama_token> tmp(params.n_ctx);
        auto t_load_start = std::chrono::high_resolution_clock::now();

        if (!llama_state_load_file(ctx, state_name.c_str(), tmp.data(), tmp.size(), &n_token_count)) {
            std::cerr << "ERROR: failed to load state file\n";
            return 1;
        }

        auto t_load_end = std::chrono::high_resolution_clock::now();
        double t_load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
        printf("Loaded state in %.2f ms\n", t_load_ms);
    } else {
        fprintf(stderr, "ERROR: state file '%s' does not exist\n", state_name.c_str());
        return 1;
    }

    std::cout << "\n[3/7] Unloading weights and KV cache...\n";
    MemInfo init = get_meminfo();
    size_t init_mem_bytes = init.mem_available_bytes;
    
    std::vector<int> layers_to_save;
    for(int i=0; i<number_of_layers_to_test; ++i){
        layers_to_save.push_back(i);
    }

    std::vector<int> kv_layers_to_save;
    for(int i=0; i<number_of_kv_layers_to_test; ++i){
        kv_layers_to_save.push_back(i);
    }

    std::vector<int> tensors_to_unload = profiler.get_weights_from_layers(layers_to_save);
    
    timespec unload_start = get_monotonic_time();
    if (!model->unload_buffers(tensors_to_unload)) {
        std::cout << "ERROR: model->unload_buffers failed\n";
        return 1;
    }
    timespec unload_end = get_monotonic_time();
    long unload_duration_us = diff_microseconds(unload_start, unload_end);

    if(!llama_state_free_buffers(ctx, number_of_kv_layers_to_test)){
        std::cout << "ERROR: llama_state_free_buffers failed\n";
        return 1;
    }

    sleep(1);
    MemInfo mem_unload = get_meminfo();
    size_t unload_mem_bytes = mem_unload.mem_available_bytes;

    std::cout << "[4/7] Loading KV cache...\n";
    
    // Power logging enable flag — turned on via the MZ_POWER_LOG=1 env var without recompiling
    // (the original 90248ae used a compile-time bool; changed for easier AE reproduction)
    bool enable_power_logging = (getenv("MZ_POWER_LOG") != nullptr);
    std::thread power_thread;
    if (enable_power_logging) {
        power_logging_active.store(true);
        power_thread = std::thread(power_logger_thread);
    } else {
        power_logging_active.store(false);
    }
    
    timespec TTFT_start = get_monotonic_time();
    double kv_alloc_time = 0;
    double kv_read_time = 0;
    
    if (std::filesystem::exists(state_name) && number_of_kv_layers_to_test != 0) {
        std::vector<llama_token> tmp(params.n_ctx);
        if(!llama_state_partial_alloc_load(ctx,
                                        state_name.c_str(),
                                        tmp.data(),
                                        tmp.size(),
                                        &n_token_count,
                                        number_of_kv_layers_to_test,
                                        kv_alloc_time,
                                        kv_read_time)){
            std::cerr << "ERROR: failed to reload KVs\n";
            return 1;
        }
    }
    timespec KV_load_done = get_monotonic_time();
    long KV_load_duration_us = diff_microseconds(TTFT_start, KV_load_done);

    std::cout << "[5/7] Loading weights...\n";
    timespec load_start = get_monotonic_time();
    double weight_alloc_time = 0;
    double weight_read_time = 0;
    
    if (!mzCache::mz_llama_model_load_buffers(model, model_loader, tensors_to_unload,
                                              weight_alloc_time, weight_read_time)) {
        std::cout << "ERROR: Failed to load buffers from model loader\n";
        return 1;
    }

    timespec load_end = get_monotonic_time();
    long weight_load_duration_us = diff_microseconds(load_start, load_end);

    MemInfo mem_load = get_meminfo();
    size_t load_mem_bytes = mem_load.mem_available_bytes;

    std::cout << "[6/7] Running inference...\n";
    timespec prefill_start = get_monotonic_time();

    std::string prompt = "Hello";
    int NCOLS = 8;
    int n_tok = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    while (n_tok < NCOLS) {
        prompt += " Hello";
        n_tok = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    }

    std::vector<llama_token> prompt_tokens(n_tok);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "ERROR: failed to tokenize the prompt\n");
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_init(prompt_tokens.size(), 0, 1);
    int32_t pos = static_cast<int32_t>(n_token_count);
    
    for (size_t i = 0; i < prompt_tokens.size(); ++i, ++pos) {
        common_batch_add(batch, prompt_tokens[i], pos, {0}, false);
    }
    
    int n_ctx = llama_n_ctx(ctx);
    int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    
    if (n_ctx_used + batch.n_tokens > n_ctx) {
        fprintf(stderr, "ERROR: context size exceeded\n");
        return 1;
    }
    
    batch.logits[batch.n_tokens - 1] = true;

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;
    timespec TTFT_end;
    timespec prefill_end;

    for (int i = 0; i < params.n_predict; ++i, ++pos) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "ERROR: failed to eval\n");
            return 1;
        }

        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "ERROR: failed to convert token to piece\n");
            return 1;
        }

        common_batch_clear(batch);
        common_batch_add(batch, new_token_id, pos, {0}, true);
        n_decode += 1;
        
        if(i==0){
            TTFT_end = get_monotonic_time();
            prefill_end = get_monotonic_time();
            break;
        }
    }

    long TTFT_duration_us = diff_microseconds(TTFT_start, TTFT_end);
    long prefill_duration_us = diff_microseconds(prefill_start, prefill_end);

    const auto t_main_end = ggml_time_us();

    // stop power logging
    power_logging_active.store(false);
    if (power_thread.joinable()) {
        power_thread.join();
    }

    // save power log to CSV file
    save_power_log(power_log_filename);

    llama_batch_free(batch);
    llama_sampler_free(smpl);

    std::cout << "\n[7/7] Results\n";
    std::cout << "========================================\n";
    std::cout << "Configuration:\n";
    std::cout << "  Model: " << model_name << "\n";
    std::cout << "  Context Size: " << context_size_of_states << "\n";
    std::cout << "  Weight Layers: " << number_of_layers_to_test << "\n";
    std::cout << "  KV Layers: " << number_of_kv_layers_to_test << "\n";
    std::cout << "  Power Log: " << power_log_filename << "\n\n";
    
    std::cout << "Timing Results (microseconds):\n";
    std::cout << "  TTFT: " << TTFT_duration_us << " us (" << TTFT_duration_us/1000.0 << " ms)\n";
    std::cout << "  Prefill: " << prefill_duration_us << " us (" << prefill_duration_us/1000.0 << " ms)\n";
    std::cout << "  KV Load: " << KV_load_duration_us << " us (" << KV_load_duration_us/1000.0 << " ms)\n";
    std::cout << "    - Alloc: " << static_cast<long>(kv_alloc_time) << " us\n";
    std::cout << "    - Read: " << static_cast<long>(kv_read_time) << " us\n";
    std::cout << "  Weight Load: " << weight_load_duration_us << " us (" << weight_load_duration_us/1000.0 << " ms)\n";
    std::cout << "    - Alloc: " << static_cast<long>(weight_alloc_time) << " us\n";
    std::cout << "    - Read: " << static_cast<long>(weight_read_time) << " us\n";
    std::cout << "  Total Alloc: " << static_cast<long>(kv_alloc_time + weight_alloc_time) << " us\n";
    std::cout << "  Total Read: " << static_cast<long>(kv_read_time + weight_read_time) << " us\n\n";
    
    std::cout << "Memory Usage (MiB):\n";
    std::cout << "  Initial Available: " << to_mb(init_mem_bytes) << " MiB\n";
    std::cout << "  After Unload: " << to_mb(unload_mem_bytes) << " MiB\n";
    std::cout << "  After Load: " << to_mb(load_mem_bytes) << " MiB\n";
    std::cout << "  Memory Freed: " << to_mb(unload_mem_bytes - init_mem_bytes) << " MiB\n";
    std::cout << "  Memory Used: " << to_mb(unload_mem_bytes - load_mem_bytes) << " MiB\n";
    std::cout << "========================================\n\n";

    // CSV format for easy parsing
    std::cout << "CSV Format:\n";
    std::cout << "model,context_size,weight_layers,kv_layers,ttft_us,prefill_us,kv_load_us,weight_load_us,alloc_us,read_us\n";
    std::cout << model_name << "," 
              << context_size_of_states << ","
              << number_of_layers_to_test << ","
              << number_of_kv_layers_to_test << ","
              << TTFT_duration_us << ","
              << prefill_duration_us << ","
              << KV_load_duration_us << ","
              << weight_load_duration_us << ","
              << static_cast<long>(kv_alloc_time + weight_alloc_time) << ","
              << static_cast<long>(kv_read_time + weight_read_time) << "\n";

    return 0;
}