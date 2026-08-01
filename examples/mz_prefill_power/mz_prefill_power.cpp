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

#include <chrono>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <atomic>
#include <mutex>
#include <iomanip>

#include <sys/types.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <sched.h>

timespec get_monotonic_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

long diff_microseconds(const timespec &start, const timespec &end) {
    return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
}

extern std::vector<chunk_ptrs *> g_svm_chunk_ptrs;

// ---------------------------------------------------------------------------
// Power sampling (ported from the v0.4/v0.6 mz_prefill_power example).
// A dedicated thread pinned to a little core reads the battery fuel gauge
// (voltage_now/current_now) every 100 ms while the swap-in + prefill runs,
// then dumps Timestamp_us,Voltage_uV,Current_uA,Power_W rows to a CSV under
// STATES_DIR. Power_W = V * |I| (battery discharge power).
// ---------------------------------------------------------------------------
static std::atomic<bool> power_logging_active{false};

struct PowerMeasurement {
    int64_t timestamp_us;
    long voltage_uV;
    long current_uA;
    double power_W;
};

static std::vector<PowerMeasurement> power_measurements;
static std::mutex power_mutex;

void power_logger_thread() {
    // Pin the sampler off the benchmark cores (CPU 6).
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(6, &cpuset);
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

    std::cout << "Power logging started (CPU 6, 100ms interval)...\n";

    while (power_logging_active.load()) {
        auto start_time = std::chrono::high_resolution_clock::now();

        long voltage_uV = read_sysfs_value(voltage_path);
        long current_uA = read_sysfs_value(current_path);

        double voltage_V = voltage_uV / 1000000.0;
        double current_A = std::abs(current_uA) / 1000000.0;
        double power_W = voltage_V * current_A;

        auto timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            start_time.time_since_epoch()
        ).count();

        {
            std::lock_guard<std::mutex> lock(power_mutex);
            power_measurements.push_back({timestamp_us, voltage_uV, current_uA, power_W});
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto sleep_duration = std::chrono::milliseconds(100) - elapsed;

        if (sleep_duration.count() > 0) {
            std::this_thread::sleep_for(sleep_duration);
        }
    }

    std::cout << "Power logging stopped. Total measurements: " << power_measurements.size() << "\n";
}

void save_power_log(const std::string& log_filename) {
    std::ofstream log_file(log_filename);
    if (!log_file.is_open()) {
        std::cerr << "ERROR: Failed to open power log file: " << log_filename << "\n";
        return;
    }

    log_file << "Timestamp_us,Voltage_uV,Current_uA,Power_W\n";

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

// Change the state_name definition to use states directory
static const std::string STATES_DIR = "/data/local/tmp/mzcache/states/";
static std::string state_name_prefix = STATES_DIR;
int state_size;

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

    if(context_size_of_states_ == 2049){
        state_name += "_2049";
    }else if(context_size_of_states_ == 4097){
        state_name += "_4097";
    }else if(context_size_of_states_ == 8193){
        state_name += "_8193";
    }else if(context_size_of_states_ == 16385){
        state_name += "_16385";
    }else if(context_size_of_states_ == 32700){
        state_name += "_32700";
    }else if(context_size_of_states_ == 65500){
        state_name += "_65500";
    }

    state_name += ".kv";

    return state_name;
}


struct MemInfo {
    // size_t mem_free_bytes = 0;
    size_t mem_available_bytes = 0;
    // size_t cached_bytes = 0;
};

// logger
void append_log(const std::string &log_path, const std::string &log_file_name, int context_size_of_states_,
                float context_remaining_in_mem,
                long prefill_us, double alloc_time_us, double restore_time_us, int n_comp_chunks, int n_store_chunks, 
                int n_unload_layers, const std::string &model_name) {

    if (log_path.empty() || log_file_name.empty()) {
        std::cout << "[mzCache Log Output]" << std::endl;
        std::cout << "  Model Name              : " << model_name << std::endl;
        std::cout << "  Context Size (states)   : " << context_size_of_states_ << std::endl;
        std::cout << "  Context Remaining Ratio : " << context_remaining_in_mem << std::endl;
        std::cout << "  Prefill Time (ms)       : " << (double)prefill_us / 1000 << std::endl;
        std::cout << "  Alloc Time (ms)         : " << (double)alloc_time_us / 1000 << std::endl;
        std::cout << "  Restore Time (ms)       : " << (double)restore_time_us / 1000 << std::endl;
        std::cout << "  #Compressed Chunks      : " << n_comp_chunks << std::endl;
        std::cout << "  #Stored Chunks          : " << n_store_chunks << std::endl;
        std::cout << "  #Unloaded Layers        : " << n_unload_layers << std::endl;
        std::cout << std::endl;
        return;
    }

    std::string full_path = log_path + "/" + log_file_name;

    bool file_exists = std::filesystem::exists(full_path);

    std::ofstream log_file(full_path, std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Error: Cannot open log file at " << full_path << std::endl;
        return;
    }

    // On first creation, write experiment metadata header
    if (!file_exists) {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);

        log_file << "# mzCache Experiment Log, Minsung Kim" << std::endl;
        log_file << "# Commit hash: not set" << std::endl;
        log_file << "# Start Time: " << std::ctime(&now_c);  // includes trailing newline
        log_file << "# Model Name: " << model_name << std::endl;
        log_file << "# Context Size: " << context_size_of_states_ << std::endl;
        log_file << "# Context ratio to restore: " << context_remaining_in_mem << std::endl;
        log_file << "# Format:" << std::endl;
        log_file << "context_remaining_in_mem prefill_us alloc_time_us restore_time_us comp_kv_load_us comp_chunks store_chunks unload_layers"
                 << std::endl;
        log_file << std::endl;
    }

    // Write data row
    log_file << context_remaining_in_mem << " "
             << prefill_us << " "
             << alloc_time_us << " "
             << restore_time_us  << " "
             << n_comp_chunks << " "
             << n_store_chunks << " "
             << n_unload_layers << " "
             << std::endl;
    log_file.close();
}


// Read /proc/meminfo
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

        // Strip trailing ':' from the key
        if (!key.empty() && key.back() == ':') {
            key.pop_back();
        }

        meminfo[key] = value * 1024; // kB → Byte
    }

    MemInfo result;
    // if (meminfo.count("MemFree"))      result.mem_free_bytes = meminfo["MemFree"];
    if (meminfo.count("MemAvailable")) result.mem_available_bytes = meminfo["MemAvailable"];
    // if (meminfo.count("Cached"))       result.cached_bytes = meminfo["Cached"];

    return result;
}

auto copy_str = [](const std::string& s) {
    char* result = new char[s.size() + 1];
    strcpy(result, s.c_str());
    return result;
};

// Helper: get free system memory in bytes (Linux)
size_t get_free_mem_bytes() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return (size_t)info.freeram * info.mem_unit;
    }
    return 0;
}

int main(int argc, char** argv) {

    auto to_mb = [](size_t bytes) {
        return static_cast<double>(bytes) / (1024 * 1024);
    };

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                << " <model_name> <context_size_of_states> <context_remaining_in_mem> [prefill_type] [log_path] [log_file_name]\n\n"
                << "Arguments:\n"
                << "  model_name               Model filename (e.g., Qwen3-0.6B-FP16.gguf)\n"
                << "  context_size_of_states   Number of context states (e.g., 4097, 8193, 16385, 32700)\n"
                << "  context_remaining_in_mem Float between 0 and 1, ratio of context kept in memory\n"
                << "  prefill_type             (Optional) Prefill overlap type (default: 0)\n"
                << "                           0 = prefill\n"
                << "                           1 = no_alloc_overlap\n"
                << "                           2 = no_prefill_overlap\n"
                << "                           3 = no_overlap\n"
                << "  log_path                 (Optional) Directory to save log file\n"
                << "  log_file_name            (Optional) Name of log file\n"
                << std::endl;
        return 1;
    }

    // Parse required args
    std::string model_name        = argv[1];
    int context_size_of_states    = atoi(argv[2]);
    float context_remaining_in_mem = atof(argv[3]);
    float context_remaining_in_mem_log = context_remaining_in_mem;

    // prefill_type (optional, default 0)
    int prefill_type = (argc > 4) ? atoi(argv[4]) : 0;

    // log_path, log_file_name (optional)
    std::string log_path      = (argc > 5) ? argv[5] : "";
    std::string log_file_name = (argc > 6) ? argv[6] : "";

    std::cout << "Model: " << model_name << "\n";
    std::cout << "Context size: " << context_size_of_states << "\n";
    std::cout << "Context remaining ratio: " << context_remaining_in_mem << "\n";
    std::cout << "Prefill type: " << prefill_type << "\n";

    // Power log destination (basename of the model path keeps '/' out of the name)
    const std::string power_log_filename = STATES_DIR + "power_log_"
        + std::filesystem::path(model_name).filename().string() + "_"
        + std::to_string(context_size_of_states) + "_"
        + std::to_string(static_cast<int>(context_remaining_in_mem * 100)) + "pct_"
        + "prefill" + std::to_string(prefill_type) + ".csv";

    state_size = context_size_of_states + 32; // 32 is the padding size for KV cache

    common_params params;
    params.sampling.seed  = 1234;
    params.use_mmap      = false;  // !!!mzCache: need behavior analysis
    params.n_ctx         = state_size;  // Context size from KV
    params.n_predict     = 32;   // Maximum number of tokens to generate
    params.n_batch       = state_size;
    params.n_ubatch      = 16;
    params.warmup        = false;
    params.n_gpu_layers  = 100; // offload all layers
    params.flash_attn    = true;
    params.model.path = model_name;

    common_init();
    // 2. Initialize model and context
    auto init_res = common_init_from_params(params);
    llama_model*   model = init_res.model.get();
    llama_context* ctx   = init_res.context.get();
    if (!model || !ctx) {
        std::fprintf(stderr, "ERROR: model/context initialization failed\n");
        return 1;
    }

    long alloc_time = 0;
    long restore_time = 0;
    long prefill_time = 0;

    std::string state_name = set_state_name(model, params, context_size_of_states);


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


    std::string follow_prompt = "Hello";
    int NCOLS = 8;

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

    MZ_LOG_INFO("Token Count: %d", n_tok);
    for(int i = n_tok; i < NCOLS; i++) {
        follow_prompt += " Hello";
    }

    std::vector<llama_token> follow_tokens(NCOLS);
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



    // Get KV
    auto * kv_cache = static_cast<llama_kv_cache_unified *>(llama_get_memory(ctx));

    // Get Weight

#ifdef MZCACHE_SVM_KV_CHUNK
    const int   n_layers = kv_cache->layers_k_chunks.size();
    const int   layer    = n_layers - 1;

    struct ggml_tensor_ptrs k_tensors = kv_cache->layers_k_chunks[layer];
    const int   hdim   = model->hparams.n_embd_k_gqa(layer);
    const int   n_chunks_per_tensor = (n_past + TOKENS_PER_CHUNK - 1) / TOKENS_PER_CHUNK;
    const size_t stride = ggml_row_size(GGML_TYPE_F16, hdim);

    /////// Compress and Decompress Example ///////
    std::cout << "Initializing mzcache_core with " << n_layers << " layers, hidden dimension " << hdim
              << ", and " << n_chunks_per_tensor << " chunks per tensor.\n";

    mzcache_core * core = new mzcache_core(n_layers, hdim, n_chunks_per_tensor, model);
    std::cout << "mzcache_core initialized successfully.\n";

    mzcache_kv_state * kv_state_ptr = core->get_kv_state();
    std::cout << "mzcache_kv_state pointer obtained successfully.\n";

    mzcache_weight * weight_ptr = core->get_weight();
    std::cout << "mzcache_weight pointer obtained successfully.\n";

    std::cout << "kv_state_ptr->cur_chunk_idx_to_store: " << kv_state_ptr->cur_chunk_idx_to_store << "\n";

    __fp16* last_store_chunk = (__fp16*)g_svm_chunk_ptrs[2 * layer + 1]->c[kv_state_ptr->cur_chunk_idx_to_store];
    __fp16* last_comp_chunk = (__fp16*)g_svm_chunk_ptrs[2 * layer + 1]->c[kv_state_ptr->cur_chunk_idx_to_compress];


    // std::cout << "current layer index: " << layer << "\n";
    // std::cout << "current chunk index to store: " << kv_state_ptr->cur_chunk_idx_to_store << "\n";
    // std::cout << "Last value chunk after loading:\n";
    // for (int i = 0; i < 20; ++i) {
    //     float f = (float)(last_store_chunk[i]);
    //     std::cout << f << " ";
    // }
    // std::cout << "\n";

    // std::cout << "cuurent chunk index to compress: " << kv_state_ptr->cur_chunk_idx_to_compress << "\n";
    // std::cout << "Last value chunk before compressing:\n";
    // for (int i = 0; i < 20; ++i) {
    //     float f = (float)(last_comp_chunk[i]);
    //     std::cout << f << " ";
    // }
    // std::cout << "\n";
    


    ///////// parallel decompress, load example /////////

    int n_comp_chunks;
    int n_store_chunks;
    int n_unload_layers;

    int max_comp_chunks = 0;
    int max_store_chunks = 0;

    for(int i = 0; i < n_layers; ++i) {
        max_comp_chunks += kv_state_ptr->chunk_bound[i];
        max_store_chunks += n_chunks_per_tensor - kv_state_ptr->chunk_bound[i];

        std::cout << "kv_state_ptr->chunk_bound[" << i << "]: " << kv_state_ptr->chunk_bound[i] << "\n";
    }
    std::cout << "max_comp_chunks: " << max_comp_chunks << ", max_store_chunks: " << max_store_chunks << "\n";


    // float target_ratio;
    // std::cout << "Enter target compression ratio (0.0 to 1.0): ";
    // std::cin >> target_ratio;


    // auto [cur_ratio, tmp_comp_chunks, tmp_store_chunks, tmp_unload_layers] = core->mzcache_profile_chunks_layers(target_ratio);
    bool real_time_profiling = true;

    if(real_time_profiling) {

        auto [cur_ratio, tmp_comp_chunks, tmp_store_chunks, tmp_unload_layers] = core->swapout(context_remaining_in_mem);

        // The realtime function already performed the operations.
        // The returned values are deltas for this call, which can be used for logging or verification.
        n_comp_chunks = tmp_comp_chunks;
        n_store_chunks = tmp_store_chunks;
        n_unload_layers = tmp_unload_layers;

    }
    else {
        auto [cur_ratio, tmp_comp_chunks, tmp_store_chunks, tmp_unload_layers] = core->mzcache_profile_chunks_layers(context_remaining_in_mem);

        n_comp_chunks = tmp_comp_chunks;
        n_store_chunks = tmp_store_chunks;
        n_unload_layers = tmp_unload_layers;

        kv_state_ptr->mzcache_compress_chunks(n_comp_chunks);
        kv_state_ptr->mzcache_store_chunks(n_store_chunks);
        kv_state_ptr->mzcache_close_stored_files();
        weight_ptr->mzcache_unload_layers(n_unload_layers);
    }

    bool offload_compressed_kv = false;

    if(n_comp_chunks == max_comp_chunks && n_store_chunks == max_store_chunks && n_unload_layers == n_layers) {
        offload_compressed_kv = true;
    }
    
    // Start the 100ms power sampler for the duration of the swap-in + prefill.
    power_logging_active.store(true);
    std::thread power_thread(power_logger_thread);

    switch(prefill_type) {
        case 0:
        {
            // ret = core->swapin_generate(ctx, batch2);
            long ret = core->swapin_generate(ctx, batch2, offload_compressed_kv);
            prefill_time = ret;
            break;
        }
        case 1:
        {
            std::tuple<long, long> ret = core->swapin_generate_no_alloc_overlap(ctx, batch2, offload_compressed_kv);
            alloc_time = std::get<0>(ret); //alloc
            prefill_time = std::get<1>(ret); //restore+prefill
            break;
        }
        case 2:
        {
            std::tuple<long, long> ret = core->swapin_generate_no_prefill_overlap(ctx, batch2, offload_compressed_kv);
            restore_time = std::get<0>(ret); //alloc+restore
            prefill_time = std::get<1>(ret); //prefill
            break;
        }
        case 3:
        {
            std::tuple<long, long, long> ret = core->swapin_generate_no_overlap(ctx, batch2, offload_compressed_kv);
            alloc_time = std::get<0>(ret); //alloc
            restore_time = std::get<1>(ret); //restore
            prefill_time = std::get<2>(ret); //prefill
            break;
        }
        default:
            std::cerr << "Invalid prefill type: " << prefill_type << std::endl;
            power_logging_active.store(false);
            if (power_thread.joinable()) power_thread.join();
            return 1;
    }

    // Stop the sampler right after the measured section and persist the trace.
    power_logging_active.store(false);
    if (power_thread.joinable()) {
        power_thread.join();
    }
    save_power_log(power_log_filename);

    std::cout << "Restore time: " << restore_time / 1000 << " ms\n";
    std::cout << "Alloc time: " << alloc_time / 1000 << " ms\n";
    std::cout << "Prefill time: " << prefill_time / 1000 << " ms\n";
    std::cout << "comp_chunk size " << n_comp_chunks << "\n";
    std::cout << "tmp_store_chunk size " << n_store_chunks << "\n";
    std::cout << "unload_layers size " << n_unload_layers << "\n";


    // last_store_chunk = (__fp16*)g_svm_chunk_ptrs[2 * layer + 1]->c[kv_state_ptr->cur_chunk_idx_to_store];
    // std::cout << "current layer index: " << layer << "\n";
    // std::cout << "current chunk index to store: " << kv_state_ptr->cur_chunk_idx_to_store << "\n";
    // std::cout << "Last value chunk after loading:\n";
    // for (int i = 0; i < 20; ++i) {
    //     float f = (float)(last_store_chunk[i]);
    //     std::cout << f << " ";
    // }
    // std::cout << "\n";


    // std::cout << "current chunk index to compress: " << kv_state_ptr->cur_chunk_idx_to_compress << "\n";
    // last_comp_chunk = (__fp16*)g_svm_chunk_ptrs[2 * layer + 1]->c[kv_state_ptr->cur_chunk_idx_to_compress];
    // std::cout << "Last value chunk after compressing:\n";
    // for (int i = 0; i < 20; ++i) {
    //     float f = (float)(last_comp_chunk[i]);
    //     std::cout << f << " ";
    // }
    // std::cout << "\n";

    
    int tok = llama_sampler_sample(sampler, ctx, -1);
    std::cout << common_token_to_piece(ctx, tok) << std::endl;

    const float * logits = llama_get_logits(ctx);

    for (int i = 0; i < 20; ++i) {
        printf("logits[%d] = %f\n", i, logits[i]);
    }



    delete kv_state_ptr;

#else

    printf("Error: MZCACHE_SVM_KV_CHUNK is not defined. Please define it to use chunked KV cache.\n");
    return 1;

#endif

    // 9. Cleanup

    printf("Freeing sampler...\n");

    llama_sampler_free(sampler);

    append_log(log_path, log_file_name, context_size_of_states, context_remaining_in_mem_log, 
           prefill_time, alloc_time, restore_time, n_comp_chunks, n_store_chunks, n_unload_layers, model_name);
    
    return 0;
}
