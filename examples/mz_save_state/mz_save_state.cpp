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

#include <sys/types.h>
#include <sys/sysinfo.h>
#include <unistd.h>
// To test the inference fuctionality.
#define INFERENCE_TEST

#define MEMORY_LOGGING
#define MODEL_ARG   "-m"
#define NGL         "-ngl"
#define NGL_NUM     "100"
#define FLASHATTENTION "-fa"

/*
    ====== How to use ======
    mz_save_state <model.gguf> <context_size> <comp_chunk> <store_chunk> <unload_layer>

    Prefills <context_size> tokens and saves the KV state (.kv) that
    mz_prefill / mz_prefill_repeated / the Android app later restore.
*/

timespec get_monotonic_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

long diff_microseconds(const timespec &start, const timespec &end) {
    return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
}

extern std::vector<chunk_ptrs *> g_svm_chunk_ptrs;

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
        state_size = 65532;
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
                float context_ratio_to_restore,
                long prefill_us, double alloc_time_us, double restore_time_us,
                const std::string &model_name) {

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
        log_file << "# Context ratio to restore: " << context_ratio_to_restore << std::endl;
        log_file << "# Format:" << std::endl;
        log_file << "# context_ratio_to_restore "
                    "TTFT_us alloc_time_us restore_time_us alloc_time_us"
                 << std::endl;
        log_file << std::endl;
    }

    // Write data row
    log_file << context_ratio_to_restore << " "
             << prefill_us << " "
             << alloc_time_us << " "
             << restore_time_us << std::endl;
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

    if(argc < 5) {
        std::cerr << "Usage: " << argv[0] << " [options]\n";
        return 1;
    }
    // Parse command line arguments
    
    int context_size_of_states = atoi(argv[2]);
    std::string model_name = argv[1]; // model name (e.g., Qwen3-0.6B-FP16.gguf, now this is )
    int comp_chunk = atoi(argv[3]);
    int store_chunk = atoi(argv[4]);
    int unload_layer = atoi(argv[5]);

    bool mmap_usage = false;  // check if the 5th argument is "true"
    state_size = context_size_of_states + 32; // 32 is the padding size for KV cache


    common_params params;
    params.sampling.seed  = 1234;
    params.use_mmap      = mmap_usage;  // !!!mzCache: need behavior analysis
    params.n_ctx         = state_size;  // Context size from KV
    params.n_predict     = 32;   // Maximum number of tokens to generate
    params.n_batch       = state_size;
    params.n_ubatch      = 16;
    params.warmup        = false;

    // Dummy arguments for common_params_parse
    int fixed_argc = 6;
    char ** fixed_argv = new char*[fixed_argc];
    fixed_argv[0] = copy_str(argv[0]);
    fixed_argv[1] = copy_str(MODEL_ARG);
    fixed_argv[2] = copy_str(model_name);
    fixed_argv[3] = copy_str(NGL);
    fixed_argv[4] = copy_str(NGL_NUM);
    fixed_argv[5] = copy_str(FLASHATTENTION);
    // parse the fixed arguments
    if (!common_params_parse(fixed_argc, fixed_argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();

    long alloc_time = 0;
    long restore_time = 0;
    long prefill_time = 0;

    // 2. Initialize model and context
    auto init_res = common_init_from_params(params);
    llama_model*   model = init_res.model.get();
    llama_context* ctx   = init_res.context.get();
    if (!model || !ctx) {
        std::fprintf(stderr, "ERROR: model/context initialization failed\n");
        return 1;
    }

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
    // auto [cur_ratio, tmp_comp_chunks, tmp_store_chunks, tmp_unload_layers] = core->mzcache_profile_chunks_layers(context_ratio_to_restore);

    n_comp_chunks = comp_chunk;
    n_store_chunks = store_chunk;
    n_unload_layers = unload_layer;

    kv_state_ptr->mzcache_compress_chunks(n_comp_chunks);
    kv_state_ptr->mzcache_store_chunks(n_store_chunks);
    kv_state_ptr->mzcache_close_stored_files();
    weight_ptr->mzcache_unload_layers(n_unload_layers);

    bool offload_compressed_kv = false;

    if(n_comp_chunks == max_comp_chunks && n_store_chunks == max_store_chunks && n_unload_layers == n_layers) {
        offload_compressed_kv = true;
    }

    if (offload_compressed_kv) {
        for (int i = 0; i < n_layers; ++i)
            kv_state_ptr->mzcache_offload_arenas();
    }

    ThreadPool * thread_pool = core->get_thread_pool();

    // core->initialize_layer_sync(); // Ensure layer sync is initialized

    std::cout << "alloc weight" << "\n";
    timespec load_begin = get_monotonic_time();
    kv_state_ptr->mzcache_enqueue_load_chunks(n_store_chunks);
    weight_ptr->mzcache_reload_w_layers();

    thread_pool->wait_idle(CoreType::READ);
    timespec load_end = get_monotonic_time();
    double load_time_ms = diff_microseconds(load_begin, load_end)/ 1000; 

    //////////////////////
    timespec decomp_begin= get_monotonic_time();

    kv_state_ptr->mzcache_enqueue_decomp_chunks(n_comp_chunks);
    thread_pool->wait_idle(CoreType::DECOMP);
    
    timespec decomp_end= get_monotonic_time();
    double decomp_time_ms = diff_microseconds(decomp_begin, decomp_end)/ 1000; 

    kv_state_ptr->clean_after_decompress(n_comp_chunks);
    kv_state_ptr->clean_after_read(n_store_chunks);

    // std::cout << "load state" << "\n";
    // double t_load_ms = kv_state_ptr->mzcache_load_chunks_profile(n_store_chunks, n_comp_chunks).second;

    __fp16* last_store_chunk = (__fp16*)g_svm_chunk_ptrs[2 * layer + 1]->c[kv_state_ptr->cur_chunk_idx_to_store];
    __fp16* last_comp_chunk = (__fp16*)g_svm_chunk_ptrs[2 * layer + 1]->c[kv_state_ptr->cur_chunk_idx_to_compress - 1];


    std::cout << "current layer index: " << layer << "\n";
    std::cout << "current chunk index to store: " << kv_state_ptr->cur_chunk_idx_to_store << "\n";
    std::cout << "Last value chunk after loading:\n";
    for (int i = 0; i < 20; ++i) {
        float f = (float)(last_store_chunk[i]);
        std::cout << f << " ";
    }
    std::cout << "\n";



    /// save compressed KV cache
    std::string compressed_state_name = state_name.substr(0, state_name.find_last_of(".")) + "_compressed.kv";

    if (!llama_state_save_file(
        ctx,
        compressed_state_name.c_str(),
        session_tokens.data(),
        n_past
    )) {
        std::cerr << "ERROR: failed to save session to " << compressed_state_name << "\n";
    } else {
        std::cout << "[Info] Session saved with "
                    << n_past << " tokens\n";
    }

    delete kv_state_ptr;

#else

    printf("Error: MZCACHE_SVM_KV_CHUNK is not defined. Please define it to use chunked KV cache.\n");
    return 1;

#endif

    // 9. Cleanup

    printf("Freeing sampler...\n");

    llama_sampler_free(sampler);

    std::cout << "comp_chunk size " << comp_chunk << "\n";
    std::cout << "store_chunk size " << store_chunk << "\n";
    std::cout << "unload_layers size " << unload_layer << "\n";
    std::cout << "context size: " << context_size_of_states << "\n";
    // std::cout << "KV load time: " << t_load_ms << " ms\n";
    // std::cout << "KV decomp time: " << t_decomp_ms << " ms\n";
    // std::cout << "Weight load time: " << weight_load_time_ms << " ms\n";
    std::cout << "Load time " << load_time_ms << "\n";
    std::cout << "Decomp time " << decomp_time_ms << "\n";
    return 0;
}