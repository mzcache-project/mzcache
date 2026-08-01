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

// To test the inference fuctionality.
#define INFERENCE_TEST

#define MEMORY_LOGGING
#define MODEL_ARG   "-m"


#define MODEL_PATH  "/data/local/tmp/mzcache/Qwen3-0.6B-FP16.gguf"
// QWEN has 28 layers

// #define MODEL_PATH  "Llama-3.2-1B-Instruct-FP16.gguf"
// LLAMA has 16 layers

// #define MODEL_PATH  "/data/local/tmp/mskim/EXAONE-4.0-1.2B-FP16.gguf"
// EXAONE has 30 layers


#define NGL         "-ngl"
#define NGL_NUM     "100"
#define FLASHATTENTION false

/*
    ====== How to use ======
    run ./android-build/bin/mz_baseline_swap 28 28 28 32700 QWEN /data/local/tmp/mskim log

    Input params: model, state dir, layers to offload/load, log path

*/

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
        state_size = 65500;
    }

    state_name += ".kv";

    return state_name;
}


struct MemInfo {
    // size_t mem_free_bytes = 0;
    size_t mem_available_bytes = 0;
    // size_t cached_bytes = 0;
};

// timer
timespec get_monotonic_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

long diff_microseconds(const timespec &start, const timespec &end) {
    return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
}

// logger
void append_log(const std::string &log_path, const std::string &log_file_name, int context_size_of_states_,
                int layer_num, int kv_layer_num, size_t init_mem, size_t unload_mem, size_t load_mem,
                long ttft_us, long prefill_us, long alloc_us, long read_us, long kv_load_time_us, long weight_load_time_us, 
                const std::string &model_name) {

    std::string full_path = log_path + "/" + log_file_name;

    bool file_exists = std::filesystem::exists(full_path);

    std::ofstream log_file(full_path, std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Error: Cannot open log file at " << full_path << std::endl;
        return;
    }

    // on first creation, write experiment info
    if (!file_exists) {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);

        log_file << "# mzCache Experiment Log, Minsung Kim" << std::endl;
        log_file << "# Commit hash: not set" << std::endl;
        log_file << "# Start Time: " << std::ctime(&now_c);  // includes trailing newline
        log_file << "# Model: " << MODEL_PATH << std::endl;
        log_file << "# Context Size: " << context_size_of_states_ << std::endl;
        log_file << "# Format:" << std::endl;
        log_file << "layer_num kv_layer_num "
                    "TTFT_us prefill_us alloc_us read_us KV_load_time_us weight_load_time_us"
                 << std::endl;
        log_file << std::endl;
    }

    // write data row
    log_file << layer_num << " "
             << kv_layer_num << " "
             << ttft_us << " "
             << prefill_us << " "
             << alloc_us << " "
             << read_us << " "
             << kv_load_time_us << " "
             << weight_load_time_us << std::endl;
    log_file.close();
}


// read /proc/meminfo
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

        // strip trailing ':' from key
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

    if(argc < 6) {
        std::cerr << "Usage: " << argv[0] << " [options]\n";
        return 1;
    }
    // Parse command line arguments
    
    int number_of_layers_to_test = atoi(argv[1]); // number of test to weight unload/load 
    int number_of_kv_layers_to_test = atoi(argv[2]); // number of test to KV unload/load
    int number_of_layers_in_model  = atoi(argv[3]); // number of layers(MHA) in model
    int context_size_of_states = atoi(argv[4]);
    std::string model_name = argv[5]; // model name (e.g., Qwen3-0.6B-FP16.gguf, now this is )
    std::string log_path = argv[6];
    std::string log_file_name = argv[7];

    bool mmap_usage = false;  // check if the 5th argument is "true"
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


    std::cout << "################ Test start ################" << "\n";
    std::cout << "Number of layers to test: " << number_of_layers_to_test << "\n";
    std::cout << "Number of layers in model: " << number_of_layers_in_model << "\n";
    std::cout << model_name << " " << log_path << " " << log_file_name << " " << mmap_usage << "\n";  

    common_params params;
    params.sampling.seed  = 1234;
    params.use_mmap      = mmap_usage;  // !!!mzCache: need behavior analysis
    // params.n_ctx         = 32768;  // Very large context size
    params.n_ctx         = state_size;  // Context size from KV
    params.n_predict     = 32;   // Maximum number of tokens to generate
    params.n_batch       = state_size;
    params.n_ubatch      = 16;
    params.warmup        = false;
    params.flash_attn    = FLASHATTENTION;
    params.layers_to_load = number_of_kv_layers_to_test;
    if(params.layers_to_load < 0){
        std::cout << "Wrong params, layers_to_preload set " <<  params.layers_to_load \
                  << "\n";
        return 0;
    }

    // Dummy arguments for common_params_parse
    int fixed_argc = 5;
    char ** fixed_argv = new char*[fixed_argc];
    fixed_argv[0] = copy_str(argv[0]);
    fixed_argv[1] = copy_str(MODEL_ARG);
    fixed_argv[2] = copy_str(MODEL_PATH);
    fixed_argv[3] = copy_str(NGL);
    fixed_argv[4] = copy_str(NGL_NUM);
    // fixed_argv[5] = copy_str(FLASHATTENTION);
    // parse the fixed arguments
    if (!common_params_parse(fixed_argc, fixed_argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();

    // Minsung baseline implemantation todo.    
    // Todo.1 find where allocation is sized by n_ctx
    // Todo.2 switch to partial alloc (requires adjusting max_size and passing tensor indices)
    // Todo.3 partial load 

    // 2. Initialize model and context
    // auto init_res = common_init_from_params(params);
    std::cout << "===================== Init ======================" << "\n";
    auto init_res = mzCache::mz_common_init_from_params(params);
    std::cout << "===================== Init done ======================" << "\n";
    
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

    // Get vocabulary
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto * kv_cache = static_cast<llama_kv_cache_unified *>(llama_get_memory(ctx));

    // // Load state from file
    std::string state_name = set_state_name(model, params, context_size_of_states);
    size_t n_token_count = 0;
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
    } 
    else {
        fprintf(stderr, "ERROR: state file '%s' does not exist\n", state_name.c_str());
        return 1;
    }
  
    std::cout << "===================== Model load ======================" << "\n";
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
    std::vector<int> kv_tensors_to_unload = profiler.get_weights_from_layers(kv_layers_to_save);
    std::cout << "===================== Unload " << layers_to_save.size() \
              << " layers ======================" << "\n";
    timespec unload_start = get_monotonic_time();
    if (!model->unload_buffers(tensors_to_unload)) {
        std::cout << "model->unload_buffers returned error" << "\n";
        return 0;
    }
    timespec unload_end = get_monotonic_time();
    long unload_duration_us = diff_microseconds(unload_start, unload_end);

    std::cout << "======= FREE KVS =======" << "\n";
    if(!llama_state_free_buffers(ctx, number_of_kv_layers_to_test)){
        std::cout << "llama_state_free_buffers returned ERROR" << "\n";
        return 1;
    }

    sleep(1);
    MemInfo mem_unload = get_meminfo();
    size_t unload_mem_bytes = mem_unload.mem_available_bytes;

    std::cout << "======= ALLOC & READ KVS =======" << "\n";
    timespec TTFT_start = get_monotonic_time();
    double kv_alloc_time = 0;
    double kv_read_time = 0;
    if (std::filesystem::exists(state_name) && number_of_kv_layers_to_test != 0) {
        // Load tokens into temporary buffer
        std::vector<llama_token> tmp(params.n_ctx);
        if(!llama_state_partial_alloc_load(ctx,
                                        state_name.c_str(),
                                        tmp.data(),
                                        tmp.size(),
                                        &n_token_count,
                                        number_of_kv_layers_to_test,
                                        kv_alloc_time,
                                        kv_read_time)){
            std::cerr << "ERROR: failed to reload KVs" << "\n";
            return 1;
        }
    }
    timespec KV_load_done = get_monotonic_time();
    long KV_load_duration_us = diff_microseconds(TTFT_start, KV_load_done);

    // 3. measure load
    std::cout << "===================== LOAD WEIGHTS ======================" << "\n";
    timespec load_start = get_monotonic_time();

    double weight_alloc_time = 0;
    double weight_read_time = 0;
    if (!mzCache::mz_llama_model_load_buffers(model, model_loader, tensors_to_unload,
                                              weight_alloc_time, weight_read_time)) {
        std::cout << "ERROR: Failed to load buffers from model loader." << "\n";
    }
    

    timespec load_end = get_monotonic_time();
    long weight_load_duration_us = diff_microseconds(load_start, load_end);
    std::cout << "========= Load done =========" << "\n";
    

    MemInfo mem_load = get_meminfo();
    size_t load_mem_bytes = mem_load.mem_available_bytes;

    // 4. write log (argv[3] = log_path, argv[4] = log_file_name)
    int layer_num = number_of_layers_to_test;

    // completion output
    
   
    std::cout << "============================================================" << "\n";
    std::cout << "===================== Inference Start ======================" << "\n";
    std::cout << "============================================================" << "\n";
    // tokenize the prompt
    timespec prefill_start = get_monotonic_time();


    // find the number of tokens in the prompt
    std::string prompt = "Hello";
    int NCOLS = 8;
    int n_tok = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true); // n_tok: actual number of tokens needed

    // grow the prompt length
    while (n_tok < NCOLS) {
        prompt += " Hello";
        n_tok = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    }

    std::cout << "Input:" << prompt << "\n";

    // reallocate the vector to the exact size
    std::vector<llama_token> prompt_tokens(n_tok);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    
    llama_batch batch = llama_batch_init(prompt_tokens.size(), 0, 1);
    int32_t pos = static_cast<int32_t>(n_token_count);
    for (size_t i = 0; i < prompt_tokens.size(); ++i, ++pos) {
        common_batch_add(batch,
                        prompt_tokens[i],
                        pos,
                        {0},
                        false);
    }
    int n_ctx = llama_n_ctx(ctx);
    int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (n_ctx_used + batch.n_tokens > n_ctx) {
        fprintf(stderr, "context size exceeded\n");
        exit(0);
    }
    batch.logits[batch.n_tokens - 1] = true; // logits for the last token

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;
    std::cout << "Starts token inference" << "\n";
    timespec TTFT_end;
    timespec prefill_end;
    timespec decode_end;
    for (int i = 0; i < params.n_predict; ++i, ++pos) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            common_batch_clear(batch);
            common_batch_add(batch, new_token_id, pos, {0}, true);

            n_decode += 1;
            if(i==0){
              TTFT_end = get_monotonic_time();
              prefill_end = get_monotonic_time();
              i = params.n_predict;
              break;
            }
        }
    }
    decode_end = get_monotonic_time();
    long TTFT_duration_us = diff_microseconds(TTFT_start, TTFT_end);
    long prefill_duration_us = diff_microseconds(prefill_start, prefill_end);
    // long decode_duration_us = diff_microseconds(prefill_start, decode_end);

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    // Check free memory before unload
    size_t mem_decode = get_free_mem_bytes();
    printf("Free memory after mem_decode(inference): %.2f MiB\n", mem_decode / 1024.0 / 1024.0);

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    std::cout << "===================== Inference End ======================" << "\n";
    
    std::cout << "TTFT: " << TTFT_duration_us << "\n";
    std::cout << "prefill_duration_us: " << prefill_duration_us << "\n";
    std::cout << "KV_load_duration_us: " << KV_load_duration_us << "\n";
    std::cout << "weight_load_duration_us: " << weight_load_duration_us << "\n";

    append_log(log_path, log_file_name, context_size_of_states, layer_num, number_of_kv_layers_to_test,
           init_mem_bytes, unload_mem_bytes, load_mem_bytes,
           TTFT_duration_us, prefill_duration_us, kv_alloc_time + weight_alloc_time, kv_read_time + weight_read_time,
           KV_load_duration_us, weight_load_duration_us, 
           model_name);
    return 0;
}
