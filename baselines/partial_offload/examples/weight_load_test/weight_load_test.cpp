#include "arg.h"
#include "common.h"
#include "llama.h"
#include "json.hpp"
#include "llama-model-loader.h"
#include "mzCache_common.h"

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

// THIS CODE IS DEPRECATED (USE mz_save_layers_to_files)
// THIS CODE IS DEPRECATED (USE mz_save_layers_to_files)
// THIS CODE IS DEPRECATED (USE mz_save_layers_to_files)
// THIS CODE IS DEPRECATED (USE mz_save_layers_to_files)

// To test the inference fuctionality.
#define INFERENCE_TEST

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
    std::cout << "Test start" << "\n";
    std::string prompt = "How are you?";
    // 1. Configure parameters
    common_params params;
    params.sampling.seed  = 1234;
    // params.n_ctx         = 10000;  // Very large context size
    // params.n_predict     = 8192;   // Maximum number of tokens to generate
    // params.n_batch       = 10000;
    // params.n_ubatch      = 1024;

    params.sampling.seed  = 1234;
    params.n_ctx         = 128;  // Very large context size
    params.n_predict     = 32;   // Maximum number of tokens to generate
    params.n_batch       = 1;
    params.n_ubatch      = 1;

    params.warmup        = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    // 2. Initialize model and context
    // auto init_res = common_init_from_params(params);
    auto init_res = mzCache::mz_common_init_from_params(params);

    llama_model*   model = init_res.model.get();
    llama_context* ctx   = init_res.context.get();
    llama_model_loader* ml = init_res.model_loader;
    if (!model || !ctx) {
        std::fprintf(stderr, "ERROR: model/context initialization failed\n");
        return 1;
    }

    // 3. Check tensor count
    size_t n_tensors = llama_model_get_n_tensors(model);
    // printf("Total tensors in model: %zu\n", n_tensors);

    // 4. Check free memory before unload
    size_t mem_start = get_free_mem_bytes();
    printf("Free memory before unload: %.2f MiB\n", mem_start / 1024.0 / 1024.0);

    // 5. Unload half of the buffers
    int n_unload = n_tensors / 2;
    printf("Unloading %d buffers...\n", n_unload);
    std::vector<int> tensors_to_unload;
    if(!model->unload_buffers(tensors_to_unload)){
        std::cout << "model->unload_buffers returned error" << "\n";
        return 0;
    }

    // This function is deprecated
    // llama_model_unload_buffers(model, n_unload);

    sleep(3);

    // 6. Check free memory after unload
    size_t mem_unload = get_free_mem_bytes();
    printf("Free memory after unload: %.2f MiB\n", mem_unload / 1024.0 / 1024.0);



    // 7. Report result
    double mem_diff = (double)(mem_unload - mem_start) / (1024.0 * 1024.0);
    printf("System available memory increased by: %.2f MiB\n", mem_diff);

    
    
    // 8. Load model weight
    // Todo
    // if(!mzCache::mz_llama_model_load_buffers(model, ml)){
    //   std::cout << "ERROR: Failed to load buffers from model loader." << "\n";
    // }
    sleep(3);
    // check tensor count
    n_tensors = llama_model_get_n_tensors(model);
    printf("Total tensors in model: %zu\n", n_tensors);

    // 4. Check free memory before unload
    size_t mem_load = get_free_mem_bytes();
    printf("Free memory after reload: %.2f MiB\n", mem_load / 1024.0 / 1024.0);
    // load

    // 9. Check free space
    mem_diff = (double)(mem_unload - mem_load) / (1024.0 * 1024.0);
    printf("System available memory decreased by: %.2f MiB\n", mem_diff);
    

#ifdef INFERENCE_TEST
    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize the prompt


    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token
    std::cout << "print the prompt token-by-token" << "\n";
    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }
    std::cout << "\n";

    // prepare a batch for the prompt
    
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;
    std::cout << "Starts token inference" << "\n";
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + params.n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

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
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

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
    printf("Free memory after mem_decode: %.2f MiB\n", mem_decode / 1024.0 / 1024.0);
    // load

    // Check free space
    mem_diff = (double)(mem_load - mem_decode) / (1024.0 * 1024.0);
    printf("System available memory decreased by: %.2f MiB\n", mem_diff);

#endif

    return 0;
}
