#include "mzcache_kernels.h"
#include "mzcache_kernels.h"
#include "mzCache_kvchunk.h"

#include "mzCache_threadpool.h"

#include "ggml-opencl-extra.h"
#include <cstdint>
#include <thread>
#include <vector>

#include <fcntl.h>      // open, O_RDONLY, O_DIRECT
#include <unistd.h>     // pread, close
#include <sys/types.h>  // off_t

#include <utility>  // std::pair

extern std::vector<chunk_ptrs *> g_svm_chunk_ptrs;

float alloc_time = 0;
#ifdef MZCACHE_SVM_KV_CHUNK
mzcache_kv_cache::mzcache_kv_cache(int _num_layers, int _kv_hidden_dim, int _num_chunks_per_tensor, std::string _model_name)
    : num_layers(_num_layers), kv_hidden_dim(_kv_hidden_dim), num_chunks_per_tensor(_num_chunks_per_tensor), model_name(std::move(_model_name)) 
{
    // This bounds are determined by profiler.

    cur_layer_idx = num_layers - 1;
    cur_chunk_idx_to_compress = chunk_bound -1; // current chunk index to compress
    cur_chunk_idx_to_store = num_chunks_per_tensor - 1;   // current chunk index to store

    // initialize arenas
    arenas.push_back(new Arena());

    // initializekkv_store_ofs and kv_store_offsets
    // if (kv_store_ofs.empty()) {
    //     kv_store_ofs.resize(num_layers);

    // }

    layer_fds.assign(num_layers, -1);
    kv_store_offsets.assign(num_layers, 0);


    std::vector<int> phys_big = {6,7}; // 6,7 are big cores
    std::vector<int> phys_middle = {0,1,2,3,4,5};

    num_middle_cores = phys_middle.size();
    num_big_cores = phys_big.size();

    // initialize thread pool
    thread_pool = std::make_unique<ThreadPool>(core_configs);

    alloc_done.reserve(num_layers * chunk_bound); // Reserve space for shared futures map

    for (int i = 0; i < num_layers; ++i) {

        chunk_ptrs * k_chunk_ptrs = g_svm_chunk_ptrs[2*i+0];
        chunk_ptrs * v_chunk_ptrs = g_svm_chunk_ptrs[2*i+1];

        // ggml_tensor * k = k_tensors.c[0];
        // ggml_tensor * v = v_tensors.c[0];

        // ggml_tensor_extra_cl * k_extra = (ggml_tensor_extra_cl *)k->extra;
        // ggml_tensor_extra_cl * v_extra = (ggml_tensor_extra_cl *)v->extra;

        // chunk_ptrs * k_chunk_ptrs = (chunk_ptrs *)k_extra->svm_chunk_ptrs;
        // chunk_ptrs * v_chunk_ptrs = (chunk_ptrs *)v_extra->svm_chunk_ptrs;

        layers[i].layer_idx = i;

        for(int j = 0; j < num_chunks_per_tensor; ++j) {

            // This code is more independent to backend implementation. (Can be used with SVM or CUDA Shared memory)
            if (k_chunk_ptrs->c[j] == nullptr || v_chunk_ptrs->c[j] == nullptr) {
                throw std::runtime_error("Chunk tensor is null in the kv_cache.");
            }
            layers[i].k_chunks.raw_chunks[j].c = k_chunk_ptrs->c[j];
            layers[i].v_chunks.raw_chunks[j].c = v_chunk_ptrs->c[j];
        

            layers[i].states[j] = KVstate::Raw; // Initialize all chunks to Raw state

            // flexgen allocate min, max arrays for compression
            layers[i].k_chunks.compressed_chunks[j].mins = new __fp16[TOKENS_PER_CHUNK * kv_hidden_dim / 64];
            layers[i].k_chunks.compressed_chunks[j].maxs = new __fp16[TOKENS_PER_CHUNK * kv_hidden_dim / 64];

            layers[i].v_chunks.compressed_chunks[j].mins = new __fp16[TOKENS_PER_CHUNK * kv_hidden_dim / 64];
            layers[i].v_chunks.compressed_chunks[j].maxs = new __fp16[TOKENS_PER_CHUNK * kv_hidden_dim / 64];

            if(i == num_layers -1 && j == num_chunks_per_tensor - 1) {

                printf("Last layer %d, last chunk %d initialized.\n", i, j);

                // print first 10 elements of the last chunk
                __fp16* last_chunk_data = (__fp16*)layers[i].k_chunks.raw_chunks[j].c;
                printf("Last chunk data first 10 elements: ");
                for (int t = 0; t < 10; ++t) {
                    printf("%f ", (float)(last_chunk_data[t]));
                }
                printf("\n");

            }
            

        }
    }
}

mzcache_kv_cache::~mzcache_kv_cache() {
    // Clean up arenas
    for (auto arena : arenas) {
        delete arena;
    }
    arenas.clear();

    // Clean up min/max arrays
    for (int i = 0; i < num_layers; ++i) {
        for (int j = 0; j < num_chunks_per_tensor; ++j)
        {
            delete[] layers[i].k_chunks.compressed_chunks[j].mins;
            delete[] layers[i].k_chunks.compressed_chunks[j].maxs;

            delete[] layers[i].v_chunks.compressed_chunks[j].mins;
            delete[] layers[i].v_chunks.compressed_chunks[j].maxs;

        }
    }
    
    // free svm pointers
    for (auto chunk_ptrs : g_svm_chunk_ptrs) {
        for (int i = 0; i < MAX_CHUNKS_PER_TENSOR * 2; i += 2) {
            if (chunk_ptrs->c[i] != nullptr) {
#ifdef MZCACHE_USE_OPENCL
                clSVMFree(g_opencl_context, chunk_ptrs->c[i]);
#endif
                chunk_ptrs->c[i] = nullptr; // Set to nullptr after freeing
            }
        }
    }

    std::cout << "alloc_time = " << alloc_time << " ms" << std::endl;
}

// Compress should be done sequentially, so that the compressed chunks are stored in order.
void mzcache_kv_cache::mzcache_compress_chunks(int num_chunks) {
    int num_threads = core_configs[CoreType::DECOMP].size();

    for (int iter = 0; iter < num_chunks; ++iter) {
        __fp16* key_data_ptr = (__fp16 *)layers[cur_layer_idx].k_chunks.raw_chunks[cur_chunk_idx_to_compress].c;
        __fp16* value_data_ptr = (__fp16 *)layers[cur_layer_idx].v_chunks.raw_chunks[cur_chunk_idx_to_compress].c;

        // print first 10 elements of key_data_ptr
        // printf("Compressing layer %d, chunk %d\n", cur_layer_idx, cur_chunk_idx_to_compress);
        // printf("Key data first 10 elements: ");
        // for (int i = 0; i < 10; ++i) {
        //     printf("%f ", (float)(key_data_ptr[i]));
        // }
        // printf("\n");


        size_t comp_size = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(uint8_t) / 2; // 4-bit packed size

        size_t total = sizeof(ChunkHeader) + comp_size;
        Arena* cur = arenas.back();

        // if the current arena is full, add a new one
        if (cur->top + total * 2 > STACK_SIZE) {
            cur = new Arena();
            arenas.push_back(cur);
        }

        uint8_t * key_out_ptr = cur->pool + cur->top + sizeof(ChunkHeader);

        compress_fp16_to_4bit(
            key_data_ptr, 
            kv_hidden_dim, 
            key_out_ptr, 
            layers[cur_layer_idx].k_chunks.compressed_chunks[cur_chunk_idx_to_compress].mins, 
            layers[cur_layer_idx].k_chunks.compressed_chunks[cur_chunk_idx_to_compress].maxs,
            
            *thread_pool,
            num_threads
        );

        // printf("key compress done.\n");

        uint8_t * value_out_ptr = cur->pool + cur->top + sizeof(ChunkHeader) + total;

        compress_fp16_to_4bit(
            value_data_ptr, 
            kv_hidden_dim, 
            value_out_ptr, 
            layers[cur_layer_idx].v_chunks.compressed_chunks[cur_chunk_idx_to_compress].mins, 
            layers[cur_layer_idx].v_chunks.compressed_chunks[cur_chunk_idx_to_compress].maxs,
            
            *thread_pool,
            num_threads
        );

        layers[cur_layer_idx].k_chunks.compressed_chunks[cur_chunk_idx_to_compress].stack_id = arenas.size() - 1;
        layers[cur_layer_idx].v_chunks.compressed_chunks[cur_chunk_idx_to_compress].stack_id = arenas.size() - 1;

        layers[cur_layer_idx].k_chunks.compressed_chunks[cur_chunk_idx_to_compress].offset = cur->top + sizeof(ChunkHeader);
        layers[cur_layer_idx].v_chunks.compressed_chunks[cur_chunk_idx_to_compress].offset = cur->top + sizeof(ChunkHeader) + total;

        layers[cur_layer_idx].k_chunks.compressed_chunks[cur_chunk_idx_to_compress].comp_size = comp_size;
        layers[cur_layer_idx].v_chunks.compressed_chunks[cur_chunk_idx_to_compress].comp_size = comp_size;

        layers[cur_layer_idx].states[cur_chunk_idx_to_compress] = KVstate::Compressed; // Update state to Compressed

        // Should free raw chunks after compression
        // To-Do..
#ifdef MZCACHE_USE_OPENCL
        // Free the SVM pointers if using OpenCL
        // std::cout << "Freeing SVM pointers for key and value data." << std::endl;
        clSVMFree(g_opencl_context, key_data_ptr);
        clSVMFree(g_opencl_context, value_data_ptr);

        g_svm_chunk_ptrs[2 * cur_layer_idx + 0]->c[cur_chunk_idx_to_compress] = nullptr;
        g_svm_chunk_ptrs[2 * cur_layer_idx + 1]->c[cur_chunk_idx_to_compress] = nullptr;
#endif

        cur->top += total * 2; // Update the top pointer for both key and value chunks
        cur_chunk_idx_to_compress--; // Move to the next chunk to compress
        if (cur_chunk_idx_to_compress < 0) {
            if (cur_layer_idx == 0) {
                printf("All chunks compressed.\n");
                return; // All chunks compressed
            }
            else {
                cur_layer_idx--; // Move to the previous layer
                cur_chunk_idx_to_compress = chunk_bound - 1; // Reset to the last chunk index
            }
        }

    }
}

void mzcache_kv_cache::mzcache_decompress_chunks(int num_chunks) {
    // const int num_threads     = core_configs[CoreType::DECOMP_KERNEL].size();
    const size_t header_sz    = sizeof(ChunkHeader);
    const size_t single_total = header_sz
        + layers[0].k_chunks.compressed_chunks[0].comp_size;

    // reserve
    // std::vector<std::future<void>> alloc_futs;
    // std::vector<std::future<void>> decomp_futs;
    // alloc_futs.reserve(num_chunks);
    // decomp_futs.reserve(num_chunks);

    // (1) enqueue alloc & restore tasks
    MZ_TIME_START(enqueue_start)
    for (int iter = 0; iter < num_chunks; ++iter) {
        // ── select next chunk index ──
        cur_chunk_idx_to_compress++;
        if (cur_chunk_idx_to_compress >= chunk_bound) {
            cur_chunk_idx_to_compress = 0;
            cur_layer_idx++;
            if (cur_layer_idx >= num_layers) {
                fprintf(stderr, "Error: Cannot decompress more chunks than available.\n");
                return;
            }
        }
        int layer_idx = cur_layer_idx;
        int chunk_idx = cur_chunk_idx_to_compress;

        // ── copy metadata ──
        auto Kcc       = layers[layer_idx].k_chunks.compressed_chunks[chunk_idx];
        auto Vcc       = layers[layer_idx].v_chunks.compressed_chunks[chunk_idx];
        Arena* curArena = arenas[Kcc.stack_id];

        auto alloc_prom = std::make_shared<std::promise<void>>();
        ShFut sfut      = alloc_prom->get_future().share();
        alloc_done.emplace(std::make_pair(layer_idx, chunk_idx), sfut);

        // ── (2) alloc task (BIG) ──
        thread_pool->enqueue(
            CoreType::ALLOC,
            [=, prom = alloc_prom]() mutable {
                __fp16* new_k = (__fp16*)clSVMAlloc(
                    g_opencl_context,
                    CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                    TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16) * 2,
                    0
                );
                if (!new_k) throw std::runtime_error("SVM alloc failed");
                __fp16* new_v = new_k + TOKENS_PER_CHUNK * kv_hidden_dim;

                // register pointers
                layers[layer_idx].k_chunks.raw_chunks[chunk_idx].c = new_k;
                g_svm_chunk_ptrs[2*layer_idx + 0]->c[chunk_idx]   = new_k;
                layers[layer_idx].v_chunks.raw_chunks[chunk_idx].c = new_v;
                g_svm_chunk_ptrs[2*layer_idx + 1]->c[chunk_idx]   = new_v;

                prom->set_value();
            }
        );

        // // Single-threaded version
        thread_pool->enqueue(CoreType::DECOMP, [&, layer_idx, chunk_idx]() mutable {

            // 1) wait for alloc completion
            alloc_done.at({layer_idx, chunk_idx}).wait();

            // 2) metadata/pointers
            auto &Kcc  = layers[layer_idx].k_chunks.compressed_chunks[chunk_idx];
            auto &Vcc  = layers[layer_idx].v_chunks.compressed_chunks[chunk_idx];
            Arena *cur = arenas[Kcc.stack_id];

            __fp16* new_k = (__fp16*)g_svm_chunk_ptrs[2*layer_idx + 0]->c[chunk_idx];
            __fp16* new_v = (__fp16*)g_svm_chunk_ptrs[2*layer_idx + 1]->c[chunk_idx];

            uint8_t* comp_k = cur->pool + Kcc.offset;
            uint8_t* comp_v = cur->pool + Vcc.offset;

            // 3) restore --> **single-threaded function call**
            decompress_4bit_to_fp16_single_thread(
                comp_k, kv_hidden_dim, new_k,
                Kcc.mins, Kcc.maxs);

            decompress_4bit_to_fp16_single_thread(
                comp_v, kv_hidden_dim, new_v,
                Vcc.mins, Vcc.maxs);

            // 4) state update
            layers[layer_idx].states[chunk_idx] = KVstate::Raw;
        });
    }
    MZ_TIME_END(enqueue_start);

    // (4) wait for all restore tasks to complete
    // for (auto &f : decomp_futs) {
    //     f.get();
    // }
    thread_pool->wait_idle(CoreType::DECOMP);

    // (5) reset top in one shot from the last restored chunk's info
    const auto &lastKcc = 
        layers[cur_layer_idx].k_chunks.compressed_chunks[cur_chunk_idx_to_compress];
    Arena* lastArena = arenas[lastKcc.stack_id];

    // at compress time,
    //   offset = old_top + header_sz
    //   new_top  = old_top + 2*(header_sz + comp_size)
    //
    // therefore old_top = offset - header_sz
    // new_top = old_top + 2*(header_sz+comp_size)
    //         = (offset - header_sz) + 2*header_sz + 2*comp_size
    //         = offset + header_sz + 2*comp_size
    lastArena->top = lastKcc.offset
                  + header_sz
                  + 2 * lastKcc.comp_size;

    for (int i = (int)arenas.size() - 1; i > lastKcc.stack_id; --i) {
        delete arenas[i];
        arenas.pop_back();
    }
}

void mzcache_kv_cache::mzcache_decompress_chunks_sequential(int num_chunks) {
    // DECOMP number of threads
    // const int num_threads     = core_configs[CoreType::DECOMP_KERNEL].size();
    const size_t header_sz    = sizeof(ChunkHeader);
    const size_t single_total = header_sz
        + layers[0].k_chunks.compressed_chunks[0].comp_size;

    // local copies
    int local_layer = cur_layer_idx;
    int local_chunk = cur_chunk_idx_to_compress;

    // // 1) enqueue allocations only
    // std::vector<std::future<void>> alloc_futs;
    // alloc_futs.reserve(num_chunks);

    MZ_TIME_START(allocating_chunks);

    for (int iter = 0; iter < num_chunks; ++iter) {
        // advance local indices
        local_chunk++;
        if (local_chunk >= chunk_bound) {
            local_chunk = 0;
            local_layer++;
            if (local_layer >= num_layers) {
                fprintf(stderr, "Error: too many chunks in sequential.\n");
                return;
            }
        }

        int layer_idx = local_layer;
        int chunk_idx = local_chunk;

        // copy metadata
        auto Kcc = layers[layer_idx].k_chunks.compressed_chunks[chunk_idx];
        auto Vcc = layers[layer_idx].v_chunks.compressed_chunks[chunk_idx];
        Arena* curArena = arenas[Kcc.stack_id];

        // task that only performs SVM allocation
        thread_pool->enqueue(
            CoreType::ALLOC,
            [=]() {
                __fp16* new_k = (__fp16*)clSVMAlloc(
                    g_opencl_context,
                    CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                    TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16) * 2,
                    0
                );
                if (!new_k) throw std::runtime_error("SVM alloc failed");
                __fp16* new_v = new_k + TOKENS_PER_CHUNK * kv_hidden_dim;

                // register pointers
                layers[layer_idx].k_chunks.raw_chunks[chunk_idx].c = new_k;
                g_svm_chunk_ptrs[2*layer_idx + 0]->c[chunk_idx]   = new_k;
                layers[layer_idx].v_chunks.raw_chunks[chunk_idx].c = new_v;
                g_svm_chunk_ptrs[2*layer_idx + 1]->c[chunk_idx]   = new_v;
            }
        );
    }

    // 2) wait for all allocations to complete
    // for (auto &f : alloc_futs) {
    //     f.get();
    // }
    thread_pool->wait_idle(CoreType::ALLOC);

    MZ_TIME_END(allocating_chunks);
    MZ_TIME_START(decompressing_chunks);

    // 3) restore while updating the real cur_* indices
    for (int iter = 0; iter < num_chunks; ++iter) {
        // advance real indices
        cur_chunk_idx_to_compress++;
        if (cur_chunk_idx_to_compress >= chunk_bound) {
            cur_chunk_idx_to_compress = 0;
            cur_layer_idx++;
            if (cur_layer_idx >= num_layers) {
                // should not happen
                fprintf(stderr, "Error: Cannot decompress more chunks than available.\n");
                break;
            }
        }

        int layer_idx = cur_layer_idx;
        int chunk_idx = cur_chunk_idx_to_compress;

        // re-fetch metadata
        auto &Kcc     = layers[layer_idx].k_chunks.compressed_chunks[chunk_idx];
        auto &Vcc     = layers[layer_idx].v_chunks.compressed_chunks[chunk_idx];
        Arena *curArena = arenas[Kcc.stack_id];

        // fetch pointers
        __fp16* new_k = (__fp16*)g_svm_chunk_ptrs[2*layer_idx + 0]->c[chunk_idx];
        __fp16* new_v = (__fp16*)g_svm_chunk_ptrs[2*layer_idx + 1]->c[chunk_idx];

        // restore Key
        uint8_t* comp_k = curArena->pool + Kcc.offset;

        decompress_4bit_to_fp16_single_thread(
            comp_k, kv_hidden_dim, new_k,
            Kcc.mins, Kcc.maxs
        );

        // restore Value
        uint8_t* comp_v = curArena->pool + Vcc.offset;

        decompress_4bit_to_fp16_single_thread(
            comp_v, kv_hidden_dim, new_v,
            Vcc.mins, Vcc.maxs
        );

        // update state
        layers[layer_idx].states[chunk_idx] = KVstate::Raw;
    }
    MZ_TIME_END(decompressing_chunks);

    // 4) delete all arenas except the last apex arena
    const auto &lastKcc = layers[cur_layer_idx]
                             .k_chunks
                             .compressed_chunks[cur_chunk_idx_to_compress];
    int last_id = lastKcc.stack_id;

    // reset top in one shot
    arenas[last_id]->top = lastKcc.offset
                         + header_sz
                         + 2 * lastKcc.comp_size;

    // free trailing arenas
    for (int i = (int)arenas.size() - 1; i > last_id; --i) {
        delete arenas[i];
        arenas.pop_back();
    }
}

std::pair<double,double> mzcache_kv_cache::mzcache_decompress_chunks_profile(int num_chunks) {
    // DECOMP number of threads
    // const int num_threads     = core_configs[CoreType::DECOMP_KERNEL].size();
    const size_t header_sz    = sizeof(ChunkHeader);
    const size_t single_total = header_sz
        + layers[0].k_chunks.compressed_chunks[0].comp_size;

    // local copies
    int local_layer = cur_layer_idx;
    int local_chunk = cur_chunk_idx_to_compress;

    auto alloc_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_chunks; ++iter) {
        // advance local indices
        local_chunk++;
        if (local_chunk >= chunk_bound) {
            local_chunk = 0;
            local_layer++;
            if (local_layer >= num_layers) {
                fprintf(stderr, "Error: too many chunks in sequential.\n");
                return {0.0, 0.0};
            }
        }

        int layer_idx = local_layer;
        int chunk_idx = local_chunk;

        // copy metadata
        auto Kcc = layers[layer_idx].k_chunks.compressed_chunks[chunk_idx];
        auto Vcc = layers[layer_idx].v_chunks.compressed_chunks[chunk_idx];
        Arena* curArena = arenas[Kcc.stack_id];

        // task that only performs SVM allocation
        thread_pool->enqueue(
            CoreType::ALLOC,
            [=]() {
                __fp16* new_k = (__fp16*)clSVMAlloc(
                    g_opencl_context,
                    CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                    TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16) * 2,
                    0
                );
                if (!new_k) throw std::runtime_error("SVM alloc failed");
                __fp16* new_v = new_k + TOKENS_PER_CHUNK * kv_hidden_dim;

                // register pointers
                layers[layer_idx].k_chunks.raw_chunks[chunk_idx].c = new_k;
                g_svm_chunk_ptrs[2*layer_idx + 0]->c[chunk_idx]   = new_k;
                layers[layer_idx].v_chunks.raw_chunks[chunk_idx].c = new_v;
                g_svm_chunk_ptrs[2*layer_idx + 1]->c[chunk_idx]   = new_v;
            }
        );
    }

    thread_pool->wait_idle(CoreType::ALLOC);

    auto alloc_end = std::chrono::high_resolution_clock::now();
    double alloc_ms =
        std::chrono::duration<double, std::milli>(alloc_end - alloc_start).count();

    auto decomp_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_chunks; ++iter) {

        cur_chunk_idx_to_compress++;
        if (cur_chunk_idx_to_compress >= chunk_bound) {
            cur_chunk_idx_to_compress = 0;
            cur_layer_idx++;
            if (cur_layer_idx >= num_layers) {
                // should not happen
                fprintf(stderr, "Error: Cannot decompress more chunks than available.\n");
                break;
            }
        }

        int layer_idx = cur_layer_idx;
        int chunk_idx = cur_chunk_idx_to_compress;

        thread_pool->enqueue(CoreType::DECOMP, [&, layer_idx, chunk_idx]() mutable {


            // 2) metadata/pointers
            auto &Kcc  = layers[layer_idx].k_chunks.compressed_chunks[chunk_idx];
            auto &Vcc  = layers[layer_idx].v_chunks.compressed_chunks[chunk_idx];
            Arena *cur = arenas[Kcc.stack_id];

            __fp16* new_k = (__fp16*)g_svm_chunk_ptrs[2*layer_idx + 0]->c[chunk_idx];
            __fp16* new_v = (__fp16*)g_svm_chunk_ptrs[2*layer_idx + 1]->c[chunk_idx];

            uint8_t* comp_k = cur->pool + Kcc.offset;
            uint8_t* comp_v = cur->pool + Vcc.offset;

            // 3) restore --> **single-threaded function call**
            decompress_4bit_to_fp16_single_thread(
                comp_k, kv_hidden_dim, new_k,
                Kcc.mins, Kcc.maxs);

            decompress_4bit_to_fp16_single_thread(
                comp_v, kv_hidden_dim, new_v,
                Vcc.mins, Vcc.maxs);

            // 4) state update
            layers[layer_idx].states[chunk_idx] = KVstate::Raw;
        });
    }

    thread_pool->wait_idle(CoreType::DECOMP);

    auto decomp_end = std::chrono::high_resolution_clock::now();
    double decomp_ms =
        std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();


    // 4) delete all arenas except the last apex arena
    const auto &lastKcc = layers[cur_layer_idx]
                             .k_chunks
                             .compressed_chunks[cur_chunk_idx_to_compress];
    int last_id = lastKcc.stack_id;

    // reset top in one shot
    arenas[last_id]->top = lastKcc.offset
                         + header_sz
                         + 2 * lastKcc.comp_size;

    // free trailing arenas
    for (int i = (int)arenas.size() - 1; i > last_id; --i) {
        delete arenas[i];
        arenas.pop_back();
    }


    std::cout << "[MZCACHE][Timing] allocating_chunks took "
              << alloc_ms << " ms\n";
    std::cout << "[MZCACHE][Timing] decompressing_chunks took "
              << decomp_ms << " ms\n";

    return {alloc_ms, decomp_ms};
}



void mzcache_kv_cache::mzcache_store_chunks(int num_chunks) {

    const size_t raw_chunk_bytes = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16); // e.g. 256K * 2 = 512KB

    for (int iter = 0; iter < num_chunks; ++iter) {
        int L = cur_layer_idx;
        int C = cur_chunk_idx_to_store;


        if (C == num_chunks_per_tensor - 1) {
            std::string fn = "KV_" + model_name + "_L" + std::to_string(L) + ".bin";
            int fd = ::open(fn.c_str(),
                            O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT,
                            0666);
            if (fd < 0) {
                perror(("open " + fn).c_str());
                return;
            }
            layer_fds[L] = fd;
        }
        int fd = layer_fds[L];

        off_t chunk_idx0    = static_cast<off_t>(C - chunk_bound);
        off_t offset_key    = chunk_idx0 * (raw_chunk_bytes * 2);
        off_t offset_value  = offset_key + raw_chunk_bytes;

    

        // (3) fetch raw pointers
        __fp16* key_ptr   = (__fp16*)layers[L]
                            .k_chunks.raw_chunks[C].c;
        __fp16* value_ptr = (__fp16*)layers[L]
                            .v_chunks.raw_chunks[C].c;

        {
            ssize_t n = pwrite(fd, key_ptr, raw_chunk_bytes, offset_key);
            if (n != (ssize_t)raw_chunk_bytes) {
                perror("pwrite key");
            }
            // record metadata
            auto& kc = layers[L].k_chunks.stored_chunks[C];
            kc.file_offset = offset_key;
            kc.stored_size = raw_chunk_bytes;
        }
        {
            ssize_t n = pwrite(fd, value_ptr, raw_chunk_bytes, offset_value);
            if (n != (ssize_t)raw_chunk_bytes) {
                perror("pwrite value");
            }
            // record metadata
            auto& vc = layers[L].v_chunks.stored_chunks[C];
            vc.file_offset = offset_value;
            vc.stored_size = raw_chunk_bytes;
        }

        // update state
        layers[L].states[C] = KVstate::Stored;

        // (5) free OpenCL SVM
#ifdef MZCACHE_USE_OPENCL
        clSVMFree(g_opencl_context, key_ptr);
        clSVMFree(g_opencl_context, value_ptr);

        g_svm_chunk_ptrs[2 * L + 0]->c[C] = nullptr;
        g_svm_chunk_ptrs[2 * L + 1]->c[C] = nullptr;
#endif

        cur_chunk_idx_to_store--;
        if (cur_chunk_idx_to_store < chunk_bound) {
            ::close(layer_fds[L]);
            layer_fds[L] = -1;

            if (L == 0) {
                for (int i = 0; i < num_layers; ++i) {
                    int fd = layer_fds[i];
                    if (fd >= 0) {
                        fprintf(stderr, "layer %d: file descriptor %d still open\n", i, fd);
                        ::close(fd);
                        layer_fds[i] = -1;
                    }
                }
                printf("All chunks stored.\n");
                return;
            }
            else {

                cur_layer_idx--;
                cur_chunk_idx_to_store = num_chunks_per_tensor - 1; // Reset to the last chunk index of the previous layer

            }
        }
    }
    
    ::close(layer_fds[cur_layer_idx]);
    layer_fds[cur_layer_idx] = -1;

    // (7) safely close any remaining file streams
    for (int i = 0; i < num_layers; ++i) {
        int fd = layer_fds[i];
        if (fd >= 0) {
            fprintf(stderr, "layer %d: file descriptor %d still open\n", i, fd);
            ::close(fd);
            layer_fds[i] = -1;
        }
    }
}

double mzcache_kv_cache::mzcache_load_chunks_sequential(int num_chunks)
{
    const size_t raw_bytes   = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16);
    const size_t alloc_bytes = raw_bytes * 2;                 // key + value

    /* 0) O_DIRECT-open only layers from the current one onward ────────────────────── */
    for (int l = cur_layer_idx; l < num_layers; ++l) {
        if (layer_fds[l] >= 0) continue;
        std::string fn = "KV_" + model_name + "_L" + std::to_string(l) + ".bin";
        int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) { perror(("open "+fn).c_str()); throw std::runtime_error("open"); }
        layer_fds[l] = fd;
    }

    /* 1) batch-enqueue ALLOC tasks first ───────────────────── */
    MZ_TIME_START(allocating_chunks);

    int tmp_layer = cur_layer_idx;
    int tmp_chunk = cur_chunk_idx_to_store;

    for (int i = 0; i < num_chunks; ++i) {

        // next (L,C)
        tmp_chunk++;
        if (tmp_chunk == num_chunks_per_tensor) {
            tmp_chunk  = chunk_bound;
            tmp_layer++;
            if (tmp_layer >= num_layers) {
                fprintf(stderr, "Error: too many chunks.\n"); return 0.0;
            }
        }
        int L = tmp_layer, C = tmp_chunk;

        /* register alloc promise */
        auto prom = std::make_shared<std::promise<void>>();
        alloc_done[{L,C}] = prom->get_future().share();

        thread_pool->enqueue(CoreType::ALLOC,
            [=, prom = std::move(prom)]() mutable {

                __fp16* new_k = (__fp16*)clSVMAlloc(
                    g_opencl_context,
                    CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                    alloc_bytes, 0);
                if (!new_k) throw std::runtime_error("SVM alloc failed");
                __fp16* new_v = new_k + TOKENS_PER_CHUNK * kv_hidden_dim;

                layers[L].k_chunks.raw_chunks[C].c = new_k;
                g_svm_chunk_ptrs[2*L + 0]->c[C]   = new_k;
                layers[L].v_chunks.raw_chunks[C].c = new_v;
                g_svm_chunk_ptrs[2*L + 1]->c[C]   = new_v;

                prom->set_value();
            });
    }

    /* wait for alloc group (guarantees memory is reserved) */
    thread_pool->wait_idle(CoreType::ALLOC);
    MZ_TIME_END(allocating_chunks);

    int prev_cur_layer_idx = cur_layer_idx;
    int prev_cur_chunk_idx_to_store = cur_chunk_idx_to_store;

    /* 2) enqueue READ tasks sequentially  ──────────────────────────── */
    auto t_store_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_chunks; ++i) {

        // advance index (same rule)
        cur_chunk_idx_to_store++;
        if (cur_chunk_idx_to_store == num_chunks_per_tensor) {
            cur_chunk_idx_to_store  = chunk_bound;
            cur_layer_idx++;
        }
        int L = cur_layer_idx, C = cur_chunk_idx_to_store;

        thread_pool->enqueue(CoreType::READ,
            [=, this] {

                /* wait for alloc completion */
                alloc_done[{L,C}].wait();

                const auto& kc = layers[L].k_chunks.stored_chunks[C];
                const auto& vc = layers[L].v_chunks.stored_chunks[C];
                off_t off_k = kc.file_offset;
                off_t off_v = vc.file_offset;

                __fp16* new_k = (__fp16*)g_svm_chunk_ptrs[2*L + 0]->c[C];
                __fp16* new_v = (__fp16*)g_svm_chunk_ptrs[2*L + 1]->c[C];

                // (check contiguity → one read or two)
                if ((uint8_t*)new_v == (uint8_t*)new_k + raw_bytes &&
                    off_v == off_k + raw_bytes)
                {
                    ssize_t n = pread(layer_fds[L], new_k, raw_bytes*2, off_k);
                    if (n != (ssize_t)raw_bytes*2) perror("pread contiguous");
                } else {
                    ssize_t n = pread(layer_fds[L], new_k, raw_bytes, off_k);
                    if (n != (ssize_t)raw_bytes) perror("pread k");
                    n = pread(layer_fds[L], new_v, raw_bytes, off_v);
                    if (n != (ssize_t)raw_bytes) perror("pread v");
                }

                layers[L].states[C] = KVstate::Raw;
            });
    }

    thread_pool->wait_idle(CoreType::READ);       // all loads complete
    auto t_store_end = std::chrono::high_resolution_clock::now();
    double store_ms =
        std::chrono::duration<double, std::milli>(t_store_end - t_store_start).count();

    /* 3) FD close (only the opened range) */
    for (int l = prev_cur_layer_idx; l < num_layers; ++l) {
        if (layer_fds[l] >= 0) { ::close(layer_fds[l]); layer_fds[l] = -1; }
    }

    return store_ms; // return the time taken for storing chunks
}


void mzcache_kv_cache::mzcache_load_chunks(int num_chunks)
{
    const size_t raw_bytes   = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16);
    const size_t alloc_bytes = raw_bytes * 2;   // K + V

    /* ── 0) open only the layer FDs needed from now on ───────────────────── */
    for (int l = cur_layer_idx; l < num_layers; ++l) {
        if (layer_fds[l] >= 0) continue;               // skip if already open
        std::string fn = "KV_" + model_name + "_L" + std::to_string(l) + ".bin";
        int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) { perror(("open "+fn).c_str()); throw std::runtime_error("open"); }
        layer_fds[l] = fd;
    }

    int prev_cur_layer_idx = cur_layer_idx;

    /* ── 1) enqueue: ALLOC + READ  ─────────────────────────────── */
    for (int iter = 0; iter < num_chunks; ++iter) {

        cur_chunk_idx_to_store++;
        if (cur_chunk_idx_to_store == num_chunks_per_tensor) {
            cur_chunk_idx_to_store  = chunk_bound;
            cur_layer_idx++;
        }
        int L = cur_layer_idx, C = cur_chunk_idx_to_store;

        /* ── (A) ALLOC task ─────────────────────────────────── */
        auto alloc_prom = std::make_shared<std::promise<void>>();
        alloc_done[{L, C}] = alloc_prom->get_future().share();

        thread_pool->enqueue(CoreType::ALLOC,
            [=, prom = std::move(alloc_prom)]() mutable {

                __fp16* new_k = (__fp16*)clSVMAlloc(
                    g_opencl_context,
                    CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                    alloc_bytes, 0);
                if (!new_k) throw std::runtime_error("SVM alloc failed");
                __fp16* new_v = new_k + TOKENS_PER_CHUNK * kv_hidden_dim;

                layers[L].k_chunks.raw_chunks[C].c = new_k;
                g_svm_chunk_ptrs[2*L + 0]->c[C]   = new_k;
                layers[L].v_chunks.raw_chunks[C].c = new_v;
                g_svm_chunk_ptrs[2*L + 1]->c[C]   = new_v;

                prom->set_value();                    // alloc done
            });

        thread_pool->wait_idle(CoreType::ALLOC); // wait for alloc to finish

        /* ── (B) READ task ──────────────────────────────────── */
        MZ_TIME_START(pure_read);
        thread_pool->enqueue(CoreType::READ,
            [=, this] {

                alloc_done[{L, C}].wait();            // alloc complete

                const auto& kc = layers[L].k_chunks.stored_chunks[C];
                const auto& vc = layers[L].v_chunks.stored_chunks[C];
                off_t off_k = kc.file_offset;
                off_t off_v = vc.file_offset;

                __fp16* new_k = (__fp16*)g_svm_chunk_ptrs[2*L + 0]->c[C];
                __fp16* new_v = (__fp16*)g_svm_chunk_ptrs[2*L + 1]->c[C];

                if ((uint8_t*)new_v == (uint8_t*)new_k + raw_bytes &&
                    off_v == off_k + raw_bytes)
                {
                    /* contiguous region: single read */
                    ssize_t n = pread(layer_fds[L], new_k, alloc_bytes, off_k);
                    if (n != (ssize_t)alloc_bytes) perror("pread 1call");
                } else {
                    ssize_t n = pread(layer_fds[L], new_k, raw_bytes, off_k);
                    if (n != (ssize_t)raw_bytes) perror("pread key");
                    n = pread(layer_fds[L], new_v, raw_bytes, off_v);
                    if (n != (ssize_t)raw_bytes) perror("pread val");
                }

                layers[L].states[C] = KVstate::Raw;
            });
    }

    /* ── 2) wait for READ group completion ──────────────────────────────── */
    thread_pool->wait_idle(CoreType::READ);

    /* ── 3) close FDs (only needed layers were opened, so just close them) ─── */
    for (int l = prev_cur_layer_idx; l < num_layers; ++l) {
        if (layer_fds[l] >= 0) { ::close(layer_fds[l]); layer_fds[l] = -1; }
    }
}




void mzcache_kv_cache::reconfigure_thread_pool(
        const std::map<CoreType,std::vector<int>>& new_cfg)
{
    // 1) finish all remaining work first
    if (thread_pool) thread_pool->wait_idle_all();

    // 2) create a new pool and move-assign it into the unique_ptr
    thread_pool = std::make_unique<ThreadPool>(new_cfg);

    // 3) (optional) update the current config too
    core_configs = new_cfg;
}

#endif // MZCACHE_SVM_KV_CHUNK