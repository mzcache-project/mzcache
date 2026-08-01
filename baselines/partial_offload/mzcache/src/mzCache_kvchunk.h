#pragma once

#include "mzCache_types.h"

#include "llama-model.h"
#include "llama-kv-cache-unified.h"
#include "mzCache_threadpool.h"

#ifdef MZCACHE_USE_OPENCL
#include <CL/cl.h>
#endif

#include <cstdint>
#include <cassert>
#include <fstream>

using ShFut = std::shared_future<void>;

static constexpr size_t STACK_SIZE = 256ULL * 1024 * 1024; // 256 MB

struct ChunkHeader {
    size_t comp_size;
};

struct Arena {
    uint8_t* pool;     // malloc'ed 256 MB
    size_t   top;      // next allocation position within the pool (offset)

    Arena() : pool((uint8_t*)std::malloc(STACK_SIZE)), top(0) {
        assert(pool && "arena malloc failed");
    }
    ~Arena() {
        std::free(pool);
    }
};

struct PairHash {
    size_t operator()(const std::pair<int,int>& p) const noexcept {
        return (size_t(p.first) << 32) ^ size_t(p.second);
    }
}; // For unordered_map key


class mzcache_kv_cache {
public:
    // 1. Basic metadata : num_layers, raw_chunk_size (derivable from hidden_dim), compressed_block_size, num_chunks
    int     num_layers;        // number of layers in the model
    int     kv_hidden_dim;     // hidden dimension (so raw_chunk_size = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(half))
    int     num_chunks_per_tensor;        // total number of KV-chunks
    std::string model_name;   // model name


    // 3. Restoration metadata
    bool    is_stage1;         // true if currently in Stage 1

    // 4. Per-layer KV-chunk trees
    layer_kv_chunks layers[MAX_NUM_LAYERS];

    // (Optional) you can declare constructors, destructors, methods here:
    mzcache_kv_cache();
    mzcache_kv_cache(int _num_layers, int _kv_hidden_dim, int _num_chunks_per_tensor, std::string _model_name);
    ~mzcache_kv_cache();

    int layer_bound = 11;
    int chunk_bound = 28;


    int cur_layer_idx; // current layer index to compress/store
    int cur_chunk_idx_to_compress; // current chunk index to compress
    int cur_chunk_idx_to_store;   // current chunk index to store

    // compressed chunks
    std::vector<Arena*> arenas;

    // Thread pool -> This should be moved to upper level
    std::unique_ptr<ThreadPool> thread_pool;
    std::map<CoreType,std::vector<int>> core_configs = {
        { CoreType::ALLOC,   {0,1,2,3,4} },
        { CoreType::DECOMP, {6,7} },
        { CoreType::READ,  {0,1} },

        { CoreType::DECOMP_KERNEL, {6,7} } // Not used: For multi-threaded decompression kernels
    };

    std::unordered_map<std::pair<int,int>, ShFut, PairHash> alloc_done;

    std::vector<std::ofstream>   kv_store_ofs;      // per-layer binary file streams
    std::vector<int>              layer_fds;
    std::vector<size_t>          kv_store_offsets;  // per-layer current file write position (bytes)

    int num_middle_cores;
    int num_big_cores;

    // e.g.
    // void reset();
    // bool load_layer(int layer_idx);
    // …

    // For reconfiguration of thread pool
    void reconfigure_thread_pool(const std::map<CoreType, std::vector<int>>& new_cfg);
    void mzcache_compress_chunks(int num_chunks);
    void mzcache_decompress_chunks(int num_chunks);
    void mzcache_decompress_chunks_sequential(int num_chunks);
    std::pair<double,double> mzcache_decompress_chunks_profile(int num_chunks);


    void mzcache_store_chunks(int num_chunks);
    void mzcache_load_chunks(int num_chunks);
    double mzcache_load_chunks_sequential(int num_chunks);

    size_t arena_count() const { return arenas.size(); }

    

private:
    // (Any internal helpers or caches could go here)
};