#pragma once

#include "mzcache_types.h"
#include "mzcache_threadpool.h"
#include "mzcache_profile.h"
#include "llama-model.h"
#include "llama-kv-cache-unified.h"

#ifdef MZCACHE_USE_OPENCL
#include <CL/cl.h>
#endif

#include <atomic>
#include <cstdint>
#include <cassert>
#include <fstream>

class mzcache_core;

// Compressed-KV arena. A resident swap-out arena must stay in RAM while the app
// is cached in the background: an ordinary malloc pool is anonymous memory zram
// swaps out, turning the next swapin into a major-fault storm. Two ways to keep
// it resident (see mzcache_kv_state.cpp): default = malloc + mlock the used
// bytes at swap-out end (needs root / CAP_IPC_LOCK); MZCACHE_ARENA_SVM=1 =
// coarse-grain OpenCL SVM on kgsl driver pages the kernel does not reclaim (no
// mlock, no root, eager page commit). Reload arenas are always plain malloc.
// Ctor/dtor live in mzcache_kv_state.cpp: the SVM path uses mzcache's private
// MZCACHE_USE_OPENCL define, so it must not be inlined into other TUs.
struct Arena {
    uint8_t* pool;         // 4KB-aligned: malloc (default) or OpenCL SVM
    size_t   top;
    size_t   cap;
    bool     svm;          // pool is clSVMAlloc'd (MZCACHE_ARENA_SVM), not malloc
    bool     locked;       // malloc pool mlock'd by lock_used() (resident swap-out)
    size_t   locked_bytes; // bytes mlock'd (for munlock)

    // allow_svm=true (swap-out) honors MZCACHE_ARENA_SVM; false (reload) = always malloc.
    explicit Arena(size_t size, bool allow_svm = false);
    ~Arena();
    void lock_used();      // mlock [pool, top): keep resident malloc arena out of zram (no-op if svm)
};

class mzcache_kv_state {
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
    mzcache_kv_state();
    mzcache_kv_state(mzcache_core* _core, int _num_layers, int _kv_hidden_dim, int _num_chunks_per_tensor, std::string _model_name, int _per_layer_balance = -1);
    ~mzcache_kv_state();

    int per_layer_balance;
    int default_chunk_bound; // default chunk bound for decompression
    int chunk_bound[MAX_NUM_LAYERS];

    int cur_layer_idx_to_compress; // current layer index to compress
    int cur_layer_idx_to_store;    // current layer index to store
    int cur_chunk_idx_to_compress; // current chunk index to compress
    int cur_chunk_idx_to_store;   // current chunk index to store

    int max_comp_chunks;
    int max_store_chunks;

    int n_cur_comp_chunks = 0;
    int n_cur_store_chunks = 0;

    // Actual bytes freed by compression (raw size - compressed size, measured)
    // for the chunks currently in Compressed state, and by arena offload.
    // Cumulative across swapout calls so the resident-ratio math is consistent
    // from call to call (the old per-call accounting made swapout's returned
    // ratio drift once a previous call had compressed anything).
    int64_t n_cur_comp_saved_bytes = 0;
    int64_t n_cur_arena_offload_bytes = 0;

    // compressed chunks
    std::vector<Arena*> arenas[MAX_NUM_LAYERS];

    // Per-arena pool size: covers a whole layer's chunks at the compile-time
    // compression backend's worst-case output (computed in the constructor).
    size_t arena_bytes = 0;

    // Per-layer quantization metadata (mins/maxs) pools; SVM-backed like the
    // arenas so they cannot be reclaimed to zram (see constructor).
    uint8_t* meta_pool[MAX_NUM_LAYERS] = {nullptr};
    bool     meta_pool_svm[MAX_NUM_LAYERS] = {false};

    // mins/maxs blocks for runtime-grown chunks (one block per growth event
    // per layer), freed in the destructor alongside meta_pool. bool = SVM.
    std::vector<std::pair<uint8_t*, bool>> grown_meta_pools[MAX_NUM_LAYERS];

    // Compressed chunks still to decompress per layer in the current swapin
    // (seeded by initialize_layer_sync). When a layer's last one is done the
    // decomp worker hands that layer's arenas to CoreType::FREE for progressive
    // release, off the layer-sync critical path (see kv_decomp_worker).
    std::atomic<int> comp_left[MAX_NUM_LAYERS] = {};

    std::vector<std::ofstream>   kv_store_ofs;      // per-layer binary file streams
    std::vector<int>              layer_fds;
    std::vector<size_t>          kv_store_offsets;  // per-layer current file write position (bytes)

    int num_middle_cores;
    int num_big_cores;


    int64_t mzcache_compress_chunks(int num_chunks);
    void mzcache_decompress_chunks(int num_chunks);

    std::pair<double,double> mzcache_decompress_chunks_profile(int num_chunks);

    void mzcache_store_chunks(int num_chunks);
    void mzcache_close_stored_files();
    void mzcache_load_chunks(int num_chunks);

    std::pair<double,double> mzcache_load_chunks_profile(int num_chunks);


    void mzcache_enqueue_decomp_chunks(int decomp_num_chunks);
    void mzcache_enqueue_load_chunks(int load_num_chunks);
    void mzcache_enqueue_decomp_load_chunks(int decomp_num_chunks, int load_num_chunks);


    bool mzcache_enqueue_kv_alloc_only(int decomp_num_chunks, int load_num_chunks);
    bool mzcache_wait_for_kv(int decomp_num_chunks, int load_num_chunks);
    bool mzcache_enqueue_kv_decomp_read_only(int decomp_num_chunks, int load_num_chunks);

    void clean_after_decompress(int decomp_chunks);
    void clean_after_read(int read_chunks);

    // Free one fully-decompressed layer's compression arenas; enqueued to
    // CoreType::FREE from kv_decomp_worker so clSVMFree stays off the critical path.
    void free_layer_arenas(int layer);
    void mlock_resident_arenas();  // mlock resident arenas' used bytes at swap-out end

    // Runtime KV chunk growth (llama_kv_cache_unified::mz_grow_chunks →
    // mzcache_core::on_kv_chunks_grown): register the newly materialized
    // chunks (raw pointers, Raw state, mins/maxs metadata) and extend the
    // swapout plan. Planning is only re-derived while no swapout is in
    // flight; otherwise the growth stays pending until the next swapout
    // starts from a clean ladder (see apply_pending_growth).
    void register_grown_chunks(int n_chunks_total);
    void apply_pending_growth();
    int  pending_chunks_total = 0;


    // Worker functions
    void kv_alloc_worker(int layer, int chunk, std::shared_ptr<std::promise<void>> alloc_prom);
    void kv_read_worker(int layer, int chunk);
    void kv_decomp_worker(int layer, int chunk);


    // Arena offload, reload functions
    int64_t mzcache_offload_arenas();
    void mzcache_reload_arenas();
    void mzcache_reload_all_arenas();

    int next_layer_to_offload;   // index of the next layer to offload (initial value: num_layers-1)


    // set threadpool when reconfiguring
    void set_thread_pool(ThreadPool* tp) {
        thread_pool = tp;
    }

    std::vector<size_t> offload_used_bytes_[MAX_NUM_LAYERS];  // per-layer arena top snapshots
    std::string get_swap_path(int layer_idx) const {
        return "mzcache_arenas_L" + std::to_string(layer_idx) + ".swap";
    }

private:
    // (Any internal helpers or caches could go here)
    mzcache_core* core;
    ThreadPool* thread_pool;

    mzcache::SocClass soc = mzcache::SocClass::UNKNOWN; // detected chipset (see mzcache_profile.h)

    // Derive per-layer chunk_bound, max_comp/max_store and the swapout cursors
    // from the current num_chunks_per_tensor (shared by the constructor and
    // register_grown_chunks).
    void derive_swapout_plan();
};