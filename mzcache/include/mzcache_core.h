#pragma once

#include "mzcache_types.h"
#include "mzcache_threadpool.h"
#include "mzcache_profile.h"

#include <unordered_map>
#include <vector>
#include <memory>
#include <future>
#include <map>
#include <atomic>
#include <tuple>


// Forward declarations
class mzcache_kv_state;
class mzcache_weight;

struct llama_context;
struct llama_batch;
struct llama_model;

using ShFut = std::shared_future<void>;

extern std::promise<void> g_graph_alloc_promise;
extern std::shared_future<void> g_graph_alloc_future;
extern std::atomic<bool> g_graph_alloc_signaled;

// PairHash definition
struct PairHash {
    size_t operator()(const std::pair<int,int>& p) const noexcept {
        return (size_t(p.first) << 32) ^ size_t(p.second);
    }
};

struct LayerSync {
    std::atomic<int>      remaining;
    std::promise<void>    prom;    // for setting the value
    ShFut                 fut;     // for waiting (shared by multiple threads)

    explicit LayerSync(int init = 0)
        : remaining(init),
          fut(prom.get_future().share())     // created exactly once, here
    {
        if (init == 0)
            prom.set_value();                // nothing to do from the start: ready immediately
    }

    void taskDone() {
        if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1)
            prom.set_value();                // exactly once, at the moment it reaches 0
    }

    ShFut getFuture() const { return fut; }  // returns the already-shared future

    int getRemaining() const {
        return remaining.load(std::memory_order_acquire);
    }

    // Poll until the remaining task count reaches 0,
    // sleeping `interval` between checks
    void waitPolling(
        std::chrono::milliseconds interval = std::chrono::milliseconds(10)
    ) const {
        while (getRemaining() > 0) {
            std::cout << "[LayerSync] remaining = " 
                      << getRemaining() << "\n";
            std::this_thread::sleep_for(interval);
        }
    }
};

class mzcache_core {
public:
    mzcache_core(int _num_layers, int _kv_hidden_dim, int _num_chunks_per_tensor, llama_model * _model);
    ~mzcache_core();

    
    // Allocation-completion tracking
    void register_kv_allocation(int layer_idx, int chunk_idx, std::shared_ptr<std::promise<void>> prom);
    void wait_for_kv_allocation(int layer_idx, int chunk_idx);

    void register_weight_allocation(int layer_idx, std::shared_ptr<std::promise<void>> prom);
    void wait_for_weight_allocation(int layer_idx);

    std::map<CoreType, std::vector<int>>& get_core_configs() { return core_configs; }
    ThreadPool* get_thread_pool() { return thread_pool.get(); }

    mzcache_kv_state* get_kv_state() { return kv_state.get(); }
    mzcache_weight* get_weight() { return weight.get(); }
    
    float get_decomp_load_ratio() const {
        return decomp_load_ratio;
    }

    // Reconfigure the thread pool
    void reconfigure_thread_pool(const std::map<CoreType, std::vector<int>>& new_cfg);

    std::vector<std::unique_ptr<LayerSync>> layer_sync;
    void initialize_layer_sync();

    size_t layer_sync_size() const {
        return layer_sync.size();
    }
    void wait_for_layer_sync(int layer_idx) const {
        if (layer_idx >= 0 && layer_idx < (int)layer_sync.size()) {
            layer_sync[layer_idx]->getFuture().wait();
        }
    }

    void wait_polling_layer_sync(int layer_idx, std::chrono::milliseconds interval = std::chrono::milliseconds(10)) const {
        if (layer_idx >= 0 && layer_idx < (int)layer_sync.size()) {
            layer_sync[layer_idx]->waitPolling(interval);
        }
    }


    long swapin_generate(llama_context* ctx, llama_batch batch, bool offload_compressed_kv);

    std::tuple<long, long> swapin_generate_no_prefill_overlap(llama_context* ctx, llama_batch batch, bool offload_compressed_kv);
    std::tuple<long, long> swapin_generate_no_alloc_overlap(llama_context* ctx, llama_batch batch, bool offload_compressed_kv);
    std::tuple<long, long, long> swapin_generate_no_overlap(llama_context* ctx, llama_batch batch, bool offload_compressed_kv);

    int64_t get_freed_bytes(int n_cur_comp_chunks, int n_cur_store_chunks, int n_cur_unload_layers);
    std::tuple<float, int, int, int> mzcache_profile_chunks_layers(float target_ratio);

    std::tuple<float, int, int, int> swapout(float target_ratio);
    std::tuple<float, int, int, int> swapout_impl(float target_ratio);  // body; swapout() wraps + mlocks

    // Runtime KV chunk growth: called by llama_kv_cache_unified::mz_grow_chunks
    // after new chunks were materialized (decode/generation crossed into a
    // chunk that did not exist yet). Updates the core/kv_state bookkeeping so
    // later swapouts cover the new chunks too.
    void on_kv_chunks_grown(int n_chunks_total);

    int64_t weight_layer_bytes = 0;
    int64_t full_bytes = 0; 

private:

    void finish_swapin_cycle(int decomp_chunks, int load_chunks);

    int num_layers;          // number of layers in the model
    int num_chunks_per_tensor;
    float decomp_load_ratio;

    // This values are initialized in initialize_layer_sync()
    int decomp_num_chunks = 0;
    int load_num_chunks = 0;

    mzcache::SocClass soc = mzcache::SocClass::UNKNOWN; // detected chipset (see mzcache_profile.h)

    // KV cache and weight instances
    std::unique_ptr<mzcache_weight> weight;
    std::unique_ptr<mzcache_kv_state> kv_state;

    std::mutex kv_alloc_mutex;
    std::unordered_map<std::pair<int,int>, ShFut, PairHash> kv_alloc_done;

    std::mutex weight_alloc_mutex;
    std::unordered_map<int, ShFut> weight_alloc_done;

    std::unique_ptr<ThreadPool> thread_pool;
    std::map<CoreType, std::vector<int>> core_configs;
};

extern mzcache_core* g_mzcache_core;

namespace mzcache {
// Decode `batch` with llama_context::decode, split at 256-token chunk
// boundaries: a KV-cache write must not span two chunks (cpy_k/cpy_v build a
// single chunk-local view), so a batch that crosses a boundary is issued as
// consecutive sub-decodes. Assumes the batch is contiguous ascending positions
// on one sequence (cache slot == position), as in the mz flows. Returns the
// first nonzero decode result, or 0.
int decode_chunk_aligned(llama_context * ctx, const llama_batch & batch);
} // namespace mzcache
