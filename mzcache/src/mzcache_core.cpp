#include "mzcache_core.h"
#include "mzcache_profile.h"
#include "mzcache_threadpool.h"
#include "mzcache_kv_state.h"
#include "mzcache_weight.h"

#include "llama-context.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-mmap.h"

#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>
#include <errno.h>

static timespec get_monotonic_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

static long diff_microseconds(const timespec &start, const timespec &end) {
    return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
}

mzcache_core* g_mzcache_core = nullptr;
std::promise<void> g_graph_alloc_promise;
std::shared_future<void> g_graph_alloc_future;
std::atomic<bool> g_graph_alloc_signaled(false);


#ifdef MZCACHE_SVM_KV_CHUNK
mzcache_core::mzcache_core(int _num_layers, int _kv_hidden_dim, int _num_chunks_per_tensor, llama_model * _model) :
    num_layers(_num_layers), num_chunks_per_tensor(_num_chunks_per_tensor)
{
    g_mzcache_core = this;

    if (!g_graph_alloc_signaled.load()) {
        g_graph_alloc_promise = std::promise<void>();
        g_graph_alloc_future = g_graph_alloc_promise.get_future().share();
        g_graph_alloc_signaled.store(false);
    }

    // Chipset detection: core placement and the legacy constants below are
    // keyed by the SoC (CPU topology + Adreno generation), not the phone
    // vendor — any device with the same chipset takes the same paths, and
    // per-device speed differences are corrected by the offline profile.
    const std::string soc_str = mzcache::soc_model();
    soc = mzcache::classify_soc(soc_str);
    std::cout << "SoC: " << soc_str << " -> " << mzcache::soc_name(soc) << std::endl;
    if (soc == mzcache::SocClass::UNKNOWN) {
        std::cerr << "[MZCACHE][WARN] unrecognized SoC '" << soc_str
                  << "' — supported: SM8750 (Snapdragon 8 Elite), SM8650 (Snapdragon 8 Gen 3).\n"
                  << "[MZCACHE][WARN] falling back to the SM8650 core placement; scheduling may be suboptimal.\n";
    }

    // Core placement shared with examples/mz_device_profile (compression- and
    // SoC-dependent; see mzcache_profile.cpp).
    core_configs = mzcache::default_core_configs(soc);

    // The two scheduling constants are derived from the one-time device
    // profile (full swap-in emulation throughputs) plus the model shape.
    // Without a profile the engine refuses to start.
    mzcache::device_profile dev_prof;
    bool have_profile = false;
    if (mzcache::profiling_supported()) {
        const std::string profile_path = mzcache::device_profile_path();
        if (!mzcache::load_device_profile(profile_path, dev_prof)) {
            std::cerr << "[MZCACHE][FATAL] device profile '" << profile_path
                      << "' not found in the working directory.\n"
                      << "[MZCACHE][FATAL] Run the one-time offline profiler on this device first:\n"
                      << "[MZCACHE][FATAL]   cd /data/local/tmp/mzcache && ./<install>/bin/mz_device_profile\n";
            throw std::runtime_error("mzcache: device profile missing - run mz_device_profile first");
        }
        if (dev_prof.compression != mzcache::compression_name()) {
            std::cerr << "[MZCACHE][FATAL] device profile was measured with compression '"
                      << dev_prof.compression << "' but this build uses '"
                      << mzcache::compression_name() << "'. Re-run mz_device_profile.\n";
            throw std::runtime_error("mzcache: device profile compression mismatch");
        }
        have_profile = true;
        decomp_load_ratio = mzcache::derive_decomp_load_ratio(dev_prof);
        std::cout << "decomp_load_ratio derived from device profile: " << decomp_load_ratio
                  << " (D " << dev_prof.decomp_gbps << " GB/s, R_kv "
                  << dev_prof.kv_read_gbps << " GB/s)\n";
    } else {
        // Legacy constants for CACHEGEN / NONE builds (no profiler support).
        decomp_load_ratio = (soc == mzcache::SocClass::SM8750) ? 0.16f : 0.f;
    }

    // Experiment/auto-config override: value from mz_device_profile (or manual).
    if (const char * e = getenv("MZCACHE_DECOMP_LOAD_RATIO")) {
        const float v = (float) atof(e);
        if (v > 0.0f && v <= 1.0f) {
            decomp_load_ratio = v;
            std::cout << "decomp_load_ratio overridden via env: " << v << "\n";
        }
    }

    thread_pool = std::make_unique<ThreadPool>(core_configs);
    kv_alloc_done.reserve(_num_layers * _num_chunks_per_tensor);
    weight_alloc_done.reserve(_num_layers);

    //sanitize model name
    std::string model_name = _model->name;
    std::replace(model_name.begin(), model_name.end(), ' ', '_');
    _model->name = model_name;
    std::cout << "Model name sanitized to: " << _model->name << std::endl;

    if(model_name == "EXAONE-4.0-1.2B") {
        weight_layer_bytes = 71323648;
    }
    else if(model_name == "Qwen3-0.6B") {
        weight_layer_bytes = 31469568;
    }

    full_bytes = weight_layer_bytes * _num_layers + TOKENS_PER_CHUNK * _kv_hidden_dim * sizeof(__fp16) * 2 * _num_chunks_per_tensor * _num_layers;

    int profiled_balance = -1;
    if (have_profile && weight_layer_bytes > 0) {
        profiled_balance = mzcache::derive_per_layer_balance(dev_prof, weight_layer_bytes, _kv_hidden_dim);
        std::cout << "per_layer_balance derived from device profile: " << profiled_balance
                  << " (layer " << weight_layer_bytes << " B, R_w "
                  << dev_prof.weight_read_gbps << " GB/s)\n";
    }

    kv_state = std::make_unique<mzcache_kv_state>(this, _num_layers, _kv_hidden_dim, _num_chunks_per_tensor, _model->name, profiled_balance);
    weight = std::make_unique<mzcache_weight>(this, _model);
}
mzcache_core::~mzcache_core() {
    // Ensure all threads are done before destruction
    if (thread_pool) {
        thread_pool->wait_idle_all();
    }
}

// Register a promise for allocation completion
void mzcache_core::register_kv_allocation(int layer_idx, int chunk_idx, std::shared_ptr<std::promise<void>> prom) {
    auto key = std::make_pair(layer_idx, chunk_idx);
    auto fut = prom->get_future().share();
    {
        std::lock_guard lock(kv_alloc_mutex);
        kv_alloc_done[key] = fut;
    }
}

void mzcache_core::wait_for_kv_allocation(int layer_idx, int chunk_idx) {
    std::shared_future<void> f;
    {
        std::lock_guard lock(kv_alloc_mutex);
        auto it = kv_alloc_done.find({layer_idx, chunk_idx});
        if (it == kv_alloc_done.end()) {
            throw std::runtime_error("Allocation future not found for layer_idx " 
                                     + std::to_string(layer_idx) + ", chunk_idx " 
                                     + std::to_string(chunk_idx));
        }
        f = it->second;  // copy
    }
    f.wait();
}


void mzcache_core::register_weight_allocation(int layer_idx, std::shared_ptr<std::promise<void>> prom) {
    // weight_alloc_done[layer_idx] = prom->get_future().share();
    auto fut = prom->get_future().share();
    {
        std::lock_guard lock(weight_alloc_mutex);
        weight_alloc_done[layer_idx] = fut;
    }
}

void mzcache_core::wait_for_weight_allocation(int layer_idx) {
    std::shared_future<void> f;
    {
        std::lock_guard lock(weight_alloc_mutex);
        auto it = weight_alloc_done.find(layer_idx);
        if (it == weight_alloc_done.end()) {
            throw std::runtime_error("Weight allocation future not found for layer " + std::to_string(layer_idx));
        }
        f = it->second;  // copy
    }
    f.wait();
}


void mzcache_core::reconfigure_thread_pool(
        const std::map<CoreType,std::vector<int>>& new_cfg)
{
    // 1) Finish all remaining tasks
    if (thread_pool) thread_pool->wait_idle_all();

    // 2) Create a new pool and move-assign it into the unique_ptr
    thread_pool = std::make_unique<ThreadPool>(new_cfg);

    kv_state->set_thread_pool(thread_pool.get());
    weight->set_thread_pool(thread_pool.get());

    // 3) (optional) also update the current config
    core_configs = new_cfg;
}


void mzcache_core::initialize_layer_sync() {
    int  L      = num_layers;

    auto &st    = *this->kv_state;
    int  N      = st.num_chunks_per_tensor;
    int  curC   = st.cur_chunk_idx_to_compress;
    int  curS   = st.cur_chunk_idx_to_store;

    auto &weight = *this->weight;
    int  Wc     = weight.cur_layer_idx_to_unload;

    decomp_num_chunks = 0;
    load_num_chunks = 0;

    {
        std::lock_guard lock(kv_alloc_mutex);
        kv_alloc_done.clear();
    }
    {
        std::lock_guard lock(weight_alloc_mutex);
        weight_alloc_done.clear();
    }

    layer_sync.clear();
    layer_sync.reserve(L);

    for (int i = 0; i < L; ++i) {

        int Bc = st.chunk_bound[i]; // Bc: Bound of chunks for layer i
        // ── compress-side initial count ──────────────────
        int compress_init;
        if      (i <  st.cur_layer_idx_to_compress)    compress_init = 0;
        else if (i == st.cur_layer_idx_to_compress)    compress_init = Bc - (curC + 1);
        else                                           compress_init = Bc;

        // ── store-side initial count ─────────────────────
        int store_init;
        if      (i <  st.cur_layer_idx_to_store)       store_init = 0;
        else if (i == st.cur_layer_idx_to_store)       store_init = N - (curS + 1);
        else                                           store_init = N - Bc;

        // ── Sum them to manage everything with a single sync; to track them
        //    separately, a vector of two LayerSyncs would also work.
        int total_init = compress_init + store_init;

        // ── weight-side initial count ────────────────────
        if(i > Wc)
            total_init += 1;

        decomp_num_chunks += compress_init;
        load_num_chunks += store_init;

        // for progressive per-layer arena free during swapin (see kv_decomp_worker)
        st.comp_left[i].store(compress_init);

        // std::cout << "Layer " << i
        //           << ": compress_init = " << compress_init 
        //           << ", store_init = " << store_init 
        //           << ", total_init = " << total_init << std::endl;

        layer_sync.emplace_back(std::make_unique<LayerSync>(total_init));
    }
}

void mzcache_core::finish_swapin_cycle(int decomp_chunks, int load_chunks) {
    for (int i = 0; i < num_layers; ++i) {
        wait_for_layer_sync(i);
    }

    if (thread_pool) {
        thread_pool->wait_idle_all();
    }

    kv_state->clean_after_decompress(decomp_chunks);
    kv_state->clean_after_read(load_chunks);

    weight->cur_layer_idx_to_unload = num_layers - 1;
    weight->n_cur_unload_layers = 0;
}

long mzcache_core::swapin_generate(llama_context* ctx, llama_batch batch, bool offload_compressed_kv) {

    // Reset the future
    if (g_graph_alloc_signaled.load()) {
        g_graph_alloc_promise = std::promise<void>();
        g_graph_alloc_future = g_graph_alloc_promise.get_future().share();
        g_graph_alloc_signaled.store(false);
    }

    initialize_layer_sync(); // Ensure layer sync is initialized

    timespec TTFT_s = get_monotonic_time();

    kv_state->mzcache_reload_all_arenas();

    timespec t_arena = get_monotonic_time();

    std::future<int> decode_result = std::async(std::launch::async, [ctx, batch]() {

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(7, &cpuset); // bind to CPU 7 (big core)
        
        // Get the current thread ID and set its affinity
        pid_t tid = syscall(SYS_gettid);
        if (sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "Failed to set CPU affinity for decode thread, errno=" << errno << std::endl;
        } else {
            // std::cout << "Successfully set decode thread to run on CPU 7" << std::endl;
        }

        const int ret = mzcache::decode_chunk_aligned(ctx, batch);
        if (ret != 0 && ret != 1) {
            LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
        }
        return ret;
    });
    
    // Wait until graph allocation completes
    // std::cout << "Waiting for graph allocation to complete..." << std::endl;
    g_graph_alloc_future.wait();
    // std::cout << "Graph allocation completed, starting mzCache operations" << std::endl;

    timespec t_graph = get_monotonic_time();

    // Now run the mzCache work
    kv_state->mzcache_enqueue_decomp_load_chunks(decomp_num_chunks, load_num_chunks);

    timespec t_enq = get_monotonic_time();

    weight->mzcache_reload_w_layers();

    timespec t_weight = get_monotonic_time();

    // Wait for decode to finish and fetch the result
    int ret = decode_result.get();

    timespec TTFT_e = get_monotonic_time();

    long TTFT = diff_microseconds(TTFT_s, TTFT_e);

    MZ_LOG_INFO("swapin phases (ms): arena_reload=%.1f graph_wait=%.1f enqueue=%.1f weight_reload=%.1f decode_wait=%.1f total=%.1f",
                diff_microseconds(TTFT_s, t_arena) / 1000.0,
                diff_microseconds(t_arena, t_graph) / 1000.0,
                diff_microseconds(t_graph, t_enq) / 1000.0,
                diff_microseconds(t_enq, t_weight) / 1000.0,
                diff_microseconds(t_weight, TTFT_e) / 1000.0,
                TTFT / 1000.0);

    finish_swapin_cycle(decomp_num_chunks, load_num_chunks);

    return TTFT;
}

std::tuple<long, long> mzcache_core::swapin_generate_no_prefill_overlap(llama_context* ctx, llama_batch batch, bool offload_compressed_kv) {

    // Now run the mzCache work
    initialize_layer_sync(); // Ensure layer sync is initialized

    timespec t0 = get_monotonic_time();

    kv_state->mzcache_reload_all_arenas();

    kv_state->mzcache_enqueue_decomp_load_chunks(decomp_num_chunks, load_num_chunks);    
    weight->mzcache_reload_w_layers();


    for(int i = 0; i < num_layers; i++) {
        wait_for_layer_sync(i);
    }

    timespec t2 = get_monotonic_time();


    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(7, &cpuset); // bind to CPU 7 (big core)

        pid_t tid = syscall(SYS_gettid);
        if (sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "Failed to set CPU affinity for decode (tid=" << tid
                      << "), errno=" << errno << std::endl;
        } else {
            std::cout << "Successfully set decode to run on CPU 7 (tid=" << tid << ")\n";
        }
    }

    const int ret = mzcache::decode_chunk_aligned(ctx, batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    timespec t3 = get_monotonic_time();
    long alloc_restore = diff_microseconds(t0, t2);
    long prefill = diff_microseconds(t2, t3);

    finish_swapin_cycle(decomp_num_chunks, load_num_chunks);

    return std::make_tuple(alloc_restore, prefill);
}

std::tuple<long, long> mzcache_core::swapin_generate_no_alloc_overlap(llama_context* ctx, llama_batch batch, bool offload_compressed_kv) {
    // Reset the future
    if (g_graph_alloc_signaled.load()) {
        g_graph_alloc_promise = std::promise<void>();
        g_graph_alloc_future = g_graph_alloc_promise.get_future().share();
        g_graph_alloc_signaled.store(false);
    }


    
    // Start the decode function first (runs asynchronously and waits until graph allocation completes)
    std::future<int> decode_result = std::async(std::launch::async, [ctx, batch]() {

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(7, &cpuset); // bind to CPU 7 (big core)
        
        // Get the current thread ID and set its affinity
        pid_t tid = syscall(SYS_gettid);
        if (sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "Failed to set CPU affinity for decode thread, errno=" << errno << std::endl;
        } else {
            // std::cout << "Successfully set decode thread to run on CPU 7" << std::endl;
        }

        const int ret = mzcache::decode_chunk_aligned(ctx, batch);
        if (ret != 0 && ret != 1) {
            LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
        }
        return ret;
    });
    
    // Wait until graph allocation completes
    // std::cout << "Waiting for graph allocation to complete..." << std::endl;
    g_graph_alloc_future.wait();
    // std::cout << "Graph allocation completed, starting mzCache operations" << std::endl;
    
    // Now run the mzCache work
    initialize_layer_sync(); // Ensure layer sync is initialized

    timespec t0 = get_monotonic_time(); /////////////////

    kv_state->mzcache_reload_all_arenas();

    weight->mzcache_reload_w_layers_alloc_only();
    kv_state->mzcache_enqueue_kv_alloc_only(decomp_num_chunks, load_num_chunks);
    for (int L = weight->cur_layer_idx_to_unload + 1; L < num_layers; L++) {
        wait_for_weight_allocation(L);
    }
    kv_state->mzcache_wait_for_kv(decomp_num_chunks, load_num_chunks);

    timespec t1 = get_monotonic_time(); ////////////////////

    kv_state->mzcache_enqueue_kv_decomp_read_only(decomp_num_chunks, load_num_chunks);
    weight->mzcache_reload_w_layers_reload_only();


    // Wait for decode to finish and fetch the result
    int ret = decode_result.get();
    timespec t3 = get_monotonic_time(); ////////////////////////

    long alloc_time = diff_microseconds(t0, t1);
    long prefill_restore_time = diff_microseconds(t1, t3);

    finish_swapin_cycle(decomp_num_chunks, load_num_chunks);

    return std::make_tuple(alloc_time, prefill_restore_time);
}

std::tuple<long, long, long> mzcache_core::swapin_generate_no_overlap(llama_context* ctx, llama_batch batch, bool offload_compressed_kv) {

    // Now run the mzCache work
    initialize_layer_sync(); // Ensure layer sync is initialized

    timespec t0 = get_monotonic_time(); ////////////////////////

    kv_state->mzcache_reload_all_arenas();
    
    weight->mzcache_reload_w_layers_alloc_only();
    kv_state->mzcache_enqueue_kv_alloc_only(decomp_num_chunks, load_num_chunks);
    for (int L = weight->cur_layer_idx_to_unload + 1; L < num_layers; L++) {
        wait_for_weight_allocation(L);
    }
    kv_state->mzcache_wait_for_kv(decomp_num_chunks, load_num_chunks);

    timespec t1 = get_monotonic_time(); ////////////////////////

    kv_state->mzcache_enqueue_kv_decomp_read_only(decomp_num_chunks, load_num_chunks);
    weight->mzcache_reload_w_layers_reload_only();

    for(int i = 0; i < num_layers; i++) {
        wait_for_layer_sync(i);
    }

    timespec t2 = get_monotonic_time(); ////////////////////////


    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(7, &cpuset); // bind to CPU 7 (big core)

        pid_t tid = syscall(SYS_gettid);
        if (sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "Failed to set CPU affinity for decode (tid=" << tid
                      << "), errno=" << errno << std::endl;
        } else {
            std::cout << "Successfully set decode to run on CPU 7 (tid=" << tid << ")\n";
        }
    }

    const int ret = mzcache::decode_chunk_aligned(ctx, batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }
    timespec t3 = get_monotonic_time(); ////////////////////////

    long alloc_time = diff_microseconds(t0, t1);
    long restore_time = diff_microseconds(t1, t2);
    long prefill_time = diff_microseconds(t2, t3);

    finish_swapin_cycle(decomp_num_chunks, load_num_chunks);

    return std::make_tuple(alloc_time, restore_time, prefill_time);
}

// Nominal fraction of a raw chunk freed by compressing it, used for *planning*
// (predicting how many chunks a target ratio needs). Matches the sizes emitted
// by mzcache_compress_chunks: comp = raw/N + mins/maxs metadata (raw/32).
//   FLEXGEN (4-bit): raw/4 + raw/32 resident -> 23/32 freed
//   FLEXGEN_8BIT   : raw/2 + raw/32 resident -> 15/32 freed
#if defined(MZCACHE_COMPRESS_FLEXGEN_8BIT)
static constexpr int64_t COMP_FREED_NUM = 15;
#elif defined(MZCACHE_COMPRESS_NONE)
static constexpr int64_t COMP_FREED_NUM = 0;
#else // FLEXGEN 4-bit; CACHEGEN is variable-rate, keep the 4-bit heuristic
static constexpr int64_t COMP_FREED_NUM = 23;
#endif
static constexpr int64_t COMP_FREED_DEN = 32;

int64_t mzcache_core::get_freed_bytes(int n_cur_comp_chunks, int n_cur_store_chunks, int n_cur_unload_layers) {
    int64_t chunk_bytes = TOKENS_PER_CHUNK * kv_state->kv_hidden_dim * sizeof(__fp16) * 2;

    return n_cur_comp_chunks * chunk_bytes * COMP_FREED_NUM / COMP_FREED_DEN + // compressed chunks
           n_cur_store_chunks * chunk_bytes +            // stored chunks
           weight_layer_bytes * n_cur_unload_layers;     // unloaded layers
}

std::tuple<float, int, int, int> mzcache_core::mzcache_profile_chunks_layers(float target_ratio) {
    const int total_chunks = num_chunks_per_tensor * num_layers;
    const int step = 20;
    const int bound_chunks = total_chunks - kv_state->per_layer_balance * num_layers;
    
    // Initialize working variables from current state
    int comp_chunks = kv_state->n_cur_comp_chunks;
    int store_chunks = kv_state->n_cur_store_chunks;
    int unload_layers = weight->n_cur_unload_layers;
    int processed_chunks = comp_chunks + store_chunks;
    
    // Helper lambda to check if target is reached and print status
    auto check_target = [&]() -> std::optional<std::tuple<float, int, int, int>> {
        int64_t freed_bytes = get_freed_bytes(comp_chunks, store_chunks, unload_layers);
        float cur_ratio = 1.0f - (double)freed_bytes / (double)full_bytes;
        
        if (cur_ratio <= target_ratio) {
            std::cout << "Target ratio reached: " << cur_ratio << "\n"
                     << "comp_chunks: " << comp_chunks << "\n"
                     << "store_chunks: " << store_chunks << "\n"
                     << "unload_layers: " << unload_layers << "\n";
            return std::make_tuple(cur_ratio, comp_chunks, store_chunks, unload_layers);
        }
        return std::nullopt;
    };
    
    // Helper lambda to allocate chunks with capacity constraints
    auto allocate_chunks = [&](int requested) -> std::pair<int, int> {
        int max_comp = kv_state->max_comp_chunks - kv_state->per_layer_balance * num_layers;
        int max_store = kv_state->max_store_chunks;
        
        // Calculate desired distribution
        int want_comp = static_cast<int>(requested * decomp_load_ratio);
        int want_store = requested - want_comp;
        
        // Apply capacity constraints
        int comp_room = std::max(0, max_comp - comp_chunks);
        int store_room = std::max(0, max_store - store_chunks);
        
        int add_comp = std::min(want_comp, comp_room);
        int add_store = std::min(want_store, store_room);
        
        // Try to maintain total allocation by redistributing if needed
        int shortage = requested - (add_comp + add_store);
        if (shortage > 0) {
            // Try to compensate with remaining room
            if (add_comp < want_comp && store_room > add_store) {
                int compensate = std::min(shortage, store_room - add_store);
                add_store += compensate;
            } else if (add_store < want_store && comp_room > add_comp) {
                int compensate = std::min(shortage, comp_room - add_comp);
                add_comp += compensate;
            }
        }
        
        return {add_comp, add_store};
    };
    
    // Phase 1: Process KV chunks up to boundary
    while (processed_chunks < bound_chunks) {
        if (auto result = check_target()) return *result;
        
        int remaining = total_chunks - processed_chunks;
        int to_process = std::min(step, remaining);
        
        auto [add_comp, add_store] = allocate_chunks(to_process);
        int total_added = add_comp + add_store;
        
        if (total_added == 0) break;  // Can't allocate more
        
        comp_chunks += add_comp;
        store_chunks += add_store;
        processed_chunks += total_added;
    }
    
    // Phase 2: Unload layers while adding balance chunks
    for (int i = num_layers; i > 0; i--) {
        if (auto result = check_target()) return *result;

        int remaining = total_chunks - processed_chunks;
        int to_process = std::min(kv_state->per_layer_balance, remaining);

        comp_chunks += to_process;
        unload_layers++;
        processed_chunks += to_process;

    }
    
    // Final check
    int64_t freed_bytes = get_freed_bytes(comp_chunks, store_chunks, unload_layers);
    float final_ratio = 1.0f - (double)freed_bytes / (double)full_bytes;
    
    std::cout << (final_ratio <= target_ratio ? "Target ratio reached: " : "Cannot reach target ratio: ")
             << final_ratio << "\n"
             << "comp_chunks: " << comp_chunks << "\n"
             << "store_chunks: " << store_chunks << "\n"
             << "unload_layers: " << unload_layers << "\n";
    
    return std::make_tuple(final_ratio, comp_chunks, store_chunks, unload_layers);
}


void mzcache_core::on_kv_chunks_grown(int n_chunks_total) {
    kv_state->register_grown_chunks(n_chunks_total);

    // Follow kv_state's (possibly deferred) count so planning stays coherent.
    if (kv_state->num_chunks_per_tensor != num_chunks_per_tensor) {
        num_chunks_per_tensor = kv_state->num_chunks_per_tensor;
        full_bytes = weight_layer_bytes * num_layers +
                     (int64_t) TOKENS_PER_CHUNK * kv_state->kv_hidden_dim * sizeof(__fp16) * 2 *
                     num_chunks_per_tensor * num_layers;
    }
}

std::tuple<float, int, int, int> mzcache_core::swapout(float target_ratio) {
    auto result = swapout_impl(target_ratio);
    // Lock the resident compressed KV (arenas' used bytes) so it is not
    // reclaimed to zram while the app is backgrounded. Best-effort (needs root);
    // done here after swap-out, never on the timed swap-in path.
    kv_state->mlock_resident_arenas();
    return result;
}

std::tuple<float, int, int, int> mzcache_core::swapout_impl(float target_ratio) {
    // Chunk growth that happened while a previous ladder was in flight is
    // folded in here, from a clean starting point (no-op otherwise).
    kv_state->apply_pending_growth();
    if (kv_state->num_chunks_per_tensor != num_chunks_per_tensor) {
        num_chunks_per_tensor = kv_state->num_chunks_per_tensor;
        full_bytes = weight_layer_bytes * num_layers +
                     (int64_t) TOKENS_PER_CHUNK * kv_state->kv_hidden_dim * sizeof(__fp16) * 2 *
                     num_chunks_per_tensor * num_layers;
    }

    const int total_chunks = num_chunks_per_tensor * num_layers;
    const int step = 20;
    const int bound_chunks = total_chunks - kv_state->per_layer_balance * num_layers;
    const int64_t chunk_bytes = TOKENS_PER_CHUNK * kv_state->kv_hidden_dim * sizeof(__fp16) * 2;

    // Initialize from the current state
    const int initial_comp_chunks = kv_state->n_cur_comp_chunks;
    const int initial_store_chunks = kv_state->n_cur_store_chunks;
    const int initial_unload_layers = weight->n_cur_unload_layers;
    int processed_chunks = initial_comp_chunks + initial_store_chunks;

    // Measured freed bytes are tracked via kv_state's cumulative counters
    // (n_cur_comp_saved_bytes, n_cur_arena_offload_bytes). Not per-call sums,
    // so the returned ratio always reflects earlier calls' compression savings (no cross-call drift).
    auto current_freed = [&]() -> int64_t {
        return kv_state->n_cur_comp_saved_bytes +
               kv_state->n_cur_arena_offload_bytes +
               (int64_t)kv_state->n_cur_store_chunks * chunk_bytes +
               (int64_t)weight->n_cur_unload_layers * weight_layer_bytes;
    };

    // Helper lambda to check if target is reached
    auto check_target = [&]() -> std::optional<std::tuple<float, int, int, int>> {
        float cur_ratio = 1.0f - (double)current_freed() / (double)full_bytes;
        
        if (cur_ratio <= target_ratio) {
            std::cout << "Target ratio reached: " << cur_ratio << "\n"
                      << "comp_delta: " << kv_state->n_cur_comp_chunks - initial_comp_chunks << "\n"
                      << "store_delta: " << kv_state->n_cur_store_chunks - initial_store_chunks << "\n"
                      << "unload_delta: " << weight->n_cur_unload_layers - initial_unload_layers << "\n";
            return std::make_tuple(cur_ratio, 
                                   kv_state->n_cur_comp_chunks - initial_comp_chunks, 
                                   kv_state->n_cur_store_chunks - initial_store_chunks, 
                                   weight->n_cur_unload_layers - initial_unload_layers);
        }
        return std::nullopt;
    };
    
    // Helper lambda to allocate chunks with capacity constraints
    auto allocate_chunks = [&](int requested) -> std::pair<int, int> {
        int max_comp = kv_state->max_comp_chunks - kv_state->per_layer_balance * num_layers;
        int max_store = kv_state->max_store_chunks;
        
        int want_comp = static_cast<int>(requested * decomp_load_ratio);
        int want_store = requested - want_comp;
        
        // Apply capacity constraints using the actual current state from kv_state
        int comp_room = std::max(0, max_comp - kv_state->n_cur_comp_chunks);
        int store_room = std::max(0, max_store - kv_state->n_cur_store_chunks);
        
        int add_comp = std::min(want_comp, comp_room);
        int add_store = std::min(want_store, store_room);
        
        // Try to maintain total allocation by redistributing if needed
        int shortage = requested - (add_comp + add_store);
        if (shortage > 0) {
            // Re-check room after initial allocation to be precise for redistribution
            int current_comp_room = std::max(0, max_comp - (kv_state->n_cur_comp_chunks + add_comp));
            int current_store_room = std::max(0, max_store - (kv_state->n_cur_store_chunks + add_store));

            if (add_comp < want_comp && current_store_room > 0) {
                int compensate = std::min({shortage, want_comp - add_comp, current_store_room});
                add_store += compensate;
                shortage -= compensate;
            }
            
            if (shortage > 0 && add_store < want_store && current_comp_room > 0) {
                int compensate = std::min({shortage, want_store - add_store, current_comp_room});
                add_comp += compensate;
            }
        }
        
        return {add_comp, add_store};
    };
    
    // Initial check
    if (auto result = check_target()) return *result;

    // Phase 1: Process KV chunks up to boundary
    while (processed_chunks < bound_chunks) {
        int remaining = total_chunks - processed_chunks;
        int to_process = std::min(step, remaining);
        
        auto [add_comp, add_store] = allocate_chunks(to_process);
        
        if (add_comp > 0) {
            uint64_t compressed = kv_state->mzcache_compress_chunks(add_comp);
            if(compressed < 0) {
                std::cerr << "Error during compression in profiling.\n";
                break;
            }
        }
        if (add_store > 0) {
            kv_state->mzcache_store_chunks(add_store);
        }
        
        processed_chunks = kv_state->n_cur_comp_chunks + kv_state->n_cur_store_chunks;

        if (auto result = check_target()) {
            kv_state->mzcache_close_stored_files();
            return *result;
        }
        if (add_comp + add_store == 0) break;
    }

    // Close any open stored files before unloading layers
    kv_state->mzcache_close_stored_files();
    
    // Phase 2: Unload layers while adding balance chunks
    for (int i = num_layers - initial_unload_layers; i > 0; i--) {
        int remaining = total_chunks - processed_chunks;
        int to_process = std::min(kv_state->per_layer_balance, remaining);

        // std::cout << "i: " << i << ", to_process: " << to_process << ", total_chunks: " << total_chunks << ", processed_chunks: " << processed_chunks << "\n";

        if (to_process > 0) {
            uint64_t compressed = kv_state->mzcache_compress_chunks(to_process);
            if(compressed < 0) {
                std::cerr << "Error during compression in profiling.\n";
                break;
            }
        }
        
        if (!weight->mzcache_unload_layers(1)) {
            MZ_LOG_ERROR("swapout: stopping after weight unload failure at layer %d",
                         weight->cur_layer_idx_to_unload);
            break;
        }
        
        processed_chunks = kv_state->n_cur_comp_chunks + kv_state->n_cur_store_chunks;

        if (auto result = check_target()) return *result;
        if (weight->n_cur_unload_layers == num_layers && (kv_state->n_cur_comp_chunks + kv_state->n_cur_store_chunks) == total_chunks) break;
    }
    
    // Phase 3: Offload compressed arenas if all KV chunks and weights are processed
    int remaining = total_chunks - (kv_state->n_cur_comp_chunks + kv_state->n_cur_store_chunks);
    if (remaining == 0 && weight->n_cur_unload_layers == num_layers) {
        std::cout << "All KV chunks and weights processed. Starting arena offload phase...\n";
        
        // Offload arenas one by one until target is reached

        int num_offloaded = 0;
        while (kv_state->next_layer_to_offload >= 0) {
            kv_state->mzcache_offload_arenas();

            num_offloaded++;
            
            if (auto result = check_target()) {
                MZ_LOG_INFO("Offloaded %d arenas to reach target ratio.\n", num_offloaded);
                return *result;
            }
        }
        MZ_LOG_INFO("All arenas offloaded.\n");
        std::cout << "All arenas offloaded.\n";
    }
    
    // Final check
    float final_ratio = 1.0f - (double)current_freed() / (double)full_bytes;
    
    std::cout << (final_ratio <= target_ratio ? "Target ratio reached: " : "Cannot reach target ratio: ")
             << final_ratio << "\n"
             << "comp_delta: " << kv_state->n_cur_comp_chunks - initial_comp_chunks << "\n"
             << "store_delta: " << kv_state->n_cur_store_chunks - initial_store_chunks << "\n"
             << "unload_delta: " << weight->n_cur_unload_layers - initial_unload_layers << "\n";
    
    return std::make_tuple(final_ratio, 
                           kv_state->n_cur_comp_chunks - initial_comp_chunks, 
                           kv_state->n_cur_store_chunks - initial_store_chunks, 
                           weight->n_cur_unload_layers - initial_unload_layers);
}

namespace mzcache {

int decode_chunk_aligned(llama_context * ctx, const llama_batch & batch) {
    if (batch.n_tokens <= 0) {
        return ctx->decode(batch);
    }
    // mz batches are token batches (no embeddings) with contiguous positions.
    GGML_ASSERT(batch.token != nullptr && batch.pos != nullptr);

    int32_t start = 0;
    while (start < batch.n_tokens) {
        // extend until the next position that begins a new 256-token chunk
        int32_t end = start + 1;
        while (end < batch.n_tokens && (batch.pos[end] % TOKENS_PER_CHUNK) != 0) {
            end++;
        }

        llama_batch sub = batch;
        sub.n_tokens = end - start;
        sub.token    = batch.token + start;
        sub.pos      = batch.pos   + start;
        sub.n_seq_id = batch.n_seq_id ? batch.n_seq_id + start : nullptr;
        sub.seq_id   = batch.seq_id   ? batch.seq_id   + start : nullptr;
        sub.logits   = batch.logits   ? batch.logits   + start : nullptr;

        const int ret = ctx->decode(sub);
        if (ret != 0) {
            return ret;
        }
        start = end;
    }
    return 0;
}

} // namespace mzcache

#endif // MZCACHE_SVM_KV_CHUNK
