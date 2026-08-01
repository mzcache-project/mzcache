#include "mzcache_kernels.h"
#include "mzcache_kv_state.h"
#include "mzcache_core.h"
#include "mzcache_globals.h"
#include "cachegen_utils.h"

#include "mzcache_threadpool.h"

#include "ggml-opencl-extra.h"
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>

#include <fcntl.h>      // open, O_RDONLY, O_DIRECT
#include <unistd.h>     // pread, close
#include <sys/types.h>  // off_t
#include <sys/mman.h>       // mlock, munlock (resident swap-out arenas)
#include <sys/resource.h>   // getrlimit/setrlimit (raise RLIMIT_MEMLOCK)

#include <utility>  // std::pair

static timespec get_monotonic_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts;
}

static long diff_microseconds(const timespec &start, const timespec &end) {
    return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
}

float alloc_time = 0;

// Compression arenas hold the resident compressed KV. The arena is read on the
// CPU (NEON) during swap-in decompression, never by the GPU, so it is plain
// malloc by default. The arenas allocated while swapping out
// (mzcache_compress_chunks) stay resident while the app is backgrounded and must
// be kept out of zram, else the next swap-in re-faults them back. Two ways:
//
//   default              malloc, then mlock the used bytes at swap-out end
//                        (mlock_resident_arenas). Needs RLIMIT_MEMLOCK headroom
//                        (root / CAP_IPC_LOCK), raised to unlimited once below.
//   MZCACHE_ARENA_SVM=1  back them with coarse-grain OpenCL SVM instead. SVM
//                        lives on kgsl driver pages the kernel does not reclaim
//                        to zram, so it stays resident with no mlock and no root
//                        — at the cost of eager page commit (~1.7x slower reload
//                        at full offload). Off by default.
//
// Either way, only swap-out arenas are protected. Arenas re-allocated from
// storage during swap-in (mzcache_reload_all_arenas) are freed again almost
// immediately, so they are always plain malloc — never SVM, never mlock'd.
//
// RLIMIT_MEMLOCK defaults to 64 KB on Android, which would fail every arena
// mlock. Raise it to unlimited once (succeeds under root / CAP_SYS_RESOURCE);
// best-effort — if it can't be raised (unprivileged) the mlock just fails and
// the arena stays unlocked (functionally fine, only zram-reclaimable); use
// MZCACHE_ARENA_SVM=1 there instead.
static void raise_memlock_limit_once() {
    static const bool done = [] {
        struct rlimit rl;
        if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
            rl.rlim_cur = rl.rlim_max = RLIM_INFINITY;
            setrlimit(RLIMIT_MEMLOCK, &rl);
        }
        return true;
    }();
    (void) done;
}

// MZCACHE_ARENA_SVM=1 backs the resident swap-out arenas with OpenCL SVM instead
// of malloc+mlock, for environments without the root / CAP_IPC_LOCK that mlock
// needs. Off by default (malloc+mlock). Read once.
static bool arena_svm_enabled() {
    static const bool on = [] {
        const char* e = getenv("MZCACHE_ARENA_SVM");
        return e && e[0] == '1';
    }();
    return on;
}

// allow_svm: swap-out arenas (mzcache_compress_chunks) pass true so they honor
// MZCACHE_ARENA_SVM; reload arenas pass false and are always malloc. The SVM
// path falls back to malloc if clSVMAlloc fails or returns an unaligned pointer.
Arena::Arena(size_t size, bool allow_svm)
    : pool(nullptr), top(0), cap(size), svm(false), locked(false), locked_bytes(0) {
    if (allow_svm && arena_svm_enabled()) {
        pool = (uint8_t*) clSVMAlloc(g_opencl_context, CL_MEM_READ_WRITE, size, 0);
        if (pool && ((uintptr_t) pool & 4095) == 0) { svm = true; return; }
        if (pool) { clSVMFree(g_opencl_context, pool); pool = nullptr; }
        MZ_LOG_ERROR("Arena: clSVMAlloc(%zu) failed/unaligned, using malloc", size);
    }
    int rc = posix_memalign((void**)&pool, 4096, size);
    assert(rc == 0 && pool && "arena posix_memalign failed");
    (void) rc;
}

// Lock the *used* portion [pool, top) into RAM so the resident compressed KV is
// not reclaimed to zram while the app is backgrounded. Called at the end of
// swap-out (mlock_resident_arenas), when top is final, so we lock only the bytes
// actually filled — not the capacity, which for CacheGen (variable-rate, sized
// to the raw upper bound) is heavily over-provisioned. Best-effort: needs
// RLIMIT_MEMLOCK headroom (root); a failure just leaves the arena unlocked. This
// runs on the swap-out path, never the timed swap-in restore.
void Arena::lock_used() {
    if (svm || locked || top == 0) return;  // SVM arenas already resist zram; nothing to mlock
    raise_memlock_limit_once();
    const size_t n = (top + 4095) & ~size_t(4095);
    if (mlock(pool, n) == 0) {
        locked = true;
        locked_bytes = n;
    } else {
        MZ_LOG_ERROR("Arena: mlock(%zu) failed (RLIMIT_MEMLOCK? needs root), staying unlocked", n);
    }
}

Arena::~Arena() {
    if (svm) { clSVMFree(g_opencl_context, pool); return; }
    if (locked) munlock(pool, locked_bytes);
    std::free(pool);
}

#ifdef MZCACHE_SVM_KV_CHUNK
mzcache_kv_state::mzcache_kv_state(mzcache_core* _core, int _num_layers, int _kv_hidden_dim, int _num_chunks_per_tensor, std::string _model_name, int _per_layer_balance)
    : num_layers(_num_layers),
      kv_hidden_dim(_kv_hidden_dim),
      num_chunks_per_tensor(_num_chunks_per_tensor),
      model_name(std::move(_model_name)),
      max_comp_chunks(0),
      max_store_chunks(0),
      core(_core)
{

    thread_pool = core->get_thread_pool();

    // Arena pool size: one layer's worth of chunks at the backend's
    // worst-case compressed size (see mzcache_compress_chunks). SVM commits
    // physical pages eagerly, so pools are allocated on demand in
    // mzcache_compress_chunks instead of up front here.
    {
        const size_t raw_chunk = (size_t) TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16) * 2;
#if defined(MZCACHE_COMPRESS_FLEXGEN_8BIT)
        const size_t comp_bound = raw_chunk * 17 / 32; // raw/2 + mins/maxs meta
#elif defined(MZCACHE_COMPRESS_FLEXGEN)
        const size_t comp_bound = raw_chunk * 9 / 32;  // raw/4 + mins/maxs meta
#else
        const size_t comp_bound = raw_chunk;           // CACHEGEN is variable-rate
#endif
        arena_bytes = ((size_t) num_chunks_per_tensor * comp_bound + 4095) & ~size_t(4095);
    }

    layer_fds.assign(num_layers, -1);
    kv_store_offsets.assign(num_layers, 0);

    // The legacy per_layer_balance table below is keyed by the SoC (chipset),
    // not the phone vendor — see mzcache_profile.h.
    soc = mzcache::classify_soc(mzcache::soc_model());
    std::cout << "SoC: " << mzcache::soc_name(soc) << std::endl;
    std::cout << "Model name: " << model_name << std::endl;

    if (_per_layer_balance > 0) {
        // Derived by mzcache_core from the one-time device profile.
        per_layer_balance = _per_layer_balance;
        std::cout << "per_layer_balance set from device profile: " << per_layer_balance << "\n";
    } else {
    // Legacy hard-coded values (CACHEGEN / NONE builds without profiler support).
    #ifdef MZCACHE_COMPRESS_FLEXGEN

        if(model_name == "EXAONE-4.0-1.2B") {

            if(soc == mzcache::SocClass::SM8750) {
                // per_layer_balance = 175;
                per_layer_balance = 240;
                // n_weight_balance_chunks = 5300; // Other devices: 4 chunks per tensor
            } else {
                per_layer_balance = 140;
                // n_weight_balance_chunks = 2500; // Samsung devices: 8 chunks per tensor
            }
        }

        if(model_name == "Qwen3-0.6B") {
            if(soc == mzcache::SocClass::SM8750) {
                per_layer_balance = 57;
                // n_weight_balance_chunks = 1600; // Samsung devices: 8 chunks per tensor
            } else {
                per_layer_balance = 36;
                // n_weight_balance_chunks = 1000; // Other devices: 4 chunks per tensor
            }
        }

    #elif defined(MZCACHE_COMPRESS_FLEXGEN_8BIT)

        if(model_name == "EXAONE-4.0-1.2B") {

            if(soc == mzcache::SocClass::SM8750) {
                // per_layer_balance = 175;
                per_layer_balance = 240;
                // n_weight_balance_chunks = 5300; // Other devices: 4 chunks per tensor
            } else {
                per_layer_balance = 140;
                // n_weight_balance_chunks = 2500; // Samsung devices: 8 chunks per tensor
            }
        }

        if(model_name == "Qwen3-0.6B") {
            if(soc == mzcache::SocClass::SM8750) {
                per_layer_balance = 57;
                // n_weight_balance_chunks = 1600; // Samsung devices: 8 chunks per tensor
            } else {
                per_layer_balance = 36;
                // n_weight_balance_chunks = 1000; // Other devices: 4 chunks per tensor
            }
        }

    #elif defined(MZCACHE_COMPRESS_CACHEGEN)

        if(model_name == "EXAONE-4.0-1.2B") {

            if(soc == mzcache::SocClass::SM8750) {
                // per_layer_balance = 175;
                per_layer_balance = 40;
                // n_weight_balance_chunks = 5300; // Other devices: 4 chunks per tensor
            } else {
                per_layer_balance = -1; // todo : set this value after profiling
            }
        }

        if(model_name == "Qwen3-0.6B") {
            if(soc == mzcache::SocClass::SM8750) {
                per_layer_balance = 10;
            } else {
                per_layer_balance = -1; // todo : set this value after profiling
            }
        }

    #endif

    }

    // Experiment/auto-config override: value from mz_device_profile (or manual).
    if (const char * e = getenv("MZCACHE_PER_LAYER_BALANCE")) {
        const int v = atoi(e);
        if (v > 0) {
            per_layer_balance = v;
            std::cout << "per_layer_balance overridden via env: " << v << "\n";
        }
    }

    derive_swapout_plan();

    next_layer_to_offload = num_layers - 1;

    for (int i = 0; i < num_layers; ++i) {

        chunk_ptrs * k_chunk_ptrs = g_svm_chunk_ptrs[2*i+0];
        chunk_ptrs * v_chunk_ptrs = g_svm_chunk_ptrs[2*i+1];


        layers[i].layer_idx = i;

#if defined(MZCACHE_COMPRESS_FLEXGEN) || defined(MZCACHE_COMPRESS_FLEXGEN_8BIT)
        // Quantization metadata (mins/maxs) pool, one block per layer instead
        // of 4 small news per chunk. SVM-backed like the arenas: the whole pool
        // is read during swapin decompression, and as malloc'd memory it
        // zram-swaps while the app idles in the background (measured ~114MB of
        // the swapin major-fault storm).
        {
            const size_t meta_elems = (size_t) TOKENS_PER_CHUNK * kv_hidden_dim / 64;
            const size_t pool_bytes = (size_t) num_chunks_per_tensor * 4 * meta_elems * sizeof(__fp16);
            // Plain malloc: the mins/maxs metadata pool is only ~1/32 of the KV;
            // the bulk of the resident compressed footprint is the arenas, which
            // are the ones mlock'd. (Locking this too would need per-pool size
            // tracking for munlock; left as a possible extension.)
            uint8_t * pool = (uint8_t*) malloc(pool_bytes);
            assert(pool && "meta pool malloc failed");
            meta_pool_svm[i] = false;
            meta_pool[i] = pool;
            __fp16 * p = (__fp16*) pool;
            for (int j = 0; j < num_chunks_per_tensor; ++j) {
                layers[i].k_chunks.compressed_chunks[j].mins = p; p += meta_elems;
                layers[i].k_chunks.compressed_chunks[j].maxs = p; p += meta_elems;
                layers[i].v_chunks.compressed_chunks[j].mins = p; p += meta_elems;
                layers[i].v_chunks.compressed_chunks[j].maxs = p; p += meta_elems;
            }
        }
#endif

        for(int j = 0; j < num_chunks_per_tensor; ++j) {

            // This code is more independent to backend implementation. (Can be used with SVM or CUDA Shared memory)
            if (k_chunk_ptrs->c[j] == nullptr || v_chunk_ptrs->c[j] == nullptr) {
                throw std::runtime_error("Chunk tensor is null in the kv_cache.");
            }
            layers[i].k_chunks.raw_chunks[j].c = k_chunk_ptrs->c[j];
            layers[i].v_chunks.raw_chunks[j].c = v_chunk_ptrs->c[j];
        

            layers[i].states[j] = KVstate::Raw; // Initialize all chunks to Raw state

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

            // (mins/maxs pointers were carved from the per-layer meta pool above)

        }

#ifdef MZCACHE_COMPRESS_CACHEGEN
        layers[i].encoder_meta = new EncoderMeta();

        encode_meta_function_new(layers[i], num_chunks_per_tensor * TOKENS_PER_CHUNK, kv_hidden_dim, layers[i].encoder_meta); ;// can change tokens_per_chunk to chunk_bound[i] later
#endif

    }
}

mzcache_kv_state::~mzcache_kv_state() {
    // Clean up arenas
    for (int i = 0; i < num_layers; ++i) {
        for (auto arena : arenas[i]) {
            delete arena;
        }
        arenas[i].clear();
    }

    // Clean up min/max metadata pools
#if defined(MZCACHE_COMPRESS_FLEXGEN) || defined(MZCACHE_COMPRESS_FLEXGEN_8BIT)
    for (int i = 0; i < num_layers; ++i) {
        for (auto & [pool, svm] : grown_meta_pools[i]) {
#ifdef MZCACHE_USE_OPENCL
            if (svm) { clSVMFree(g_opencl_context, pool); continue; }
#endif
            free(pool);
        }
        grown_meta_pools[i].clear();

        if (!meta_pool[i]) continue;
#ifdef MZCACHE_USE_OPENCL
        if (meta_pool_svm[i]) {
            clSVMFree(g_opencl_context, meta_pool[i]);
            meta_pool[i] = nullptr;
            continue;
        }
#endif
        free(meta_pool[i]);
        meta_pool[i] = nullptr;
    }
#elif defined(MZCACHE_COMPRESS_CACHEGEN)
    for (int i = 0; i < num_layers; ++i) {
        delete layers[i].encoder_meta;
    }
#endif

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


    // delete KV files
    // std::string fn = "KV_" + model_name + "_L" + std::to_string(temp_layer_idx_to_store) + ".bin";
    for (int i = 0; i < num_layers; ++i) {
        // Close the file descriptor first
        if (layer_fds[i] >= 0) {
            ::close(layer_fds[i]);
            layer_fds[i] = -1;
        }
        
        // Delete the file
        std::string fn = "KV_" + model_name + "_L" + std::to_string(i) + ".bin";
        if (::unlink(fn.c_str()) != 0) {
            // Ignore a missing file; report only other errors
            if (errno != ENOENT) {
                perror(("Failed to delete " + fn).c_str());
            }
        } else {
            printf("Deleted file: %s\n", fn.c_str());
        }
    }
}

void mzcache_kv_state::derive_swapout_plan() {
    // Per-layer split between compressible chunks (kept in the arenas) and
    // store-to-disk chunks, plus the descending swapout cursors. Everything
    // here is a pure function of num_chunks_per_tensor, so runtime chunk
    // growth re-derives the plan by calling this again (from a clean ladder).
    default_chunk_bound = static_cast<int>(num_chunks_per_tensor * core->get_decomp_load_ratio());

    const int n_weight_balance_chunks = per_layer_balance * num_layers;

    int cur_chunks = 0;
    bool balance_chunks = true;
    for (int i = 0; i < num_layers; ++i) {
        if(balance_chunks) {
            if(cur_chunks + num_chunks_per_tensor > n_weight_balance_chunks) {
                int comp_only = (n_weight_balance_chunks - cur_chunks);
                int n_remain = (num_chunks_per_tensor - comp_only);
                chunk_bound[i] = static_cast<int>(comp_only + n_remain * core->get_decomp_load_ratio());
                balance_chunks = false;
            } else {
                chunk_bound[i] = num_chunks_per_tensor;
                cur_chunks += num_chunks_per_tensor;
            }
        }
        else {
            chunk_bound[i] = default_chunk_bound;
        }
    }

    max_comp_chunks  = 0;
    max_store_chunks = 0;
    for(int i = 0; i < num_layers; ++i) {
        max_comp_chunks += chunk_bound[i];
        max_store_chunks += num_chunks_per_tensor - chunk_bound[i];
    }

    cur_layer_idx_to_compress = num_layers - 1;
    cur_layer_idx_to_store = num_layers - 1; // current layer index to store

    cur_chunk_idx_to_compress = chunk_bound[num_layers - 1] - 1; // current chunk index to compress
    cur_chunk_idx_to_store = num_chunks_per_tensor - 1;   // current chunk index to store
}

void mzcache_kv_state::register_grown_chunks(int n_chunks_total) {
    if (n_chunks_total <= num_chunks_per_tensor) {
        return;
    }
    if (n_chunks_total > MAX_CHUNKS_PER_TENSOR) {
        MZ_LOG_ERROR("register_grown_chunks: %d exceeds MAX_CHUNKS_PER_TENSOR", n_chunks_total);
        return;
    }

    // The swapout plan (chunk_bound, max counters, descending cursors) can
    // only be re-derived from a clean ladder; mid-ladder growth is parked and
    // applied when the next swapout starts (apply_pending_growth).
    const bool clean_ladder = (n_cur_comp_chunks == 0 && n_cur_store_chunks == 0 &&
                               n_cur_arena_offload_bytes == 0);
    if (!clean_ladder) {
        pending_chunks_total = std::max(pending_chunks_total, n_chunks_total);
        MZ_LOG_WARN("register_grown_chunks: swapout ladder in flight; growth to "
                    + std::to_string(n_chunks_total) + " chunks deferred to the next cycle");
        return;
    }

    const int old_count = num_chunks_per_tensor;
    const size_t meta_elems = (size_t) TOKENS_PER_CHUNK * kv_hidden_dim / 64;

    for (int i = 0; i < num_layers; ++i) {
        chunk_ptrs * k_chunk_ptrs = g_svm_chunk_ptrs[2*i+0];
        chunk_ptrs * v_chunk_ptrs = g_svm_chunk_ptrs[2*i+1];

#if defined(MZCACHE_COMPRESS_FLEXGEN) || defined(MZCACHE_COMPRESS_FLEXGEN_8BIT)
        // mins/maxs block for the new chunk range, laid out like the ctor's
        // per-layer pool (4 arrays per chunk), allocator matching the pools.
        {
            const size_t pool_bytes = (size_t)(n_chunks_total - old_count) * 4 * meta_elems * sizeof(__fp16);
            uint8_t * pool = (uint8_t*) malloc(pool_bytes);  // plain malloc, like the initial meta pool
            assert(pool && "grown meta pool malloc failed");
            const bool svm = false;
            grown_meta_pools[i].emplace_back(pool, svm);

            __fp16 * p = (__fp16*) pool;
            for (int j = old_count; j < n_chunks_total; ++j) {
                layers[i].k_chunks.compressed_chunks[j].mins = p; p += meta_elems;
                layers[i].k_chunks.compressed_chunks[j].maxs = p; p += meta_elems;
                layers[i].v_chunks.compressed_chunks[j].mins = p; p += meta_elems;
                layers[i].v_chunks.compressed_chunks[j].maxs = p; p += meta_elems;
            }
        }
#endif

        for (int j = old_count; j < n_chunks_total; ++j) {
            if (k_chunk_ptrs->c[j] == nullptr || v_chunk_ptrs->c[j] == nullptr) {
                MZ_LOG_ERROR("register_grown_chunks: chunk %d of layer %d is not resident", j, i);
                return;
            }
            layers[i].k_chunks.raw_chunks[j].c = k_chunk_ptrs->c[j];
            layers[i].v_chunks.raw_chunks[j].c = v_chunk_ptrs->c[j];
            layers[i].states[j] = KVstate::Raw;
        }
    }

    num_chunks_per_tensor = n_chunks_total;
    derive_swapout_plan();

    MZ_LOG_INFO("register_grown_chunks: %d -> %d chunks per tensor "
                "(max_comp %d, max_store %d)",
                old_count, n_chunks_total, max_comp_chunks, max_store_chunks);
}

void mzcache_kv_state::apply_pending_growth() {
    if (pending_chunks_total > num_chunks_per_tensor) {
        const int want = pending_chunks_total;
        pending_chunks_total = 0;
        register_grown_chunks(want);
    } else {
        pending_chunks_total = 0;
    }
}

void mzcache_kv_state::kv_alloc_worker(int layer, int chunk, std::shared_ptr<std::promise<void>> alloc_prom) {
    // The promise object arrives as a shared_ptr
    try {
        // The actual SVM allocation
        __fp16* new_k = (__fp16*)clSVMAlloc(
            g_opencl_context,
            CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
            TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16) * 2, 0);
        if (!new_k) {
            throw std::runtime_error("SVM alloc failed");
        }
        __fp16* new_v = new_k + TOKENS_PER_CHUNK * kv_hidden_dim;

        // Register the pointers in the metadata
        layers[layer].k_chunks.raw_chunks[chunk].c = new_k;
        g_svm_chunk_ptrs[2*layer + 0]->c[chunk]   = new_k;
        layers[layer].v_chunks.raw_chunks[chunk].c = new_v;
        g_svm_chunk_ptrs[2*layer + 1]->c[chunk]   = new_v;

        // Fulfill the promise on success
        alloc_prom->set_value();
    }
    catch (...) {
        // On an exception, pass it into the promise so the future side does not hang
        alloc_prom->set_exception(std::current_exception());
        // Propagate to the caller as well if needed
        throw;
    }
}

void mzcache_kv_state::kv_read_worker(int layer, int chunk) {
    // std::cout << "kv_read_worker: layer = " << layer << ", chunk = " << chunk << std::endl;
    core->wait_for_kv_allocation(layer, chunk);

    const auto &kc = layers[layer].k_chunks.stored_chunks[chunk];
    const auto &vc = layers[layer].v_chunks.stored_chunks[chunk];
    off_t off_k = kc.file_offset;
    off_t off_v = vc.file_offset;

    __fp16* new_k = (__fp16*)g_svm_chunk_ptrs[2*layer + 0]->c[chunk];
    __fp16* new_v = (__fp16*)g_svm_chunk_ptrs[2*layer + 1]->c[chunk];

    size_t raw_bytes = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16);

    if ((uint8_t*)new_v == (uint8_t*)new_k + raw_bytes &&
        off_v == off_k + raw_bytes) {
        ssize_t n = pread(layer_fds[layer], new_k, raw_bytes*2, off_k);
        if (n != (ssize_t)(raw_bytes*2)) perror("pread contiguous");
    } else {
        ssize_t n = pread(layer_fds[layer], new_k, raw_bytes, off_k);
        if (n != (ssize_t)raw_bytes) perror("pread k");
        n = pread(layer_fds[layer], new_v, raw_bytes, off_v);
        if (n != (ssize_t)raw_bytes) perror("pread v");
    }

    layers[layer].states[chunk] = KVstate::Raw;

    if(core->layer_sync.size() == num_layers) {
        core->layer_sync[layer]->taskDone();
    }
}

void mzcache_kv_state::free_layer_arenas(int layer) {
    // Runs on the FREE pool thread (see kv_decomp_worker), off the layer-sync
    // critical path. The layer is fully decompressed here, so no decomp worker
    // still reads arenas[layer]; clean_after_decompress runs later (after
    // wait_idle_all drains FREE) and finds the vector empty.
    for (Arena* a : arenas[layer]) {
        delete a;  // Arena dtor munlocks (if locked) and frees
    }
    arenas[layer].clear();
}

// mlock the used portion of every resident compression arena. Called at the end
// of swap-out, when each arena's top is final, so the resident compressed KV
// stays out of zram while the app is backgrounded. Idempotent (lock_used no-ops
// an already-locked arena); the arenas re-allocated during swap-in reload are
// never passed here, so they stay plain malloc and reclaimable.
void mzcache_kv_state::mlock_resident_arenas() {
    for (int layer = 0; layer < num_layers; ++layer) {
        for (Arena* a : arenas[layer]) {
            a->lock_used();
        }
    }
}

void mzcache_kv_state::kv_decomp_worker(int layer, int chunk) {
    core->wait_for_kv_allocation(layer, chunk);

    auto &Kcc = layers[layer].k_chunks.compressed_chunks[chunk];
    auto &Vcc = layers[layer].v_chunks.compressed_chunks[chunk];
    
    // Use this layer's arena
    Arena *cur = arenas[layer][Kcc.stack_id];

    __fp16* new_k = (__fp16*)g_svm_chunk_ptrs[2*layer + 0]->c[chunk];
    __fp16* new_v = (__fp16*)g_svm_chunk_ptrs[2*layer + 1]->c[chunk];

    uint8_t* comp_k = cur->pool + Kcc.offset;
    uint8_t* comp_v = cur->pool + Vcc.offset;

#if defined(MZCACHE_COMPRESS_FLEXGEN)
    flexgen_decompress_single_thread(
        comp_k, kv_hidden_dim, new_k,
        Kcc.mins, Kcc.maxs);
    flexgen_decompress_single_thread(
        comp_v, kv_hidden_dim, new_v,
        Vcc.mins, Vcc.maxs);

#elif defined(MZCACHE_COMPRESS_FLEXGEN_8BIT)
    flexgen_decompress_8bit(
        comp_k, kv_hidden_dim, new_k,
        Kcc.mins, Kcc.maxs);
    flexgen_decompress_8bit(
        comp_v, kv_hidden_dim, new_v,
        Vcc.mins, Vcc.maxs);

#elif defined(MZCACHE_COMPRESS_CACHEGEN)
    decode_function_new(
        comp_k, kv_hidden_dim, new_k,
        layers[layer].k_chunks.compressed_chunks[chunk].lengths_arr,
        true,
        layers[layer].encoder_meta->cdf_key,
        &layers[layer].encoder_meta->max_key[chunk * TOKENS_PER_CHUNK],
        layers[layer].k_chunks.compressed_chunks[chunk].byte_sums_arr
    );
    decode_function_new(
        comp_v, kv_hidden_dim, new_v,
        layers[layer].v_chunks.compressed_chunks[chunk].lengths_arr,
        false,
        layers[layer].encoder_meta->cdf_value,
        &layers[layer].encoder_meta->max_value[chunk * TOKENS_PER_CHUNK],
        layers[layer].v_chunks.compressed_chunks[chunk].byte_sums_arr
    );
#endif

    layers[layer].states[chunk] = KVstate::Raw;

    // Progressive arena release, off the decomp critical path: when this layer's
    // last compressed chunk is decompressed its arenas are dead weight. Hand the
    // clSVMFree to the FREE pool group so it never delays the taskDone() the GPU
    // layer-sync waits on. comp_left is seeded by initialize_layer_sync; flows
    // that leave it 0 fall back to clean_after_decompress. finish_swapin_cycle's
    // wait_idle_all() drains FREE before clean_after_decompress runs.
    if (comp_left[layer].fetch_sub(1, std::memory_order_acq_rel) == 1) {
        thread_pool->enqueue(CoreType::FREE,
            std::bind(&mzcache_kv_state::free_layer_arenas, this, layer));
    }

    if(core->layer_sync.size() == num_layers) {
        core->layer_sync[layer]->taskDone();
    }
}

// Compress should be done sequentially, so that the compressed chunks are stored in order.
int64_t mzcache_kv_state::mzcache_compress_chunks(int num_chunks) {
    int64_t total_compressed_size = 0;
    int num_threads = core->get_core_configs()[CoreType::DECOMP].size();

    for (int iter = 0; iter < num_chunks; ++iter) {
        // Advance past any layer whose compress region is exhausted (cursor < 0),
        // so we never read raw_chunks[-1]. Guards re-entry where a previous call
        // finished a layer and left the cursor at -1.
        while (cur_layer_idx_to_compress >= 0 && cur_chunk_idx_to_compress < 0) {
            cur_layer_idx_to_compress--;
            if (cur_layer_idx_to_compress >= 0)
                cur_chunk_idx_to_compress = chunk_bound[cur_layer_idx_to_compress] - 1;
        }
        if (cur_layer_idx_to_compress < 0) break;  // all compress regions done

        int L = cur_layer_idx_to_compress;
        int C = cur_chunk_idx_to_compress;
        const int64_t iter_prev_total = total_compressed_size;
        
        __fp16* key_data_ptr = (__fp16 *)layers[L].k_chunks.raw_chunks[C].c;
        __fp16* value_data_ptr = (__fp16 *)layers[L].v_chunks.raw_chunks[C].c;

        // Take the current layer's last arena (allocate on-demand if none).
        // allow_svm=true: these swap-out arenas stay resident while the app is
        // backgrounded, so they honor MZCACHE_ARENA_SVM (else malloc + mlock).
        if (arenas[L].empty()) {
            arenas[L].push_back(new Arena(arena_bytes, /*allow_svm=*/true));
        }
        Arena* cur = arenas[L].back();
        size_t kv_raw_size = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16) * 2;

        // Create a new arena when the current layer's arena is out of space
        if (cur->top + kv_raw_size > cur->cap) {
            cur = new Arena(arena_bytes, /*allow_svm=*/true);
            arenas[L].push_back(cur);
        }

        uint8_t * key_out_ptr = cur->pool + cur->top;

        size_t key_comp_size;
        size_t value_comp_size;

#if defined(MZCACHE_COMPRESS_FLEXGEN)
        flexgen_compress(
            key_data_ptr, 
            kv_hidden_dim, 
            key_out_ptr, 
            layers[L].k_chunks.compressed_chunks[C].mins, 
            layers[L].k_chunks.compressed_chunks[C].maxs,
            thread_pool,
            num_threads
        );
        key_comp_size = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16) / 4;

        uint8_t * value_out_ptr = cur->pool + cur->top + key_comp_size;

        flexgen_compress(
            value_data_ptr, 
            kv_hidden_dim, 
            value_out_ptr, 
            layers[L].v_chunks.compressed_chunks[C].mins, 
            layers[L].v_chunks.compressed_chunks[C].maxs,
            thread_pool,
            num_threads
        );
        value_comp_size = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16) / 4;

        total_compressed_size += (key_comp_size + value_comp_size + 2 * TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(uint16_t) / 32);

#elif defined(MZCACHE_COMPRESS_FLEXGEN_8BIT)
        flexgen_compress_8bit(
            key_data_ptr, 
            kv_hidden_dim, 
            key_out_ptr, 
            layers[L].k_chunks.compressed_chunks[C].mins, 
            layers[L].k_chunks.compressed_chunks[C].maxs,
            thread_pool,
            num_threads
        );
        key_comp_size = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16) / 2;

        uint8_t * value_out_ptr = cur->pool + cur->top + key_comp_size;

        flexgen_compress_8bit(
            value_data_ptr, 
            kv_hidden_dim, 
            value_out_ptr, 
            layers[L].v_chunks.compressed_chunks[C].mins, 
            layers[L].v_chunks.compressed_chunks[C].maxs,
            thread_pool,
            num_threads
        );
        value_comp_size = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16) / 2;

        total_compressed_size += (key_comp_size + value_comp_size + 2 * TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(uint16_t) / 32);

#elif defined(MZCACHE_COMPRESS_CACHEGEN)
        key_comp_size = encode_function_new(
            key_data_ptr, 
            kv_hidden_dim, 
            key_out_ptr,
            layers[L].k_chunks.compressed_chunks[C].lengths_arr,
            true,
            layers[L].encoder_meta->cdf_key,
            layers[L].k_chunks.compressed_chunks[C].byte_sums_arr
        );

        uint8_t * value_out_ptr = cur->pool + cur->top + key_comp_size;

        value_comp_size = encode_function_new(
            value_data_ptr, 
            kv_hidden_dim, 
            value_out_ptr,
            layers[L].v_chunks.compressed_chunks[C].lengths_arr,
            false,
            layers[L].encoder_meta->cdf_value,
            layers[L].v_chunks.compressed_chunks[C].byte_sums_arr
        );

        total_compressed_size += key_comp_size + value_comp_size +
            2 * (TOKENS_PER_CHUNK / MAX_TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(uint8_t) + TOKENS_PER_CHUNK / MAX_TOKENS_PER_CHUNK * sizeof(uint16_t));
#endif

        size_t kv_compressed_size = key_comp_size + value_comp_size;

        // Record the current layer's arena index
        layers[L].k_chunks.compressed_chunks[C].stack_id = arenas[L].size() - 1;
        layers[L].v_chunks.compressed_chunks[C].stack_id = arenas[L].size() - 1;

        layers[L].k_chunks.compressed_chunks[C].offset = cur->top;
        layers[L].v_chunks.compressed_chunks[C].offset = cur->top + key_comp_size;

        layers[L].k_chunks.compressed_chunks[C].comp_size = key_comp_size;
        layers[L].v_chunks.compressed_chunks[C].comp_size = value_comp_size;

        layers[L].states[C] = KVstate::Compressed;

#ifdef MZCACHE_USE_OPENCL
        if(value_data_ptr == key_data_ptr + TOKENS_PER_CHUNK * kv_hidden_dim) {
            clSVMFree(g_opencl_context, key_data_ptr);
        } else {
            clSVMFree(g_opencl_context, key_data_ptr);
            clSVMFree(g_opencl_context, value_data_ptr);
        }

        g_svm_chunk_ptrs[2 * L + 0]->c[C] = nullptr;
        g_svm_chunk_ptrs[2 * L + 1]->c[C] = nullptr;
#endif

        cur->top += kv_compressed_size;
        // Advance within the current layer; the top-of-loop guard moves to the
        // next compress region once this layer is done.
        cur_chunk_idx_to_compress--;
        n_cur_comp_chunks += 1;
        n_cur_comp_saved_bytes += (int64_t)kv_raw_size - (total_compressed_size - iter_prev_total);
    }

    return total_compressed_size;
}
void mzcache_kv_state::mzcache_decompress_chunks(int num_chunks) {
    // const int num_threads     = core_configs[CoreType::DECOMP_KERNEL].size();

    // (1) enqueue the alloc & decompress tasks
    MZ_TIME_START(enqueue_start)
    for (int iter = 0; iter < num_chunks; ++iter) {
        // ── pick the next chunk index ──
        cur_chunk_idx_to_compress++;
        if (cur_chunk_idx_to_compress >= chunk_bound[cur_layer_idx_to_compress]) {
            cur_chunk_idx_to_compress = 0;
            cur_layer_idx_to_compress++;
            if (cur_layer_idx_to_compress >= num_layers) {
                fprintf(stderr, "Error: Cannot decompress more chunks than available.\n");
                return;
            }
        }
        int layer_idx = cur_layer_idx_to_compress;
        int chunk_idx = cur_chunk_idx_to_compress;

        // ── copy the metadata ──
        auto Kcc       = layers[layer_idx].k_chunks.compressed_chunks[chunk_idx];
        auto Vcc       = layers[layer_idx].v_chunks.compressed_chunks[chunk_idx];

        // ── alloc task (SVM) ──
        auto alloc_prom = std::make_shared<std::promise<void>>();
        core->register_kv_allocation(layer_idx, chunk_idx, alloc_prom); 

        thread_pool->enqueue(
            CoreType::ALLOC,
            std::bind(&mzcache_kv_state::kv_alloc_worker,
                    this, layer_idx, chunk_idx, alloc_prom)
        );
        
        thread_pool->enqueue(
            CoreType::DECOMP,
            std::bind(&mzcache_kv_state::kv_decomp_worker,
                    this, layer_idx, chunk_idx)
        );
    }
    MZ_TIME_END(enqueue_start);

    thread_pool->wait_idle(CoreType::DECOMP);

    clean_after_decompress(num_chunks);
}

std::pair<double,double> mzcache_kv_state::mzcache_decompress_chunks_profile(int num_chunks) {
    // DECOMP number of threads
    // const int num_threads     = core_configs[CoreType::DECOMP_KERNEL].size();

    // Local copies
    int local_layer = cur_layer_idx_to_compress;
    int local_chunk = cur_chunk_idx_to_compress;
    auto alloc_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_chunks; ++iter) {
        // advance local indices
        local_chunk++;
        if (local_chunk >= chunk_bound[local_layer]) {
            local_chunk = 0;
            local_layer++;
            if (local_layer >= num_layers) {
                fprintf(stderr, "Error: too many chunks in sequential.\n");
                return {0.0, 0.0};
            }
        }

        int layer_idx = local_layer;
        int chunk_idx = local_chunk;

        // copy the metadata
        auto Kcc = layers[layer_idx].k_chunks.compressed_chunks[chunk_idx];
        auto Vcc = layers[layer_idx].v_chunks.compressed_chunks[chunk_idx];

        auto alloc_prom = std::make_shared<std::promise<void>>();
        core->register_kv_allocation(layer_idx, chunk_idx, alloc_prom); 

        thread_pool->enqueue(
            CoreType::ALLOC,
            std::bind(&mzcache_kv_state::kv_alloc_worker,
                    this, layer_idx, chunk_idx, alloc_prom)
        );

    }

    local_layer = cur_layer_idx_to_compress;
    local_chunk = cur_chunk_idx_to_compress;

    for (int iter = 0; iter < num_chunks; ++iter) {
        // advance local indices
        local_chunk++;
        if (local_chunk >= chunk_bound[local_layer]) {
            local_chunk = 0;
            local_layer++;
            if (local_layer >= num_layers) {
                fprintf(stderr, "Error: too many chunks in sequential.\n");
                return {0.0, 0.0};
            }
        }

        int layer_idx = local_layer;
        int chunk_idx = local_chunk;

        core->wait_for_kv_allocation(layer_idx, chunk_idx);
    }
  

    auto alloc_end = std::chrono::high_resolution_clock::now();
    double alloc_ms =
        std::chrono::duration<double, std::milli>(alloc_end - alloc_start).count();
    auto decomp_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_chunks; ++iter) {

        cur_chunk_idx_to_compress++;
        if (cur_chunk_idx_to_compress >= chunk_bound[cur_layer_idx_to_compress]) {
            cur_chunk_idx_to_compress = 0;
            cur_layer_idx_to_compress++;
            if (cur_layer_idx_to_compress >= num_layers) {
                // should not happen
                fprintf(stderr, "Error: Cannot decompress more chunks than available.\n");
                break;
            }
        }

        int layer_idx = cur_layer_idx_to_compress;
        int chunk_idx = cur_chunk_idx_to_compress;

        thread_pool->enqueue(
            CoreType::DECOMP,
            std::bind(&mzcache_kv_state::kv_decomp_worker,
                    this, layer_idx, chunk_idx)
        );
    }
    thread_pool->wait_idle(CoreType::DECOMP);
    
    auto decomp_end = std::chrono::high_resolution_clock::now();
    double decomp_ms =
        std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();


    clean_after_decompress(num_chunks);

    return {alloc_ms, decomp_ms};
}

void mzcache_kv_state::mzcache_store_chunks(int num_chunks) {

    const size_t raw_chunk_bytes = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16); // e.g. 256K * 2 = 512KB

    for (int iter = 0; iter < num_chunks; ++iter) {
        // Advance past any layer whose store region is exhausted: chunks below
        // chunk_bound belong to the compress region, so storing them would give a
        // negative file offset (chunk_idx0 < 0) and crash. This also guards
        // re-entry, where a previous call left the cursor at a layer boundary.
        while (cur_layer_idx_to_store >= 0 &&
               cur_chunk_idx_to_store < chunk_bound[cur_layer_idx_to_store]) {
            if (layer_fds[cur_layer_idx_to_store] >= 0) {
                ::close(layer_fds[cur_layer_idx_to_store]);
                layer_fds[cur_layer_idx_to_store] = -1;
            }
            cur_layer_idx_to_store--;
            cur_chunk_idx_to_store = num_chunks_per_tensor - 1;
        }
        if (cur_layer_idx_to_store < 0) break;  // all store regions exhausted

        int L = cur_layer_idx_to_store;
        int C = cur_chunk_idx_to_store;

        // Open file only if it's not already open.
        // No O_TRUNC: chunks are written at absolute offsets, and consecutive
        // swapouts may reopen a layer file that already holds stored chunks
        // (swapout closes fds on every exit path since 3a92595f).
        if (layer_fds[L] < 0) {
            std::string fn = "KV_" + model_name + "_L" + std::to_string(L) + ".bin";
            int fd = ::open(fn.c_str(),
                            O_WRONLY | O_CREAT | O_DIRECT,
                            0666);
            if (fd < 0) {
                perror(("open " + fn).c_str());
                return;
            }
            layer_fds[L] = fd;
        }
        int fd = layer_fds[L];

        off_t chunk_idx0    = static_cast<off_t>(C - chunk_bound[L]);
        off_t offset_key    = chunk_idx0 * (raw_chunk_bytes * 2);
        off_t offset_value  = offset_key + raw_chunk_bytes;

    
        // (3) fetch the raw pointers
        __fp16* key_ptr   = (__fp16*)layers[L]
                            .k_chunks.raw_chunks[C].c;
        __fp16* value_ptr = (__fp16*)layers[L]
                            .v_chunks.raw_chunks[C].c;

        {
            ssize_t n = pwrite(fd, key_ptr, raw_chunk_bytes, offset_key);
            if (n != (ssize_t)raw_chunk_bytes) {
                fprintf(stderr, "pwrite key L=%d C=%d off=%ld: %s\n",
                        L, C, (long)offset_key, strerror(errno));
            }
            // record the metadata
            auto& kc = layers[L].k_chunks.stored_chunks[C];
            kc.file_offset = offset_key;
            kc.stored_size = raw_chunk_bytes;
        }
        {
            ssize_t n = pwrite(fd, value_ptr, raw_chunk_bytes, offset_value);
            if (n != (ssize_t)raw_chunk_bytes) {
                fprintf(stderr, "pwrite value L=%d C=%d off=%ld: %s\n",
                        L, C, (long)offset_value, strerror(errno));
            }
            // record the metadata
            auto& vc = layers[L].v_chunks.stored_chunks[C];
            vc.file_offset = offset_value;
            vc.stored_size = raw_chunk_bytes;
        }

        // update the state
        layers[L].states[C] = KVstate::Stored;

        // (5) free the OpenCL SVM
#ifdef MZCACHE_USE_OPENCL
        if(value_ptr == key_ptr + TOKENS_PER_CHUNK * kv_hidden_dim) {
            // If key and value are contiguous, free only once
            // std::cout << "value_ptr == key_ptr + TOKENS_PER_CHUNK * kv_hidden_dim, freeing only key_ptr." << std::endl;
            clSVMFree(g_opencl_context, key_ptr);
        } else {
            // std::cout << "value_ptr != key_ptr + TOKENS_PER_CHUNK * kv_hidden_dim, freeing both." << std::endl;
            clSVMFree(g_opencl_context, key_ptr);
            clSVMFree(g_opencl_context, value_ptr);
        }

        g_svm_chunk_ptrs[2 * L + 0]->c[C] = nullptr;
        g_svm_chunk_ptrs[2 * L + 1]->c[C] = nullptr;
#endif

        // Advance within the current layer; the top-of-loop guard moves to the
        // next layer (and closes this fd) once this layer's store region is done.
        cur_chunk_idx_to_store--;
        n_cur_store_chunks++;
    }

    // // (7) safely close any remaining file streams
    // for (int l = 0; l < num_layers; ++l) {
    //     if (layer_fds[l] >= 0) { ::close(layer_fds[l]); layer_fds[l] = -1; }
    // }
}

void mzcache_kv_state::mzcache_close_stored_files() {
    for (int l = 0; l < num_layers; ++l) {
        if (layer_fds[l] >= 0) {
            ::close(layer_fds[l]);
            layer_fds[l] = -1;
        }
    }
}

std::pair<double,double> mzcache_kv_state::mzcache_load_chunks_profile(int num_chunks)
{
    const size_t raw_bytes   = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16);
    const size_t alloc_bytes = raw_bytes * 2;                 // key + value

    int prev_layer_idx_to_store = cur_layer_idx_to_store;
    auto alloc_start = std::chrono::high_resolution_clock::now();
    /* 1) batch-enqueue only the ALLOC tasks first ───────────── */
    

    int tmp_layer = cur_layer_idx_to_store;
    int tmp_chunk = cur_chunk_idx_to_store;

    for (int i = 0; i < num_chunks; ++i) {

        // next (L,C)
        tmp_chunk++;
        if (tmp_chunk == num_chunks_per_tensor) {
            tmp_chunk  = chunk_bound[tmp_layer];
            tmp_layer++;
            if (tmp_layer >= num_layers) {
                fprintf(stderr, "Error: too many chunks.\n"); return {0.0, 0.0};
            }
        }
        int L = tmp_layer, C = tmp_chunk;

        /* register the alloc promise */
        auto alloc_prom = std::make_shared<std::promise<void>>();
        core->register_kv_allocation(L, C, alloc_prom);

        // (A) ALLOC task
        thread_pool->enqueue(
            CoreType::ALLOC,
            std::bind(&mzcache_kv_state::kv_alloc_worker,
                    this, L, C, alloc_prom)
        );
    }

    /* wait for the alloc group to finish (guarantees memory is reserved) */

    auto alloc_end = std::chrono::high_resolution_clock::now();
    double alloc_ms =
        std::chrono::duration<double, std::milli>(alloc_end - alloc_start).count();

    int prev_cur_layer_idx = cur_layer_idx_to_store;
    int prev_cur_chunk_idx_to_store = cur_chunk_idx_to_store;

    /* 2) sequentially enqueue the READ tasks ────────────────── */
    auto t_store_start = std::chrono::high_resolution_clock::now();

    if(cur_chunk_idx_to_store < num_chunks_per_tensor - 1) {
        // If we are at the last chunk of the current layer, we need to open the file
        std::string fn = "KV_" + model_name + "_L" + std::to_string(cur_layer_idx_to_store) + ".bin";
        int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) throw std::runtime_error("open failed for " + fn);
        layer_fds[cur_layer_idx_to_store] = fd;
    }

    for (int i = 0; i < num_chunks; ++i) {

        // advance index (same rule)
        cur_chunk_idx_to_store++;
        if (cur_chunk_idx_to_store == num_chunks_per_tensor) {
            cur_layer_idx_to_store++;
            if (cur_layer_idx_to_store >= num_layers) break;

            cur_chunk_idx_to_store = chunk_bound[cur_layer_idx_to_store];

            if (layer_fds[cur_layer_idx_to_store] < 0) {
                std::string fn = "KV_" + model_name + "_L" + std::to_string(cur_layer_idx_to_store) + ".bin";
                int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
                if (fd < 0) throw std::runtime_error("open failed for " + fn);
                layer_fds[cur_layer_idx_to_store] = fd;
            }
        }
        int L = cur_layer_idx_to_store, C = cur_chunk_idx_to_store;

        thread_pool->enqueue(
            CoreType::READ,
            std::bind(&mzcache_kv_state::kv_read_worker, this, L, C)
        );
    }

    thread_pool->wait_idle(CoreType::READ);       // all loads finished
    auto t_store_end = std::chrono::high_resolution_clock::now();
    double store_ms =
        std::chrono::duration<double, std::milli>(t_store_end - t_store_start).count();
    
    /* 3) FD close (only the range we opened) */
    clean_after_read(num_chunks);

    return {alloc_ms, store_ms};
}

void mzcache_kv_state::mzcache_load_chunks(int num_chunks)
{
    const size_t raw_bytes   = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16);
    const size_t alloc_bytes = raw_bytes * 2;   // K + V

    int prev_layer_idx_to_store = cur_layer_idx_to_store;

    if(cur_chunk_idx_to_store < num_chunks_per_tensor - 1) {
        // If we are at the last chunk of the current layer, we need to open the file
        std::string fn = "KV_" + model_name + "_L" + std::to_string(cur_layer_idx_to_store) + ".bin";
        int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) throw std::runtime_error("open failed for " + fn);
        layer_fds[cur_layer_idx_to_store] = fd;
    }

    /* ── 1) enqueue: ALLOC + READ  ─────────────────────────────── */
    for (int iter = 0; iter < num_chunks; ++iter) {

        cur_chunk_idx_to_store++;
        if (cur_chunk_idx_to_store == num_chunks_per_tensor) {
            cur_layer_idx_to_store++;
            
            if (cur_layer_idx_to_store >= num_layers) break;

            cur_chunk_idx_to_store  = chunk_bound[cur_layer_idx_to_store];
            
            if (layer_fds[cur_layer_idx_to_store] < 0) {
                std::string fn = "KV_" + model_name + "_L" + std::to_string(cur_layer_idx_to_store) + ".bin";
                int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
                if (fd < 0) throw std::runtime_error("open failed for " + fn);
                layer_fds[cur_layer_idx_to_store] = fd;
            }
        }
        int L = cur_layer_idx_to_store, C = cur_chunk_idx_to_store;

        /* ── (A) ALLOC task ───────────────────────────────────── */
        auto alloc_prom = std::make_shared<std::promise<void>>();
        core->register_kv_allocation(L, C, alloc_prom);

        // (A) ALLOC task
        thread_pool->enqueue(
            CoreType::ALLOC,
            std::bind(&mzcache_kv_state::kv_alloc_worker,
                    this, L, C, alloc_prom)
        );

        thread_pool->enqueue(
            CoreType::READ,
            std::bind(&mzcache_kv_state::kv_read_worker, this, L, C)
        );
    }

    /* ── 2) wait for the READ group to finish ────────────────── */
    thread_pool->wait_idle(CoreType::READ);

    /* ── 3) close FDs (only the needed layers were opened, so just close them) ─── */
    clean_after_read(num_chunks);
}

void mzcache_kv_state::mzcache_enqueue_decomp_load_chunks(int decomp_num_chunks, int load_num_chunks) {

    // ── 1) LOAD: alloc + file read ──────────────────────────────────
    // Enqueued BEFORE the decomp tasks: their alloc jobs are ungated, so
    // putting them first in the FIFO ALLOC queue keeps the paced decomp
    // allocs (see kv_alloc_worker) from starving the disk reads.
    // open the needed FDs (same logic as mzcache_load_chunks)

    if(cur_chunk_idx_to_store < num_chunks_per_tensor - 1) {
        // If we are at the last chunk of the current layer, we need to open the file
        std::string fn = "KV_" + model_name + "_L" + std::to_string(cur_layer_idx_to_store) + ".bin";
        int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) throw std::runtime_error("open failed for " + fn);
            layer_fds[cur_layer_idx_to_store] = fd;
    }

    for (int iter = 0; iter < load_num_chunks; ++iter) {
        // advance the index (same as mzcache_load_chunks)
        cur_chunk_idx_to_store++;
        if (cur_chunk_idx_to_store == num_chunks_per_tensor) {
            cur_layer_idx_to_store++;

            if (cur_layer_idx_to_store >= num_layers) break;
            cur_chunk_idx_to_store = chunk_bound[cur_layer_idx_to_store];

            if (layer_fds[cur_layer_idx_to_store] < 0) {
                std::string fn = "KV_" + model_name + "_L" + std::to_string(cur_layer_idx_to_store) + ".bin";
                int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
                if (fd < 0) throw std::runtime_error("open failed for " + fn);
                layer_fds[cur_layer_idx_to_store] = fd;
            }
        }
        int L = cur_layer_idx_to_store;
        int C = cur_chunk_idx_to_store;

        // register the alloc promise
        auto alloc_prom = std::make_shared<std::promise<void>>();
        core->register_kv_allocation(L, C, alloc_prom);

        // (A) ALLOC task
        thread_pool->enqueue(
            CoreType::ALLOC,
            std::bind(&mzcache_kv_state::kv_alloc_worker,
                    this, L, C, alloc_prom)
        );

        // (B) READ task
        thread_pool->enqueue(
            CoreType::READ,
            std::bind(&mzcache_kv_state::kv_read_worker, this, L, C)
        );
    }

    // ── 2) DECOMP: alloc + decompress ─────────────────────────────────────
    for (int iter = 0; iter < decomp_num_chunks; ++iter) {
        // advance the index (taken from the original mzcache_decompress_chunks logic)
        cur_chunk_idx_to_compress++;
        if (cur_chunk_idx_to_compress >= chunk_bound[cur_layer_idx_to_compress]) {
            cur_chunk_idx_to_compress = 0;
            cur_layer_idx_to_compress++;
            if (cur_layer_idx_to_compress >= num_layers) {
                fprintf(stderr, "Error: too many chunks for decompression.\n");
                return;
            }
        }
        int L = cur_layer_idx_to_compress;
        int C = cur_chunk_idx_to_compress;

            // (A) ALLOC task
        auto alloc_prom = std::make_shared<std::promise<void>>();
        core->register_kv_allocation(L, C, alloc_prom);

        thread_pool->enqueue(
            CoreType::ALLOC,
            std::bind(&mzcache_kv_state::kv_alloc_worker,
                    this, L, C, alloc_prom)
        );

        // // Single-threaded version
        thread_pool->enqueue(
            CoreType::DECOMP,
            std::bind(&mzcache_kv_state::kv_decomp_worker,
                    this, L, C)
        );
    }
}

void mzcache_kv_state::mzcache_enqueue_load_chunks(int load_num_chunks) {

    // ── 2) LOAD: alloc + file read ──────────────────────────────────
    // open the needed FDs (same logic as mzcache_load_chunks)

    if(cur_chunk_idx_to_store < num_chunks_per_tensor - 1) {
        // If we are at the last chunk of the current layer, we need to open the file
        std::string fn = "KV_" + model_name + "_L" + std::to_string(cur_layer_idx_to_store) + ".bin";
        int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) throw std::runtime_error("open failed for " + fn);
            layer_fds[cur_layer_idx_to_store] = fd;
    }

    for (int iter = 0; iter < load_num_chunks; ++iter) {
        // advance the index (same as mzcache_load_chunks)
        cur_chunk_idx_to_store++;
        if (cur_chunk_idx_to_store == num_chunks_per_tensor) {
            cur_layer_idx_to_store++;
            
            if (cur_layer_idx_to_store >= num_layers) break;
            cur_chunk_idx_to_store = chunk_bound[cur_layer_idx_to_store];

            if (layer_fds[cur_layer_idx_to_store] < 0) {
                std::string fn = "KV_" + model_name + "_L" + std::to_string(cur_layer_idx_to_store) + ".bin";
                int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
                if (fd < 0) throw std::runtime_error("open failed for " + fn);
                layer_fds[cur_layer_idx_to_store] = fd;
            }
        }
        int L = cur_layer_idx_to_store;
        int C = cur_chunk_idx_to_store;

        // register the alloc promise
        auto alloc_prom = std::make_shared<std::promise<void>>();
        core->register_kv_allocation(L, C, alloc_prom);

        // (A) ALLOC task
        thread_pool->enqueue(
            CoreType::ALLOC,
            std::bind(&mzcache_kv_state::kv_alloc_worker,
                    this, L, C, alloc_prom)
        );

        // (B) READ task
        thread_pool->enqueue(
            CoreType::READ,
            std::bind(&mzcache_kv_state::kv_read_worker, this, L, C)
        );
    }
}

void mzcache_kv_state::mzcache_enqueue_decomp_chunks(int decomp_num_chunks) {

    // ── 1) DECOMP: alloc + decompress ─────────────────────────────────────
    for (int iter = 0; iter < decomp_num_chunks; ++iter) {
        // advance the index (taken from the original mzcache_decompress_chunks logic)
        cur_chunk_idx_to_compress++;
        if (cur_chunk_idx_to_compress >= chunk_bound[cur_layer_idx_to_compress]) {
            cur_chunk_idx_to_compress = 0;
            cur_layer_idx_to_compress++;
            if (cur_layer_idx_to_compress >= num_layers) {
                fprintf(stderr, "Error: too many chunks for decompression.\n");
                return;
            }
        }
        int L = cur_layer_idx_to_compress;
        int C = cur_chunk_idx_to_compress;

            // (A) ALLOC task
        auto alloc_prom = std::make_shared<std::promise<void>>();
        core->register_kv_allocation(L, C, alloc_prom);

        thread_pool->enqueue(
            CoreType::ALLOC,
            std::bind(&mzcache_kv_state::kv_alloc_worker,
                    this, L, C, alloc_prom)
        );

        // // Single-threaded version
        thread_pool->enqueue(
            CoreType::DECOMP,
            std::bind(&mzcache_kv_state::kv_decomp_worker,
                    this, L, C)
        );
    }
  
}

bool mzcache_kv_state::mzcache_enqueue_kv_alloc_only(int decomp_num_chunks, int load_num_chunks) {

    // const int num_threads     = core_configs[CoreType::DECOMP_KERNEL].size();

    // Local copies
    int local_layer = cur_layer_idx_to_compress;
    int local_chunk = cur_chunk_idx_to_compress;

    for (int iter = 0; iter < decomp_num_chunks; ++iter) {
        // advance local indices
        local_chunk++;
        if (local_chunk >= chunk_bound[local_layer]) {
            local_chunk = 0;
            local_layer++;
            if (local_layer >= num_layers) {
                fprintf(stderr, "Error: too many chunks in sequential.\n");
                return false;
            }
        }

        int layer_idx = local_layer;
        int chunk_idx = local_chunk;

        // copy the metadata
        auto Kcc = layers[layer_idx].k_chunks.compressed_chunks[chunk_idx];
        auto Vcc = layers[layer_idx].v_chunks.compressed_chunks[chunk_idx];

        auto alloc_prom = std::make_shared<std::promise<void>>();
        core->register_kv_allocation(layer_idx, chunk_idx, alloc_prom); 

        thread_pool->enqueue(
            CoreType::ALLOC,
            std::bind(&mzcache_kv_state::kv_alloc_worker,
                    this, layer_idx, chunk_idx, alloc_prom)
        );

    }

    const size_t raw_bytes   = TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16);
    const size_t alloc_bytes = raw_bytes * 2;                 // key + value

    int prev_layer_idx_to_store = cur_layer_idx_to_store;

    int tmp_layer = cur_layer_idx_to_store;
    int tmp_chunk = cur_chunk_idx_to_store;

    for (int i = 0; i < load_num_chunks; ++i) {

        // next (L,C)
        tmp_chunk++;
        if (tmp_chunk == num_chunks_per_tensor) {
            tmp_layer++;
            if (tmp_layer >= num_layers) {
                fprintf(stderr, "Error: too many chunks.\n");
                return false;
            }
            tmp_chunk = chunk_bound[tmp_layer];
        }
        int L = tmp_layer, C = tmp_chunk;

        /* register the alloc promise */
        auto alloc_prom = std::make_shared<std::promise<void>>();
        core->register_kv_allocation(L, C, alloc_prom);

        // (A) ALLOC task
        thread_pool->enqueue(
            CoreType::ALLOC,
            std::bind(&mzcache_kv_state::kv_alloc_worker,
                    this, L, C, alloc_prom)
        );
    }

    local_layer = cur_layer_idx_to_compress;
    local_chunk = cur_chunk_idx_to_compress;

    for (int iter = 0; iter < decomp_num_chunks; ++iter) {
        // advance local indices
        local_chunk++;
        if (local_chunk >= chunk_bound[local_layer]) {
            local_chunk = 0;
            local_layer++;
            if (local_layer >= num_layers) {
                fprintf(stderr, "Error: too many chunks in sequential.\n");
                return false;
            }
        }

        // std::cout << "Waiting for KV allocation for layer " << local_layer
        //           << ", chunk " << local_chunk << std::endl;

        core->wait_for_kv_allocation(local_layer, local_chunk);
    }

    tmp_layer = cur_layer_idx_to_store;
    tmp_chunk = cur_chunk_idx_to_store;

    for (int iter = 0; iter < load_num_chunks; ++iter) {

        // advance local indices
        tmp_chunk++;
        if (tmp_chunk == num_chunks_per_tensor) {
            tmp_layer++;
            if (tmp_layer >= num_layers) {
                fprintf(stderr, "Error: too many chunks.\n");
                return false;
            }
            tmp_chunk = chunk_bound[tmp_layer];
        }

        // std::cout << "Waiting for KV allocation for layer " << tmp_layer
        //           << ", chunk " << tmp_chunk << std::endl;

        core->wait_for_kv_allocation(tmp_layer, tmp_chunk);
    }


    return true;

}

bool mzcache_kv_state::mzcache_wait_for_kv(int decomp_num_chunks, int load_num_chunks) {

    int local_layer = cur_layer_idx_to_compress;
    int local_chunk = cur_chunk_idx_to_compress;

    for (int iter = 0; iter < decomp_num_chunks; ++iter) {
        // advance local indices
        local_chunk++;
        if (local_chunk >= chunk_bound[local_layer]) {
            local_chunk = 0;
            local_layer++;
            if (local_layer >= num_layers) {
                fprintf(stderr, "Error: too many chunks in sequential.\n");
                return false;
            }
        }

        // std::cout << "Waiting for KV allocation for layer " << local_layer
        //           << ", chunk " << local_chunk << std::endl;

        core->wait_for_kv_allocation(local_layer, local_chunk);
    }

    int tmp_layer = cur_layer_idx_to_store;
    int tmp_chunk = cur_chunk_idx_to_store;

    for (int iter = 0; iter < load_num_chunks; ++iter) {

        // advance local indices
        tmp_chunk++;
        if (tmp_chunk == num_chunks_per_tensor) {
            tmp_layer++;
            if (tmp_layer >= num_layers) {
                fprintf(stderr, "Error: too many chunks.\n");
                return false;
            }
            tmp_chunk = chunk_bound[tmp_layer];
        }

        // std::cout << "Waiting for KV allocation for layer " << tmp_layer
        //           << ", chunk " << tmp_chunk << std::endl;

        core->wait_for_kv_allocation(tmp_layer, tmp_chunk);
    }

    return true;
}

bool mzcache_kv_state::mzcache_enqueue_kv_decomp_read_only(int decomp_num_chunks, int load_num_chunks) {

        // ── 1) DECOMP: alloc + decompress ─────────────────────────────────────
    for (int iter = 0; iter < decomp_num_chunks; ++iter) {
        // advance the index (taken from the original mzcache_decompress_chunks logic)
        cur_chunk_idx_to_compress++;
        if (cur_chunk_idx_to_compress >= chunk_bound[cur_layer_idx_to_compress]) {
            cur_chunk_idx_to_compress = 0;
            cur_layer_idx_to_compress++;
            if (cur_layer_idx_to_compress >= num_layers) {
                fprintf(stderr, "Error: too many chunks for decompression.\n");
                return false;
            }
        }
        int L = cur_layer_idx_to_compress;
        int C = cur_chunk_idx_to_compress;

        // // Single-threaded version
        thread_pool->enqueue(
            CoreType::DECOMP,
            std::bind(&mzcache_kv_state::kv_decomp_worker,
                    this, L, C)
        );
    }

    // ── 2) LOAD: alloc + file read ──────────────────────────────────
    // open the needed FDs (same logic as mzcache_load_chunks)

    if(cur_chunk_idx_to_store < num_chunks_per_tensor - 1) {
        // If we are at the last chunk of the current layer, we need to open the file
        std::string fn = "KV_" + model_name + "_L" + std::to_string(cur_layer_idx_to_store) + ".bin";
        int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) throw std::runtime_error("open failed for " + fn);
            layer_fds[cur_layer_idx_to_store] = fd;
    }

    for (int iter = 0; iter < load_num_chunks; ++iter) {
        // advance the index (same as mzcache_load_chunks)
        cur_chunk_idx_to_store++;
        if (cur_chunk_idx_to_store == num_chunks_per_tensor) {
            cur_layer_idx_to_store++;
            
            if (cur_layer_idx_to_store >= num_layers) break;
            cur_chunk_idx_to_store = chunk_bound[cur_layer_idx_to_store];

            if (layer_fds[cur_layer_idx_to_store] < 0) {
                std::string fn = "KV_" + model_name + "_L" + std::to_string(cur_layer_idx_to_store) + ".bin";
                int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
                if (fd < 0) throw std::runtime_error("open failed for " + fn);
                layer_fds[cur_layer_idx_to_store] = fd;
            }
        }
        int L = cur_layer_idx_to_store;
        int C = cur_chunk_idx_to_store;

        // (B) READ task
        thread_pool->enqueue(
            CoreType::READ,
            std::bind(&mzcache_kv_state::kv_read_worker, this, L, C)
        );
    }

    return true;

}

static ssize_t full_pwrite(int fd, const void* buf, size_t count, off_t off) {
    const char* p = static_cast<const char*>(buf);
    size_t done = 0;
    while (done < count) {
        ssize_t n = ::pwrite(fd, p + done, count - done, off + done);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (n == 0) break;
        done += static_cast<size_t>(n);
    }
    return static_cast<ssize_t>(done);
}

static ssize_t full_pread(int fd, void* buf, size_t count, off_t off) {
    char* p = static_cast<char*>(buf);
    size_t done = 0;
    while (done < count) {
        ssize_t n = ::pread(fd, p + done, count - done, off + done);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (n == 0) break;
        done += static_cast<size_t>(n);
    }
    return static_cast<ssize_t>(done);
}

static int open_odirect_write(const char* path) {
    int flags = O_CREAT | O_WRONLY | O_TRUNC;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_DIRECT
    flags |= O_DIRECT;
#endif
    int fd = ::open(path, flags, 0644);
    if (fd < 0 && (errno == EINVAL || errno == EOPNOTSUPP || errno == ENOTTY)) {
        int fb = O_CREAT | O_WRONLY | O_TRUNC;
#ifdef O_CLOEXEC
        fb |= O_CLOEXEC;
#endif
        fd = ::open(path, fb, 0644);
    }
    return fd;
}

static int open_odirect_read(const char* path) {
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_DIRECT
    flags |= O_DIRECT;
#endif
    int fd = ::open(path, flags);
    if (fd < 0 && (errno == EINVAL || errno == EOPNOTSUPP || errno == ENOTTY)) {
        int fb = O_RDONLY;
#ifdef O_CLOEXEC
        fb |= O_CLOEXEC;
#endif
        fd = ::open(path, fb);
    }
    return fd;
}

static inline size_t round_up_4k(size_t n) {
    return (n + 4095u) & ~size_t(4095u);
}

int64_t mzcache_kv_state::mzcache_offload_arenas() {
    // Offload one layer per call (in reverse order)
    if (next_layer_to_offload < 0) {
        std::fprintf(stderr, "Warning: No more layers to offload (next_layer_to_offload=%d)\n", 
                     next_layer_to_offload);
        return 0;
    }

    int layer = next_layer_to_offload;
    std::string swap_path = get_swap_path(layer);
    int fd = open_odirect_write(swap_path.c_str());
    if (fd < 0) {
        std::perror(("open(offload) layer " + std::to_string(layer)).c_str());
        return 0;
    }

    off_t off = 0;
    int64_t total_offloaded_bytes = 0;
    offload_used_bytes_[layer].clear();
    offload_used_bytes_[layer].reserve(arenas[layer].size());

    for (Arena* a : arenas[layer]) {
        const size_t used = a->top;
        if (used) {
            const size_t padded = round_up_4k(used);
            if (padded > used) {
                std::memset(a->pool + used, 0, padded - used);
            }
            ssize_t n = full_pwrite(fd, a->pool, padded, off);
            if (n != (ssize_t)padded) {
                std::perror(("pwrite(offload/O_DIRECT) layer " + std::to_string(layer)).c_str());
                ::close(fd);
                return 0;
            }
            off += (off_t)padded;
            total_offloaded_bytes += (int64_t)padded;
        }
        offload_used_bytes_[layer].push_back(used);
    }

    ::fsync(fd);
    ::close(fd);

    // Free the current layer's arena memory
    for (Arena* a : arenas[layer]) delete a;
    arenas[layer].clear();

    // std::printf("Offloaded layer %d arenas to %s (size: %lld bytes)\n", 
    //             layer, swap_path.c_str(), (long long)total_offloaded_bytes);

    // Update the next layer to offload (reverse order)
    next_layer_to_offload--;

    n_cur_arena_offload_bytes += total_offloaded_bytes;

    return total_offloaded_bytes;
}

void mzcache_kv_state::mzcache_reload_arenas() {
    // Start reloading from next_layer_to_offload + 1
    // If next_layer_to_offload == -1 start at layer 0, otherwise at next_layer_to_offload + 1
    int layer = (next_layer_to_offload < 0) ? 0 : (next_layer_to_offload + 1);
    
    // Check the layer range is valid
    if (layer >= num_layers) {
        std::fprintf(stderr, "Warning: No layers to reload (next_layer_to_offload=%d)\n", 
                     next_layer_to_offload);
        return;
    }

    // Check whether the layer was offloaded (a non-empty offload_used_bytes_ means it was)
    if (offload_used_bytes_[layer].empty()) {
        std::fprintf(stderr, "Warning: Layer %d was not offloaded\n", layer);
        return;
    }

    // Clean up the existing arenas
    for (Arena* a : arenas[layer]) delete a;
    arenas[layer].clear();

    std::string swap_path = get_swap_path(layer);
    int fd = open_odirect_read(swap_path.c_str());
    if (fd < 0) {
        std::perror(("open(reload) layer " + std::to_string(layer)).c_str());
        return;
    }

    arenas[layer].reserve(offload_used_bytes_[layer].size());

    off_t off = 0;
    for (size_t used : offload_used_bytes_[layer]) {
        if (used > arena_bytes) {
            std::fprintf(stderr, "reload layer %d: used_bytes(%zu) > arena_bytes(%zu)\n",
                         layer, used, arena_bytes);
            ::close(fd);
            for (Arena* a : arenas[layer]) delete a;
            arenas[layer].clear();
            return;
        }
        Arena* a = new Arena(arena_bytes, /*allow_svm=*/false);  // reload: always malloc
        if (used) {
            const size_t padded = round_up_4k(used);
            ssize_t n = full_pread(fd, a->pool, padded, off);
            if (n != (ssize_t)padded) {
                std::perror(("pread(reload/O_DIRECT) layer " + std::to_string(layer)).c_str());
                delete a;
                ::close(fd);
                for (Arena* aa : arenas[layer]) delete aa;
                arenas[layer].clear();
                return;
            }
            off += (off_t)padded;
        }
        a->top = used;
        arenas[layer].push_back(a);
    }

    ::close(fd);
    ::unlink(swap_path.c_str());

    // std::printf("Reloaded layer %d arenas from %s\n", layer, swap_path.c_str());

    for (size_t used : offload_used_bytes_[layer]) {
        if (used) n_cur_arena_offload_bytes -= (int64_t)round_up_4k(used);
    }
    if (n_cur_arena_offload_bytes < 0) n_cur_arena_offload_bytes = 0;

    // Increment next_layer_to_offload for the next reload
    next_layer_to_offload++;
    
    // If all layers were reloaded, reset to num_layers - 1
    if (next_layer_to_offload >= num_layers) {
        next_layer_to_offload = num_layers - 1;
    }
}

void mzcache_kv_state::mzcache_reload_all_arenas() {
    // Reload everything from next_layer_to_offload + 1 through num_layers - 1
    int start_layer = (next_layer_to_offload < 0) ? 0 : (next_layer_to_offload + 1);
    
    // Check the layer range is valid
    if (start_layer >= num_layers) {
        return;
    }

    // Reload everything from start_layer through num_layers-1
    for (int layer = start_layer; layer < num_layers; ++layer) {
        // Check whether the layer was offloaded (a non-empty offload_used_bytes_ means it was)
        if (offload_used_bytes_[layer].empty()) {
            std::fprintf(stderr, "Warning: Layer %d was not offloaded, skipping\n", layer);
            continue;
        }

        // Clean up the existing arenas
        for (Arena* a : arenas[layer]) delete a;
        arenas[layer].clear();

        std::string swap_path = get_swap_path(layer);
        int fd = open_odirect_read(swap_path.c_str());
        if (fd < 0) {
            std::perror(("open(reload) layer " + std::to_string(layer)).c_str());
            continue;
        }

        arenas[layer].reserve(offload_used_bytes_[layer].size());

        off_t off = 0;
        for (size_t used : offload_used_bytes_[layer]) {
            if (used > arena_bytes) {
                std::fprintf(stderr, "reload layer %d: used_bytes(%zu) > arena_bytes(%zu)\n",
                             layer, used, arena_bytes);
                ::close(fd);
                for (Arena* a : arenas[layer]) delete a;
                arenas[layer].clear();
                break;
            }
            Arena* a = new Arena(arena_bytes, /*allow_svm=*/false);  // reload: always malloc
            if (used) {
                const size_t padded = round_up_4k(used);
                ssize_t n = full_pread(fd, a->pool, padded, off);
                if (n != (ssize_t)padded) {
                    std::perror(("pread(reload/O_DIRECT) layer " + std::to_string(layer)).c_str());
                    delete a;
                    ::close(fd);
                    for (Arena* aa : arenas[layer]) delete aa;
                    arenas[layer].clear();
                    break;
                }
                off += (off_t)padded;
            }
            a->top = used;
            arenas[layer].push_back(a);
        }

        ::close(fd);
        ::unlink(swap_path.c_str());

        // std::printf("Reloaded layer %d arenas from %s\n", layer, swap_path.c_str());
    }

    // All layers reloaded, so reset next_layer_to_offload to num_layers - 1
    next_layer_to_offload = num_layers - 1;
    n_cur_arena_offload_bytes = 0;
    
    // std::printf("Reloaded all layers from %d to %d\n", start_layer, num_layers - 1);
}

void mzcache_kv_state::clean_after_decompress(int decomp_chunks)
{
    if (decomp_chunks <= 0) {
        return;
    }

    for (int L = 0; L < num_layers; ++L) {
        int last_stack_id = -1;
        size_t last_end = 0;

        for (int C = 0; C < num_chunks_per_tensor; ++C) {
            if (layers[L].states[C] != KVstate::Compressed) {
                continue;
            }

            auto update_last_used = [&](const compressed_chunk & cc) {
                const size_t chunk_end = (size_t)cc.offset + cc.comp_size;
                if (cc.stack_id > last_stack_id || (cc.stack_id == last_stack_id && chunk_end > last_end)) {
                    last_stack_id = cc.stack_id;
                    last_end = chunk_end;
                }
            };

            update_last_used(layers[L].k_chunks.compressed_chunks[C]);
            update_last_used(layers[L].v_chunks.compressed_chunks[C]);
        }

        if (last_stack_id < 0) {
            // Layer fully decompressed: release the pools entirely (SVM pages
            // are committed eagerly, so an empty resident arena is not free).
            // mzcache_compress_chunks reallocates on demand.
            for (Arena * arena : arenas[L]) {
                delete arena;
            }
            arenas[L].clear();
            continue;
        }

        if (arenas[L].empty()) {
            continue;
        }

        const int max_valid_stack = std::min(last_stack_id, (int)arenas[L].size() - 1);
        for (int i = (int)arenas[L].size() - 1; i > max_valid_stack; --i) {
            delete arenas[L][i];
            arenas[L].pop_back();
        }

        arenas[L][max_valid_stack]->top = last_end;
    }

    if (decomp_chunks > n_cur_comp_chunks) {
        MZ_LOG_ERROR("clean_after_decompress: decomp_chunks(%d) > n_cur_comp_chunks(%d)", decomp_chunks, n_cur_comp_chunks);
        n_cur_comp_chunks = 0;
    } else {
        // Scale the measured compression savings by the surviving fraction;
        // exact per-chunk sizes are gone once the chunk is decompressed.
        n_cur_comp_saved_bytes = n_cur_comp_saved_bytes *
            (int64_t)(n_cur_comp_chunks - decomp_chunks) / (int64_t)n_cur_comp_chunks;
        n_cur_comp_chunks -= decomp_chunks;
    }
    if (n_cur_comp_chunks == 0) {
        n_cur_comp_saved_bytes = 0;
    }
}

void mzcache_kv_state::clean_after_read(int read_chunks)
{
    for (int l = 0; l < num_layers; ++l) {
        if (layer_fds[l] >= 0) {
            ::close(layer_fds[l]);
            layer_fds[l] = -1; // reset to -1 after closing
        }
    }

    if (read_chunks > n_cur_store_chunks) {
        MZ_LOG_ERROR("clean_after_read: read_chunks(%d) > n_cur_store_chunks(%d)", read_chunks, n_cur_store_chunks);
        n_cur_store_chunks = 0;
    } else {
        n_cur_store_chunks -= read_chunks;
    }
}

#endif // MZCACHE_SVM_KV_CHUNK
