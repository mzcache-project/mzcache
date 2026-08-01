// mz_device_profile — one-time device profiling for mzcache's two scheduling
// constants, which are currently hard-coded per (device, model) in
// mzcache_core.cpp / mzcache_kv_state.cpp:
//
//   per_layer_balance   how many KV chunks the DECOMP pool can decompress in
//                       the time the READ path loads one weight layer
//                       = (weight_layer_bytes / R_weight) * D_chunks
//   decomp_load_ratio   the comp-vs-store split used by the swapout planner
//                       (want_comp = requested * ratio). For the swap-in
//                       decompress stream and the store-read stream to finish
//                       together:  ratio = D_chunks / (D_chunks + R_kv_chunks)
//
// Both derive from two *device-level* throughputs that are independent of the
// model shape (validated here by profiling two chunk sizes):
//
//   D      FLEXGEN-8bit decompression throughput (raw fp16 bytes produced/s),
//          measured with the real kernel on the real DECOMP core set
//   R_kv   KV store-chunk sequential read throughput (O_DIRECT, chunk-sized
//          preads on the READ core set — same pattern as kv_read_worker)
//   R_w    weight-layer read throughput (O_DIRECT, layer-sized preads — same
//          pattern as weight_read_worker)
//
// The model then only contributes weight_layer_bytes and the chunk size
// (TOKENS_PER_CHUNK * kv_hidden_dim * 2B * 2 tensors).
//
// Usage: mz_device_profile [file_mb=512] [passes=3]
//   Creates ./mz_profile_test.bin (file_mb MB) in the CWD on first run.
//   Prints measured device constants and the derived per-model values next to
//   the currently hard-coded ones (FLEXGEN_8BIT table).

#include "mzcache_threadpool.h"
#include "mzcache_profile.h"
#include "mzcache_types.h"

#include <CL/cl.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}


// --------------------------------------------------------------------------
// FLEXGEN-8bit decompression throughput on the DECOMP core set.
// One job == one planner chunk == K + V tensors, exactly like kv_decomp_worker.
// Buffers rotate over a working set far larger than the caches so inputs come
// from DRAM like the runtime's comp arenas.
//
// Buffers are placed exactly like the runtime's: the compressed input comes
// from a plain malloc arena (Arena is malloc + mlock now, CPU/NEON-read during
// decode), and with a CL context the decompressed output goes to
// CL_MEM_SVM_FINE_GRAIN_BUFFER chunks (see create_chunked_kv_tensors, GPU-read).
// Without a context the output also falls back to malloc, which measures the
// bare kernel and overstates the throughput the scheduler actually sees.
// --------------------------------------------------------------------------
static double bench_decomp_gbps(ThreadPool & pool, cl_context clctx,
                                int hidden_dim, int n_jobs, int passes) {
    const int    elems      = hidden_dim * TOKENS_PER_CHUNK;   // per tensor
    const int    num_groups = elems / 64;
    const size_t comp_bytes = mzcache::profile_comp_bytes(elems);
    const size_t raw_bytes  = (size_t) elems * sizeof(__fp16);

    const int n_sets = 96;  // 96 jobs * (2*comp + 2*raw) ~ 288MB @ hidden 1024

    struct Set {
        uint8_t *comp_k, *comp_v;
        std::vector<__fp16>  mins_k, maxs_k, mins_v, maxs_v;
        __fp16 *out_k, *out_v;
    };
    std::vector<Set> sets(n_sets);

    // Compressed input always malloc: the runtime's comp Arena is malloc now
    // (decode reads it on the CPU), independent of whether the output is SVM.
    auto alloc_in = [&](size_t n) -> uint8_t * {
        uint8_t * p; posix_memalign((void **) &p, 4096, n); return p;
    };
    auto alloc_out = [&](size_t n) -> __fp16 * {
        if (clctx) return (__fp16 *) clSVMAlloc(clctx,
                        CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, n, 0);
        __fp16 * p; posix_memalign((void **) &p, 4096, n); return p;
    };

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> fmin(-1.0f, 0.0f), fspan(0.1f, 2.0f);

    for (auto & s : sets) {
        s.comp_k = alloc_in(comp_bytes);
        s.comp_v = alloc_in(comp_bytes);
        for (size_t i = 0; i < comp_bytes; ++i) { s.comp_k[i] = (uint8_t) byte(rng); s.comp_v[i] = (uint8_t) byte(rng); }
        s.mins_k.resize(num_groups); s.maxs_k.resize(num_groups);
        s.mins_v.resize(num_groups); s.maxs_v.resize(num_groups);
        for (int g = 0; g < num_groups; ++g) {
            float mn = fmin(rng);
            s.mins_k[g] = (__fp16) mn; s.maxs_k[g] = (__fp16)(mn + fspan(rng));
            s.mins_v[g] = (__fp16) mn; s.maxs_v[g] = (__fp16)(mn + fspan(rng));
        }
        s.out_k = alloc_out(raw_bytes);
        s.out_v = alloc_out(raw_bytes);
        memset(s.out_k, 0, raw_bytes);  // commit pages up front
        memset(s.out_v, 0, raw_bytes);
    }

    std::vector<double> gbps;
    for (int p = 0; p < passes; ++p) {
        const double t0 = now_s();
        for (int j = 0; j < n_jobs; ++j) {
            Set & s = sets[j % n_sets];
            pool.enqueue(CoreType::DECOMP, [&s, hidden_dim] {
                mzcache::profile_decomp_chunk(s.comp_k, hidden_dim, s.out_k,
                                              s.mins_k.data(), s.maxs_k.data());
                mzcache::profile_decomp_chunk(s.comp_v, hidden_dim, s.out_v,
                                              s.mins_v.data(), s.maxs_v.data());
            });
        }
        pool.wait_idle(CoreType::DECOMP);
        const double dt = now_s() - t0;
        gbps.push_back((double) n_jobs * raw_bytes * 2 / dt / 1e9);
    }
    for (auto & s : sets) {
        free(s.comp_k); free(s.comp_v);            // input: always malloc
        if (clctx) { clSVMFree(clctx, s.out_k); clSVMFree(clctx, s.out_v); }
        else       { free(s.out_k); free(s.out_v); }
    }
    return median(gbps);
}

// --------------------------------------------------------------------------
// Sequential O_DIRECT read throughput with block-sized preads issued from the
// given core set — chunk-sized blocks model kv_read_worker, layer-sized blocks
// model weight_read_worker.
// --------------------------------------------------------------------------
static double bench_read_gbps(ThreadPool & pool, CoreType core, const char * path,
                              size_t file_bytes, size_t block_bytes, int passes) {
    std::vector<double> gbps;
    const size_t n_blocks = file_bytes / block_bytes;

    for (int p = 0; p < passes; ++p) {
        int fd = ::open(path, O_RDONLY | O_DIRECT);
        if (fd < 0) { perror("open O_DIRECT"); return 0.0; }

        const double t0 = now_s();
        for (size_t b = 0; b < n_blocks; ++b) {
            pool.enqueue(core, [fd, b, block_bytes] {
                // one aligned buffer per worker thread
                static thread_local uint8_t * buf = nullptr;
                if (!buf) posix_memalign((void **) &buf, 4096, 128u << 20);
                ssize_t n = pread(fd, buf, block_bytes, (off_t)(b * block_bytes));
                if (n != (ssize_t) block_bytes) perror("pread");
            });
        }
        pool.wait_idle(core);
        const double dt = now_s() - t0;
        ::close(fd);
        gbps.push_back((double) n_blocks * block_bytes / dt / 1e9);
    }
    return median(gbps);
}

// --------------------------------------------------------------------------
// Concurrent-stream variant: run the store-chunk read stream (READ cores) and
// the decompress stream (DECOMP cores) at the same time, the way swap-in
// actually does, and report each stream's achieved rate. Job counts are sized
// from the isolated rates so both streams span roughly the same window. This
// captures the memory-bus/IO interference between the two streams (though not
// the concurrent GPU prefill of the fully-overlapped path).
// --------------------------------------------------------------------------
// Sustained chunk-allocation churn on the ALLOC core set: each job is one
// clSVMAlloc(fine-grained, chunk_bytes) exactly like kv_alloc_worker, with a
// bounded ring so old allocations are freed as new ones appear (the runtime's
// FREE pool similarly releases comp arenas while allocation proceeds). On
// SM8750 the ALLOC cores {0..5} overlap the READ cores {0,1}, so this also
// reproduces the CPU contention the read stream sees during swap-in.
struct AllocLoad {
    ThreadPool & pool;
    cl_context   ctx;
    size_t       chunk_bytes;
    std::atomic<bool> stop{false};
    std::mutex   mu;
    std::vector<void *> ring;
    size_t       cap;

    AllocLoad(ThreadPool & p, cl_context c, size_t bytes, size_t cap_bytes = 512u << 20)
        : pool(p), ctx(c), chunk_bytes(bytes), cap(cap_bytes / bytes) {}

    void job() {
        if (stop.load()) return;
        void * p = clSVMAlloc(ctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                              chunk_bytes, 0);
        if (p) {
            void * victim = nullptr;
            {
                std::lock_guard<std::mutex> l(mu);
                ring.push_back(p);
                if (ring.size() > cap) { victim = ring.front(); ring.erase(ring.begin()); }
            }
            if (victim) clSVMFree(ctx, victim);
        }
        if (!stop.load()) pool.enqueue(CoreType::ALLOC, [this] { job(); });
    }
    void start(int workers) {
        if (!ctx) return;
        stop.store(false);
        for (int i = 0; i < workers * 2; ++i) pool.enqueue(CoreType::ALLOC, [this] { job(); });
    }
    void end() {
        if (!ctx) return;
        stop.store(true);
        pool.wait_idle(CoreType::ALLOC);
        std::lock_guard<std::mutex> l(mu);
        for (void * p : ring) clSVMFree(ctx, p);
        ring.clear();
    }
};

// Saturates GPU memory bandwidth with back-to-back device-side buffer copies —
// a stand-in for the concurrent GPU prefill of the fully-overlapped swap-in.
struct GpuLoad {
    cl_command_queue q = nullptr;
    cl_mem a = nullptr, b = nullptr;
    std::thread th;
    std::atomic<bool> stop{false};

    void start(cl_context ctx, cl_device_id dev) {
        if (!ctx) return;
        cl_int err;
        const size_t n = 128u << 20;
        q = clCreateCommandQueue(ctx, dev, 0, &err);
        a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n, nullptr, &err);
        b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n, nullptr, &err);
        if (!q || !a || !b) return;
        th = std::thread([this, n] {
            while (!stop.load()) {
                for (int i = 0; i < 4 && !stop.load(); ++i) {
                    clEnqueueCopyBuffer(q, a, b, 0, 0, n, 0, nullptr, nullptr);
                    clEnqueueCopyBuffer(q, b, a, 0, 0, n, 0, nullptr, nullptr);
                }
                clFinish(q);
            }
        });
    }
    void end() {
        if (!th.joinable()) return;
        stop.store(true);
        th.join();
        clFinish(q);
        clReleaseMemObject(a); clReleaseMemObject(b); clReleaseCommandQueue(q);
    }
};

struct ConcRates { double d_gbps, r_gbps; };

static ConcRates bench_concurrent(ThreadPool & pool, cl_context clctx, int hidden_dim,
                                  const char * path, size_t file_bytes,
                                  double d_iso_gbps, double r_iso_gbps, double window_s) {
    const size_t chunk_bytes = (size_t) TOKENS_PER_CHUNK * hidden_dim * sizeof(__fp16) * 2;
    const int n_d = std::max(1, (int) (d_iso_gbps * 1e9 * window_s / chunk_bytes));
    const int n_r = std::max(1, (int) (r_iso_gbps * 1e9 * window_s / chunk_bytes));

    // decomp buffers (same layout as bench_decomp_gbps, SVM when available)
    const int    elems      = hidden_dim * TOKENS_PER_CHUNK;
    const int    num_groups = elems / 64;
    const size_t comp_bytes = mzcache::profile_comp_bytes(elems);
    const size_t raw_bytes  = (size_t) elems * sizeof(__fp16);
    const int n_sets = 96;

    std::vector<uint8_t *> comp(n_sets * 2);
    std::vector<__fp16 *>  out(n_sets * 2);
    std::vector<__fp16>    mins(num_groups, (__fp16) -0.5f), maxs(num_groups, (__fp16) 0.5f);
    for (int i = 0; i < n_sets * 2; ++i) {
        // Compressed input: always malloc (the runtime's comp Arena is malloc,
        // CPU-read during decode). Decompressed output: SVM fine-grain when a CL
        // context is present (the GPU-read KV chunks), else malloc.
        posix_memalign((void **) &comp[i], 4096, comp_bytes);
        if (clctx) out[i] = (__fp16 *) clSVMAlloc(clctx, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, raw_bytes, 0);
        else       posix_memalign((void **) &out[i], 4096, raw_bytes);
        memset(comp[i], 0x5a, comp_bytes);
        memset(out[i], 0, raw_bytes);
    }

    int fd = ::open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open"); }
    const size_t n_blocks_in_file = file_bytes / chunk_bytes;

    std::atomic<double> t_d_done{0.0}, t_r_done{0.0};

    const double t0 = now_s();
    for (int j = 0; j < n_d; ++j) {
        const bool last = (j == n_d - 1);
        const int  i = (j % n_sets) * 2;
        pool.enqueue(CoreType::DECOMP, [&, i, last] {
            mzcache::profile_decomp_chunk(comp[i],     hidden_dim, out[i],     mins.data(), maxs.data());
            mzcache::profile_decomp_chunk(comp[i + 1], hidden_dim, out[i + 1], mins.data(), maxs.data());
            if (last) t_d_done.store(now_s());
        });
    }
    for (int j = 0; j < n_r; ++j) {
        const bool last = (j == n_r - 1);
        const size_t b = j % n_blocks_in_file;
        pool.enqueue(CoreType::READ, [&, b, last] {
            static thread_local uint8_t * buf = nullptr;
            if (!buf) posix_memalign((void **) &buf, 4096, 8u << 20);
            ssize_t n = pread(fd, buf, chunk_bytes, (off_t)(b * chunk_bytes));
            if (n != (ssize_t) chunk_bytes) perror("pread");
            if (last) t_r_done.store(now_s());
        });
    }
    pool.wait_idle(CoreType::DECOMP);
    pool.wait_idle(CoreType::READ);
    ::close(fd);

    ConcRates rr;
    rr.d_gbps = (double) n_d * chunk_bytes / (t_d_done.load() - t0) / 1e9;
    rr.r_gbps = (double) n_r * chunk_bytes / (t_r_done.load() - t0) / 1e9;

    for (int i = 0; i < n_sets * 2; ++i) {
        free(comp[i]);                             // input: always malloc
        if (clctx) clSVMFree(clctx, out[i]);
        else       free(out[i]);
    }
    return rr;
}

static void ensure_test_file(const char * path, size_t bytes) {
    struct stat st;
    if (::stat(path, &st) == 0 && (size_t) st.st_size >= bytes) return;

    std::cout << "Writing test file " << path << " (" << bytes / (1u << 20) << " MB)...\n";
    int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("create test file"); exit(1); }
    std::vector<uint8_t> blk(4u << 20);
    std::mt19937 rng(7);
    for (auto & b : blk) b = (uint8_t) rng();
    for (size_t off = 0; off < bytes; off += blk.size()) {
        if (write(fd, blk.data(), blk.size()) != (ssize_t) blk.size()) { perror("write"); exit(1); }
    }
    fsync(fd);
    ::close(fd);
}

struct ModelShape {
    const char * name;
    int64_t layer_bytes;   // mzcache_core weight_layer_bytes
    int     kv_hidden;     // per-tensor hidden dim
    int     hard_balance;  // current hard-coded per_layer_balance (per SoC, 8bit)
};

int main(int argc, char ** argv) {
    if (!mzcache::profiling_supported()) {
        std::cerr << "mz_device_profile: profiling is not supported for compression '"
                  << mzcache::compression_name() << "' (FLEXGEN / FLEXGEN_8BIT only)\n";
        return 1;
    }
    const size_t file_mb = (argc > 1) ? strtoul(argv[1], nullptr, 10) : 512;
    const int    passes  = (argc > 2) ? atoi(argv[2]) : 3;

    const std::string soc_str = mzcache::soc_model();
    const mzcache::SocClass soc = mzcache::classify_soc(soc_str);
    const bool sm8750 = (soc == mzcache::SocClass::SM8750);
    std::cout << "SoC: " << soc_str << " -> " << mzcache::soc_name(soc) << "\n";
    if (soc == mzcache::SocClass::UNKNOWN) {
        std::cerr << "WARNING: unrecognized SoC '" << soc_str
                  << "' — supported: SM8750 (Snapdragon 8 Elite), SM8650 (Snapdragon 8 Gen 3).\n"
                  << "         Profiling with the SM8650 core placement.\n";
    }

    // Same DECOMP/READ core sets as mzcache_core (FLEXGEN/FLEXGEN_8BIT table).
    auto cfg = mzcache::default_core_configs(soc);
    const int n_alloc_workers = (int) cfg.at(CoreType::ALLOC).size();
    ThreadPool pool(cfg);
    const float hard_ratio = sm8750 ? 0.65f : 0.55f;

    const ModelShape models[] = {
        { "Qwen3-0.6B",      31469568, 1024, sm8750 ?  57 :  36 },
        { "EXAONE-4.0-1.2B", 71323648,  512, sm8750 ? 240 : 140 },
    };

    const char * test_file = "./mz_profile_test.bin";
    const size_t file_bytes = file_mb << 20;
    ensure_test_file(test_file, file_bytes);

    // CL context so decomp writes land in SVM exactly like the runtime chunks.
    cl_context clctx = nullptr;
    cl_device_id cldev = nullptr;
    {
        cl_platform_id plat;
        if (clGetPlatformIDs(1, &plat, nullptr) == CL_SUCCESS &&
            clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &cldev, nullptr) == CL_SUCCESS) {
            cl_int err;
            clctx = clCreateContext(nullptr, 1, &cldev, nullptr, nullptr, &err);
            if (err != CL_SUCCESS) clctx = nullptr;
        }
        std::cout << (clctx ? "OpenCL context OK — decomp buffers in SVM (runtime-faithful)\n"
                            : "WARNING: no OpenCL context — decomp buffers in malloc (kernel-only upper bound)\n");
    }

    {
        std::cout << "\n########## compression: " << mzcache::compression_name()
                  << " ##########\n";
        std::cout << "\n=== device throughputs (median of " << passes << ") ===\n";

        // Decompression at both model chunk shapes (validates size independence).
        double d_gbps[2], d_malloc[2];
        for (int m = 0; m < 2; ++m) {
            d_gbps[m]   = bench_decomp_gbps(pool, clctx,   models[m].kv_hidden, 1024, passes);
            d_malloc[m] = bench_decomp_gbps(pool, nullptr, models[m].kv_hidden, 1024, passes);
            std::cout << "decomp  (hidden " << models[m].kv_hidden << "): "
                      << d_gbps[m] << " GB/s raw out (svm)  |  "
                      << d_malloc[m] << " GB/s (malloc, kernel-only)\n";
        }

        // KV store-chunk reads at both chunk sizes.
        double rkv_gbps[2];
        for (int m = 0; m < 2; ++m) {
            const size_t chunk = (size_t) TOKENS_PER_CHUNK * models[m].kv_hidden * sizeof(__fp16) * 2;
            rkv_gbps[m] = bench_read_gbps(pool, CoreType::READ, test_file, file_bytes, chunk, passes);
            std::cout << "kv read (chunk " << chunk / 1024 << " KB): "
                      << rkv_gbps[m] << " GB/s\n";
        }

        // Weight-layer reads at both layer sizes.
        double rw_gbps[2];
        for (int m = 0; m < 2; ++m) {
            rw_gbps[m] = bench_read_gbps(pool, CoreType::READ, test_file,
                                         (file_bytes / models[m].layer_bytes) * models[m].layer_bytes,
                                         (size_t) models[m].layer_bytes, passes);
            std::cout << "w  read (layer " << models[m].layer_bytes / (1 << 20) << " MB): "
                      << rw_gbps[m] << " GB/s\n";
        }

        // Concurrent read‖decomp interference (swap-in runs both streams at once).
        double dc_gbps[2], rc_gbps[2];
        std::cout << "\n=== concurrent read‖decomp (swap-in style) ===\n";
        for (int m = 0; m < 2; ++m) {
            ConcRates cr = bench_concurrent(pool, clctx, models[m].kv_hidden,
                                            test_file, file_bytes,
                                            d_gbps[m], rkv_gbps[m], 3.0);
            dc_gbps[m] = cr.d_gbps; rc_gbps[m] = cr.r_gbps;
            std::cout << "hidden " << models[m].kv_hidden << ": decomp "
                      << cr.d_gbps << " GB/s, kv read " << cr.r_gbps << " GB/s\n";
        }

        // Alloc-concurrent variant — the condition the hard-coded constants were
        // originally tuned under: the ALLOC pool churns chunk-sized fine-grained
        // clSVMAllocs (kv_alloc_worker) while decomp / reads run.
        double da_gbps[2], ra_gbps[2], rwa_gbps[2];
        if (clctx) {
            std::cout << "\n=== under alloc churn (alloc‖decomp, alloc‖read) ===\n";
            for (int m = 0; m < 2; ++m) {
                const size_t chunk = (size_t) TOKENS_PER_CHUNK * models[m].kv_hidden * sizeof(__fp16) * 2;
                AllocLoad churn(pool, clctx, chunk);

                churn.start(n_alloc_workers);
                da_gbps[m] = bench_decomp_gbps(pool, clctx, models[m].kv_hidden, 1024, passes);
                churn.end();

                churn.start(n_alloc_workers);
                ra_gbps[m] = bench_read_gbps(pool, CoreType::READ, test_file, file_bytes, chunk, passes);
                rwa_gbps[m] = bench_read_gbps(pool, CoreType::READ, test_file,
                                              (file_bytes / models[m].layer_bytes) * models[m].layer_bytes,
                                              (size_t) models[m].layer_bytes, 1);
                churn.end();

                std::cout << "hidden " << models[m].kv_hidden << ": decomp " << da_gbps[m]
                          << " GB/s, kv read " << ra_gbps[m]
                          << " GB/s, w read " << rwa_gbps[m] << " GB/s\n";
            }
        }

        // GPU-loaded variant: decomp + reads while the GPU streams memory flat-out,
        // approximating the fully-overlapped swap-in (prefill running concurrently).
        double dg_gbps[2], rg_gbps[2], rwg_gbps[2];
        if (clctx) {
            std::cout << "\n=== under GPU memory load (approximates overlapped prefill) ===\n";
            for (int m = 0; m < 2; ++m) {
                GpuLoad load; load.start(clctx, cldev);
                ConcRates cr = bench_concurrent(pool, clctx, models[m].kv_hidden,
                                                test_file, file_bytes,
                                                d_gbps[m], rkv_gbps[m], 3.0);
                rwg_gbps[m] = bench_read_gbps(pool, CoreType::READ, test_file,
                                              (file_bytes / models[m].layer_bytes) * models[m].layer_bytes,
                                              (size_t) models[m].layer_bytes, 1);
                load.end();
                dg_gbps[m] = cr.d_gbps; rg_gbps[m] = cr.r_gbps;
                std::cout << "hidden " << models[m].kv_hidden << ": decomp " << cr.d_gbps
                          << " GB/s, kv read " << cr.r_gbps
                          << " GB/s, w read " << rwg_gbps[m] << " GB/s\n";
            }
        }

        // Full swap-in emulation: alloc churn + GPU memory load + read‖decomp all
        // at once — every stressor of the fully-overlapped swap-in except the real
        // prefill kernels themselves.
        double df_gbps[2], rf_gbps[2], rwf_gbps[2];
        if (clctx) {
            std::cout << "\n=== full swap-in emulation (alloc + GPU + read‖decomp) ===\n";
            for (int m = 0; m < 2; ++m) {
                const size_t chunk = (size_t) TOKENS_PER_CHUNK * models[m].kv_hidden * sizeof(__fp16) * 2;
                AllocLoad churn(pool, clctx, chunk);
                GpuLoad load;
                churn.start(n_alloc_workers);
                load.start(clctx, cldev);
                ConcRates cr = bench_concurrent(pool, clctx, models[m].kv_hidden,
                                                test_file, file_bytes,
                                                da_gbps[m], ra_gbps[m], 3.0);
                rwf_gbps[m] = bench_read_gbps(pool, CoreType::READ, test_file,
                                              (file_bytes / models[m].layer_bytes) * models[m].layer_bytes,
                                              (size_t) models[m].layer_bytes, 1);
                load.end();
                churn.end();
                df_gbps[m] = cr.d_gbps; rf_gbps[m] = cr.r_gbps;
                std::cout << "hidden " << models[m].kv_hidden << ": decomp " << cr.d_gbps
                          << " GB/s, kv read " << cr.r_gbps
                          << " GB/s, w read " << rwf_gbps[m] << " GB/s\n";
            }
        }

        auto print_derived = [&](const char * tag, const double * D, const double * Rkv, const double * Rw) {
            std::cout << "\n=== derived scheduling constants (" << tag << ") ===\n";
            for (int m = 0; m < 2; ++m) {
                const ModelShape & M = models[m];
                const double chunk_bytes = (double) TOKENS_PER_CHUNK * M.kv_hidden * sizeof(__fp16) * 2;
                const double d_chunks    = D[m]   * 1e9 / chunk_bytes;  // decompressed /s
                const double r_chunks    = Rkv[m] * 1e9 / chunk_bytes;  // store-read /s

                const double balance = (double) M.layer_bytes / (Rw[m] * 1e9) * d_chunks;
                const double ratio_frac = d_chunks / (d_chunks + r_chunks);   // planner split
                const double ratio_rate = r_chunks / d_chunks;                // reads per decomp

                std::cout << M.name << ":\n"
                          << "  per_layer_balance = " << (int) (balance + 0.5)
                          << "   (hard-coded " << M.hard_balance << ")\n"
                          << "  decomp_load_ratio (equal-finish comp fraction) = " << ratio_frac
                          << "   (hard-coded " << hard_ratio << ")\n"
                          << "  [reads-per-decomp rate R/D = " << ratio_rate << "]\n";
            }
        };
        if (clctx) {
            mzcache::device_profile prof;
            prof.compression      = mzcache::compression_name();
            prof.soc              = soc_str;
            prof.decomp_gbps      = (df_gbps[0] + df_gbps[1]) / 2.0;
            prof.kv_read_gbps     = (rf_gbps[0] + rf_gbps[1]) / 2.0;
            prof.weight_read_gbps = (rwf_gbps[0] + rwf_gbps[1]) / 2.0;
            const std::string profile_path = mzcache::device_profile_path();
            if (mzcache::save_device_profile(profile_path, prof)) {
                std::cout << "\ndevice profile written to ./" << profile_path
                          << "  (D " << prof.decomp_gbps << ", R_kv " << prof.kv_read_gbps
                          << ", R_w " << prof.weight_read_gbps << " GB/s)\n"
                          << "mzcache runs in this directory will now derive "
                          << "per_layer_balance / decomp_load_ratio from it.\n";
            } else {
                std::cerr << "ERROR: failed to write " << profile_path << "\n";
            }
        } else {
            std::cerr << "\nWARNING: no OpenCL context - profile NOT written "
                      << "(values would not be runtime-faithful)\n";
        }

        print_derived("isolated", d_gbps, rkv_gbps, rw_gbps);
        print_derived("concurrent", dc_gbps, rc_gbps, rw_gbps);
        if (clctx) print_derived("under alloc churn", da_gbps, ra_gbps, rwa_gbps);
        if (clctx) print_derived("under GPU load", dg_gbps, rg_gbps, rwg_gbps);
        if (clctx) print_derived("full swap-in emulation", df_gbps, rf_gbps, rwf_gbps);

    }

    return 0;
}
