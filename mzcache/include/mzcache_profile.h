#pragma once

// One-time on-device profiling support.
//
// examples/mz_device_profile measures the device-level throughputs (decomp,
// KV-chunk read, weight-layer read) under the "full swap-in emulation"
// condition and writes them to DEVICE_PROFILE_FILE in the CWD. The engine
// (mzcache_core) refuses to start without that file and derives the two
// scheduling constants from it plus the model shape:
//
//   per_layer_balance  = weight_layer_bytes / R_weight * D_chunks
//   decomp_load_ratio  = D / (D + R_kv)          (model-independent)
//
// Everything compression-specific lives in mzcache_profile.cpp, which is
// compiled inside libmzcache with the MZCACHE_COMPRESS_* define (the define
// is PRIVATE to the target, so callers use these runtime helpers instead).

#include "mzcache_threadpool.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mzcache {

// Per-compression profile filename: "mzcache_device_profile_<COMPRESSION>.txt"
// (e.g. ..._FLEXGEN_8BIT.txt, ..._CACHEGEN.txt) so different algorithms'
// profiles never overwrite each other in the shared working directory.
std::string device_profile_path();

// Supported chipsets. Core placement and the legacy scheduling constants
// depend on the SoC — its CPU topology (which logical CPU is a prime/perf/eff
// core is fixed by the SoC's device tree) and its Adreno generation — not on
// the phone vendor, so any device with the same SoC takes the same paths here.
// Per-device differences (memory bandwidth, clocks, storage) are corrected by
// the offline profile instead.
enum class SocClass {
    SM8750,   // Snapdragon 8 Elite: 2 prime (cpu6-7) + 6 perf (cpu0-5), Adreno 830
    SM8650,   // Snapdragon 8 Gen 3: 1 prime (cpu7) + 5 perf (cpu2-6) + 2 eff (cpu0-1), Adreno 750
    UNKNOWN,
};

// Raw SoC identifier: ro.soc.model (e.g. "SM8750", "SM8650-AB"), falling back
// to the platform codename in ro.board.platform on kernels without it.
// Empty off-Android.
std::string soc_model();
SocClass    classify_soc(const std::string & soc_model);
const char * soc_name(SocClass soc);  // human-readable, for logs

struct device_profile {
    std::string compression;        // must match compression_name() at load
    std::string soc;                // provenance: ro.soc.model of the profiled device
    double decomp_gbps      = 0.0;  // raw fp16 bytes produced per second
    double kv_read_gbps     = 0.0;  // store-chunk O_DIRECT sequential read
    double weight_read_gbps = 0.0;  // layer-sized O_DIRECT sequential read
};

// Compression backend selected at build time ("FLEXGEN", "FLEXGEN_8BIT", ...).
const char * compression_name();

// Offline profiling implemented for the FLEXGEN variants only; CACHEGEN/NONE
// builds keep their legacy hard-coded constants and skip the profile gate.
bool profiling_supported();

// The build's decompression kernel on one tensor chunk (hidden*TOKENS_PER_CHUNK
// elems), and its compressed-input size — used by the profiler benchmark.
void   profile_decomp_chunk(const uint8_t * in, int hidden_dim, __fp16 * out,
                            __fp16 * mins, __fp16 * maxs);
size_t profile_comp_bytes(int elems);

// The DECOMP/READ/ALLOC/FREE core sets mzcache_core uses on this device —
// shared so the profiler always measures with the runtime's core placement.
// UNKNOWN falls back to the SM8650 placement (only valid core IDs on any
// 8-core part; callers should warn).
std::map<CoreType, std::vector<int>> default_core_configs(SocClass soc);

bool load_device_profile(const std::string & path, device_profile & out);
bool save_device_profile(const std::string & path, const device_profile & p);

int   derive_per_layer_balance(const device_profile & p, int64_t weight_layer_bytes, int kv_hidden_dim);
float derive_decomp_load_ratio(const device_profile & p);

} // namespace mzcache
