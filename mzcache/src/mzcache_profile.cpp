#include "mzcache_profile.h"
#include "mzcache_types.h"
#include "mzcache_kernels.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef __ANDROID__
#include <sys/system_properties.h>  // __system_property_get (not transitively included on NDK r25)
#endif

#if defined(MZCACHE_COMPRESS_CACHEGEN)
#include "cachegen_utils.h"   // EncoderMeta, encode/decode_function_new
#include <map>
#include <mutex>
#include <random>
#include <vector>
#endif

namespace mzcache {

const char * compression_name() {
#if defined(MZCACHE_COMPRESS_FLEXGEN)
    return "FLEXGEN";
#elif defined(MZCACHE_COMPRESS_FLEXGEN_8BIT)
    return "FLEXGEN_8BIT";
#elif defined(MZCACHE_COMPRESS_CACHEGEN)
    return "CACHEGEN";
#else
    return "NONE";
#endif
}

std::string device_profile_path() {
    return std::string("mzcache_device_profile_") + compression_name() + ".txt";
}

bool profiling_supported() {
#if defined(MZCACHE_COMPRESS_FLEXGEN) || defined(MZCACHE_COMPRESS_FLEXGEN_8BIT) \
    || defined(MZCACHE_COMPRESS_CACHEGEN)
    return true;
#else
    return false;
#endif
}

#if defined(MZCACHE_COMPRESS_CACHEGEN)
// CacheGen decode benchmark support. Unlike FLEXGEN (dequantization runs on any
// bytes), CacheGen decode needs a valid entropy-coded bitstream, so we build one
// once per hidden dim from synthetic data via the real encoder path
// (encode_meta_function_new + encode_function_new) and then decode_function_new()
// it — exactly like kv_decomp_worker — so the profiler measures real CacheGen
// decode throughput. The built state is shared read-only across DECOMP threads,
// the same way the runtime shares one EncoderMeta per layer across concurrent
// chunk decodes.
namespace {
struct CgDecodeState {
    static constexpr int NB = TOKENS_PER_CHUNK / MAX_TOKENS_PER_CHUNK;
    EncoderMeta          meta;
    std::vector<__fp16>  raw_k;              // one synthetic chunk of raw K
    std::vector<uint8_t> enc_k;              // its encoded bitstream
    std::vector<uint8_t> len_k[NB];
    uint16_t             bs_k[NB] = {0};
    layer_kv_chunks      layer{};            // wires raw pointers for encode_meta
};

CgDecodeState * build_cg_state(int hidden) {
    auto * s = new CgDecodeState();
    const int    C     = hidden;
    const size_t elems = (size_t) TOKENS_PER_CHUNK * C;

    s->raw_k.resize(elems);
    std::vector<__fp16> raw_v(elems);
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 0.3f);  // realistic-ish KV magnitudes
    for (size_t i = 0; i < elems; ++i) { s->raw_k[i] = (__fp16) nd(rng); raw_v[i] = (__fp16) nd(rng); }

    // One-chunk layer (T = TOKENS_PER_CHUNK) is enough to build the CDF/max tables.
    s->layer.k_chunks.raw_chunks[0].c = s->raw_k.data();
    s->layer.v_chunks.raw_chunks[0].c = raw_v.data();
    encode_meta_function_new(s->layer, TOKENS_PER_CHUNK, C, &s->meta);

    s->enc_k.resize(elems * sizeof(__fp16));  // raw size is a safe encoded-output bound
    encode_function_new(s->raw_k.data(), C, s->enc_k.data(), s->len_k, /*is_key=*/true,
                        s->meta.cdf_key, s->bs_k);
    return s;
}

CgDecodeState * cg_state(int hidden) {
    static std::mutex mu;
    static std::map<int, CgDecodeState *> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(hidden);
    if (it != cache.end()) return it->second;
    return cache[hidden] = build_cg_state(hidden);
}
} // namespace
#endif

void profile_decomp_chunk(const uint8_t * in, int hidden_dim, __fp16 * out,
                          __fp16 * mins, __fp16 * maxs) {
#if defined(MZCACHE_COMPRESS_FLEXGEN)
    flexgen_decompress_single_thread(in, hidden_dim, out, mins, maxs);
#elif defined(MZCACHE_COMPRESS_FLEXGEN_8BIT)
    flexgen_decompress_8bit(in, hidden_dim, out, mins, maxs);
#elif defined(MZCACHE_COMPRESS_CACHEGEN)
    (void) in; (void) mins; (void) maxs;  // CacheGen uses the pre-built valid stream
    CgDecodeState * s = cg_state(hidden_dim);
    decode_function_new(s->enc_k.data(), hidden_dim, out, s->len_k, /*is_key=*/true,
                        s->meta.cdf_key, s->meta.max_key.data(), s->bs_k);
#else
    (void) in; (void) hidden_dim; (void) out; (void) mins; (void) maxs;
#endif
}

size_t profile_comp_bytes(int elems) {
#if defined(MZCACHE_COMPRESS_FLEXGEN)
    return (size_t) elems / 2;   // 4 bits per element
#elif defined(MZCACHE_COMPRESS_CACHEGEN)
    return (size_t) elems;       // input buffer is unused by the CacheGen path
#else
    return (size_t) elems;       // 8 bits per element
#endif
}

std::string soc_model() {
#ifdef __ANDROID__
    char buf[PROP_VALUE_MAX] = {0};
    if (__system_property_get("ro.soc.model", buf) > 0 && buf[0]) return buf;
    // Kernels predating ro.soc.model (Android < 12) expose the SoC codename.
    if (__system_property_get("ro.board.platform", buf) > 0 && buf[0]) return buf;
#endif
    return "";
}

SocClass classify_soc(const std::string & m) {
    // Prefix match: vendor bins carry suffixes (SM8750-AB/-AC "for Galaxy",
    // SM8650-AB, ...) but share the topology and GPU of the base part.
    if (m.rfind("SM8750", 0) == 0 || m == "sun")       return SocClass::SM8750;
    if (m.rfind("SM8650", 0) == 0 || m == "pineapple") return SocClass::SM8650;
    return SocClass::UNKNOWN;
}

const char * soc_name(SocClass soc) {
    switch (soc) {
        case SocClass::SM8750: return "Snapdragon 8 Elite (SM8750)";
        case SocClass::SM8650: return "Snapdragon 8 Gen 3 (SM8650)";
        default:               return "unknown SoC";
    }
}

std::map<CoreType, std::vector<int>> default_core_configs(SocClass soc) {
#if defined(MZCACHE_COMPRESS_FLEXGEN) || defined(MZCACHE_COMPRESS_FLEXGEN_8BIT)
    if (soc == SocClass::SM8750) {
        return {
            { CoreType::ALLOC,  {0, 1, 2, 3, 4, 5} },
            { CoreType::DECOMP, {6, 7} },
            { CoreType::READ,   {0, 1} },
            { CoreType::FREE,   {0} },
        };
    }
    return {
        { CoreType::ALLOC,  {5, 7} },
        { CoreType::DECOMP, {3, 4, 6} },
        { CoreType::READ,   {2, 3} },
        { CoreType::FREE,   {0} },
    };
#else // CACHEGEN / NONE
    if (soc == SocClass::SM8750) {
        return {
            { CoreType::ALLOC,  {4, 5} },
            { CoreType::DECOMP, {0, 1, 2, 3, 6, 7} },
            { CoreType::READ,   {4, 5} },
            { CoreType::FREE,   {0} },
        };
    }
    return {
        { CoreType::ALLOC,  {5, 7} },
        { CoreType::DECOMP, {3, 4, 6} },
        { CoreType::READ,   {2, 3} },
        { CoreType::FREE,   {0} },
    };
#endif
}

bool load_device_profile(const std::string & path, device_profile & out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const std::string val = line.substr(eq + 1);
        if      (key == "compression")      out.compression      = val;
        else if (key == "soc")              out.soc              = val;
        else if (key == "decomp_gbps")      out.decomp_gbps      = atof(val.c_str());
        else if (key == "kv_read_gbps")     out.kv_read_gbps     = atof(val.c_str());
        else if (key == "weight_read_gbps") out.weight_read_gbps = atof(val.c_str());
    }
    return out.decomp_gbps > 0.0 && out.kv_read_gbps > 0.0 && out.weight_read_gbps > 0.0;
}

bool save_device_profile(const std::string & path, const device_profile & p) {
    std::ofstream f(path);
    if (!f.is_open()) return false;
    f << "# mzcache device profile — throughputs under the full swap-in emulation\n"
      << "# (see examples/mz_device_profile). Delete this file to force re-profiling.\n"
      << "version=1\n"
      << "compression=" << p.compression << "\n"
      << "soc=" << p.soc << "\n"
      << "decomp_gbps=" << p.decomp_gbps << "\n"
      << "kv_read_gbps=" << p.kv_read_gbps << "\n"
      << "weight_read_gbps=" << p.weight_read_gbps << "\n";
    return true;
}

int derive_per_layer_balance(const device_profile & p, int64_t weight_layer_bytes, int kv_hidden_dim) {
    const double chunk_bytes = (double) TOKENS_PER_CHUNK * kv_hidden_dim * sizeof(__fp16) * 2;
    const double d_chunks    = p.decomp_gbps * 1e9 / chunk_bytes;
    const double layer_s     = (double) weight_layer_bytes / (p.weight_read_gbps * 1e9);
    return (int) std::lround(layer_s * d_chunks);
}

float derive_decomp_load_ratio(const device_profile & p) {
    // equal-finish split between the decompress and store-read swap-in streams
    return (float) (p.decomp_gbps / (p.decomp_gbps + p.kv_read_gbps));
}

} // namespace mzcache
