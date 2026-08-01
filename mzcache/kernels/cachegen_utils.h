#pragma once
#include <cstdint>
#include <cstddef>
#include <utility>
#include <vector>

struct array_of_chunks;
struct layer_kv_chunks;

#define MAX_TOKENS_PER_CHUNK 16

#define ASSERT_MSG(cond, msg) \
    do { if (!(cond)) { fprintf(stderr, "%s\n", msg); assert(cond); } } while(0)

// ===== Per-layer bin rules (same constants as the original code) =====
extern int key_first_layers;
extern int key_second_layers;
extern int key_third_layers;

extern int key_first_bins;
extern int key_second_bins;
extern int key_third_bins;

extern int value_first_layers;
extern int value_first_bins;
extern int value_second_bins;

// ===== bins construction (raw pointer; caller must delete[]) =====
int* make_key_bins_cpp  (int n_layers);
int* make_value_bins_cpp(int n_layers);

// ===== Quantization result (all raw pointers) =====
struct QuantResultRaw {
    int8_t* q;     // [L*T*C] (layout: idx = c + C*(t + T*l))
    float*  max1;  // [L*T]   (layout: idx = t + T*l)
    int     L, T, C;
};

struct CdfRaw {
    std::vector<uint16_t> data; // [C * bins_p1]
    int C = 0;
    int bins_p1 = 0;
};

struct EncodeCPUResult {
    std::vector<uint8_t> bytestream;
    std::vector<uint8_t> lengths;  // [C]
    int t_chunk = 0;
};

// Encoder output used as input to the CPU decoder (all kept in CPU memory)
struct EncoderOutput {
    // chunked bytestreams
    std::vector<EncodeCPUResult> chunks_key;
    std::vector<EncodeCPUResult> chunks_value;

    // Total encoded bytes (key + value)
    size_t bytes_total = 0;
    size_t bytes_key   = 0;
    size_t bytes_value = 0;
};

struct EncoderMeta {
    CdfRaw cdf_key;    // [C, bins+1]
    CdfRaw cdf_value;  // [C, bins+1]

    // dequant scales (max1): [T]
    std::vector<float> max_key;     // size == T
    std::vector<float> max_value;   // size == T

    int T = 0;
    int C = 0;
};

struct DecodedQKVChunk {
    std::vector<int8_t> qk; // size == L * t_chunk * C
    std::vector<int8_t> qv; // size == L * t_chunk * C
    int t_chunk = 0;
};

std::vector<int8_t> decode_ntokens_cpu(
    const uint16_t* cdf,  // [L, C, bins_p1] as uint16
    int L, int C, int bins_p1,
    const EncodeCPUResult& chunk
);

// ===== Quantization (CPU, functionality-first scalar) =====
// Input: __fp16* x (layout: idx = c + C*(t + T*l)); L/T/C passed by the caller
// bins_L: int array of length L (bin count of each layer); q/max1 are returned new[]-allocated
QuantResultRaw quantize_from_bins_raw(
    const int*    bins_L, // [L]
    int           L,
    int           T,
    int           C,
    const __fp16* x       // [L*T*C] in C-fast order
);

QuantResultRaw quantize_from_bins_raw_chunked(
    const int*   bins_L,   // [L]
    int          L,        // usually 1
    int          T,        // total tokens (actual valid token count; if padded, pass T_pad as-is)
    int          C,        // hidden dim
    const array_of_chunks& chunks
);

// CPU CDF computation (implementation elsewhere; interface only)
//  - q: quantized input (int8) [L,T,C] contiguous (idx = c + C*(t + T*l))
//  - L, T, C: dimensions
//  - max_bins: bin count of the layer (e.g. 32/16). Used if the implementation needs it
CdfRaw cdf_calculate_cpu(const int8_t* q, int L, int T, int C, int max_bins);




// cdf: [L, C, bins+1], int16
// q  : [L, t_chunk, C], int8
// Returns: (bytestream, lengths[L*C])
EncodeCPUResult encode_ntokens_cpu(
    const uint16_t* cdf, int L, int C, int bins_p1,
    const int8_t*  q_chunk, int t_chunk
);

int encode_ntokens_cpu_new(
    const uint16_t* cdf, int C, int lp,
    const int8_t*  q_chunk, int t_chunk,
    uint8_t* bytestream, std::vector<uint8_t>& lengths
);

int encode_ntokens_cpu_new_range(
    const uint16_t* cdf, int C, int lp,
    const int8_t*  q_chunk, int t_chunk,
    uint8_t* bytestream, std::vector<uint8_t>& lengths
);


void dequantize_cpu_into(
    const int8_t* q,          // [L,T,C]
    const float*  max1,       // [L,T]
    int L, int T, int C,
    int bins,                 // e.g. 32
    __fp16* out               // [L,T,C] to write into
);


void dequantize_cpu_into_new(
    const int8_t* q,          // [L,T,C]
    const float*  max1,       // [L,T]
    int T, int C,
    int bins,                 // e.g. 32
    __fp16* out               // [L,T,C] to write into
);

void dequantize_cpu_into_new_neon(
    const int8_t* __restrict q,   // [T,C]
    const float*  __restrict max1,// [T]
    int T, int C,
    int bins,
    __fp16* __restrict out        // [T,C]
);



std::vector<int8_t> decode_ntokens_cpu_new(
    const uint16_t* cdf,  // [L, C, bins_p1] as int16_t
    int C, int bins_p1,
    const uint8_t* input, std::vector<uint8_t>& lengths
);

std::vector<int8_t> decode_ntokens_cpu_new_range(
    const uint16_t* cdf,  // [L, C, bins_p1] as int16_t
    int C, int bins_p1,
    const uint8_t* input, std::vector<uint8_t>& lengths
);




std::vector<int8_t> decode_ntokens_cpu_new_simd_range(
    const uint16_t* cdf,  // [C, bins_p1]
    int C, int bins_p1,
    const uint8_t* input, std::vector<uint8_t>& lengths
);


std::vector<int8_t> decode_ntokens_cpu_new_range_pairs(
    const uint16_t* cdf,  // [C, bins_p1]
    int C, int bins_p1,
    const uint8_t* input, std::vector<uint8_t>& lengths
);

// Memory-release helper (optional)
inline void free_quant(QuantResultRaw& r) {
    delete[] r.q;   r.q   = nullptr;
    delete[] r.max1; r.max1 = nullptr;
    r.L = r.T = r.C = 0;
}
