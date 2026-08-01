
#pragma once
#include <cstdint>
#include <vector>
#include <tuple>
#include <iostream>
#include <algorithm>

/////////////////////

class ThreadPool; // Forward declaration

void flexgen_compress(
    const __fp16* input,
    int           hidden_dim,
    uint8_t*      output,
    __fp16*       mins,
    __fp16*       maxs,

    ThreadPool*    pool,
    int            num_threads
);

void flexgen_compress_8bit(
    const __fp16* input,
    int           hidden_dim,
    uint8_t*      output,
    __fp16*       mins,
    __fp16*       maxs,

    ThreadPool*    pool,
    int            num_threads
);

void flexgen_decompress(
    const uint8_t* input,
    int            hidden_dim,
    __fp16*        output,
    __fp16*        mins,
    __fp16*        maxs,
    
    ThreadPool*    pool,
    int            num_threads
);

void flexgen_decompress_8bit(
    const uint8_t* input,
    int            hidden_dim,
    __fp16*        output,
    __fp16*        mins,
    __fp16*        maxs
);

void flexgen_decompress_single_thread(
    const uint8_t* input,
    int            hidden_dim,
    __fp16*        output,
    __fp16*        mins,
    __fp16*        maxs
);



///////////////////

struct EncoderMeta;  // forward declaration from quant_cpu_raw.h
struct EncoderOutput; // forward declaration from quant_cpu_raw.h

struct CdfRaw; // forward declaration from cachegen_utils.h

struct layer_kv_chunks; // forward declaration from mzcache_types.h



void encode_meta_function(
    const layer_kv_chunks& layer_chunks,
    int total_tokens,            // == T
    int hidden_dim,              // == C
    EncoderMeta& enc_meta_cpu
);

// encode_function returns bytestream of the compressed tensor
void encode_function(
    const __fp16* k_l,
    const __fp16* v_l,
    int n_tokens,
    const EncoderMeta& enc_meta_cpu,
    EncoderOutput& enc_out_cpu,
    bool log
);

void encode_meta_function_new(
    const layer_kv_chunks& layer_chunks,
    int total_tokens,            // == T
    int hidden_dim,              // == C
    EncoderMeta* enc_meta_cpu
);
size_t encode_function_new(
    const __fp16* input,
    int hidden_dim,
    uint8_t*  output,

    std::vector<uint8_t>* lengths_arr,
    bool is_key,
    CdfRaw& cdf,
    uint16_t* byte_sums_arr
);


void decode_function(
    const EncoderOutput& enc_out_cpu,
    const EncoderMeta&   enc_meta_cpu,
    const layer_kv_chunks& layer_chunks,
    int total_tokens,            // == T (n_past)
    int tokens_per_chunk         // == TOKENS_PER_CHUNK (e.g. 256)
);

void decode_function_one_chunk(
    const EncoderOutput&    enc_out_cpu,
    const EncoderMeta&      enc_meta_cpu,
    const layer_kv_chunks&  layer_chunks,
    int total_tokens,             // == T (n_past), always a multiple of 256 (already padded)
    int tokens_per_chunk,         // == TOKENS_PER_CHUNK (256)
    int chunk_idx                 // index of the "dst raw chunk" to decode (0-based)
);


void decode_function_new(
    const uint8_t* input,
    int hidden_dim,
    __fp16*        output,

    std::vector<uint8_t>* lengths_arr,
    bool is_key,
    CdfRaw& cdf,
    const float* m_base,
    uint16_t* byte_sums_arr
);
