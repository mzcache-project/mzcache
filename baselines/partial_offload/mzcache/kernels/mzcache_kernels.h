
#pragma once
#include <cstdint>

class ThreadPool; // Forward declaration

void compress_fp16_to_4bit(
    const __fp16* input,
    int           hidden_dim,
    uint8_t*      output,
    __fp16*       mins,
    __fp16*       maxs,

    ThreadPool&    pool,
    int            num_threads
);

void decompress_4bit_to_fp16(
    const uint8_t* input,
    int            hidden_dim,
    __fp16*        output,
    __fp16*        mins,
    __fp16*        maxs,
    
    ThreadPool&    pool,
    int            num_threads
);

void decompress_4bit_to_fp16_single_thread(
    const uint8_t* input,
    int            hidden_dim,
    __fp16*        output,
    __fp16*        mins,
    __fp16*        maxs
);