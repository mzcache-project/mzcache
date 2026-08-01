#pragma once

#include <iostream>
#include <vector>
#include "ggml.h"


#define ANSI_COLOR_RESET   "\033[0m"
#define ANSI_COLOR_RED     "\033[1;31m"
#define ANSI_COLOR_GREEN   "\033[1;32m"
#define ANSI_COLOR_YELLOW  "\033[1;33m"
#define ANSI_COLOR_CYAN    "\033[1;36m"
#define ANSI_COLOR_MAGENTA "\033[1;35m"

#define MZ_LOG_INFO(fmt, ...)                                           \
    do {                                                                 \
        std::fprintf(                                                   \
            stdout,                                                     \
            ANSI_COLOR_YELLOW "[MZCACHE][INFO] " fmt ANSI_COLOR_RESET "\n", \
            ##__VA_ARGS__                                               \
        );                                                               \
    } while (0)

#define MZ_LOG_OK(msg)      std::cout << ANSI_COLOR_GREEN   << "[MZCACHE][OK] "      << msg << ANSI_COLOR_RESET << std::endl
#define MZ_LOG_WARN(msg)    std::cout << ANSI_COLOR_MAGENTA << "[MZCACHE][WARNING] " << msg << ANSI_COLOR_RESET << std::endl

#define MZ_LOG_ERROR(fmt, ...)                                    \
    do {                                                           \
        std::fprintf(stderr,                                       \
            ANSI_COLOR_RED "[MZCACHE][ERROR] " fmt ANSI_COLOR_RESET "\n", \
            ##__VA_ARGS__);                                        \
    } while(0)

#define MZ_LOG_DEBUG(msg)   std::cout << ANSI_COLOR_CYAN    << "[MZCACHE][DEBUG] "   << msg << ANSI_COLOR_RESET << std::endl

#define MZ_TIME_START(label) \
    auto label##_start = std::chrono::high_resolution_clock::now();


#define MZ_TIME_END(label) \
    auto label##_end = std::chrono::high_resolution_clock::now(); \
    { \
        double ms = std::chrono::duration<double, std::milli>(label##_end - label##_start).count(); \
        std::cout << ANSI_COLOR_GREEN << "[MZCACHE][Timing] " << #label << " took " << ms << " ms" << ANSI_COLOR_RESET << std::endl; \
    }


////////////////////////////// hs note : required constant definitions

#define TOKENS_PER_CHUNK 256
#define MAX_CHUNKS_PER_TENSOR 256 // max token length = 256 * 256 = 65536

#define MAX_NUM_LAYERS 40 // probably also needs a definition in llama_model.cpp



////////////////////////////// hs note : the structures below are required for computation in llama.cpp; keep them as decoupled as possible from mzcache's own functionality.

    
struct ggml_tensor_ptrs {
    ggml_tensor * c[MAX_CHUNKS_PER_TENSOR]{};
};

struct chunk_ptrs { // llama only references this via pointer, so it must be malloc'ed externally, wired up, and freed by us. -> for now it is stored in extra in ggml_opencl.
    void * c[MAX_CHUNKS_PER_TENSOR]{};
};


////////////////////////////// hs note : the structures below are for managing KV inside mzcache.

enum class KVstate : uint8_t {
    Raw,         // Raw data (not compressed, not stored to SSD)
    Compressed,  // Compressed data (not stored)
    Stored       // Raw data stored in file (not compressed)
};


// 1. a raw chunk needs only one pointer. size is currently fixed at 256 x 1024 x 2 bytes but may vary per model.
struct raw_chunk { 
    void * c;
}; 

struct compressed_chunk {
    int stack_id;
    int offset;
    size_t comp_size;

    // plus variables dependent on the compression algorithm..
    // e.g. flexgen requires min/max arrays.
    __fp16* mins;
    __fp16* maxs;

};

struct stored_chunk {
    uint64_t file_offset; // offset in the file
    size_t stored_size; // size of the stored chunk, maybe 256 * 1024 * 2 bytes
};

struct array_of_chunks {
    raw_chunk raw_chunks[MAX_CHUNKS_PER_TENSOR]; // raw chunks
    compressed_chunk compressed_chunks[MAX_CHUNKS_PER_TENSOR]; // compressed chunks
    stored_chunk stored_chunks[MAX_CHUNKS_PER_TENSOR]; // stored chunks
};


struct layer_kv_chunks {
    int layer_idx; // 1. from 0 to n_layer -1

    
    // 2. file where stored chunks are written
    // llama_file * file; // or std::string file_path; -> need a struct along these lines for mzcache layerwise file i/o

    KVstate states[MAX_CHUNKS_PER_TENSOR]; // 3. states of the chunks, K,V chunks at the same position must always keep the same state.

    array_of_chunks k_chunks;
    array_of_chunks v_chunks;

};


// Structures for managing weight tensors in the cache

// std::vector<std::vector<int>> layer_idx_to_tensor_mapping; // layer_idx -> tensor_idx mapping
