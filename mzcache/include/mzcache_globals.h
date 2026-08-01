#pragma once

// [mzcache] Centralized declarations for all mzCache global variables.
// Definitions are in mzcache/src/mzcache_globals.cpp unless noted otherwise.

#include "mzcache_types.h"
#include "ggml-backend.h"

#include <vector>

// Weight context globals (defined in mzcache_globals.cpp)
extern ggml_context *              mz_weight_ctx;
extern ggml_backend_buffer_type_t  mz_weight_buft;
extern ggml_backend_buffer_t       mz_weight_buf;

// Per-layer weight metadata (defined in mzcache_globals.cpp)
extern int                    layer_0_tensor_idx;
extern int                    num_tensors_per_layer;
extern int                    num_bytes_per_layer;
extern int                    padded_num_bytes_per_layer;
extern std::vector<size_t>    layer_weight_sizes;
extern struct ggml_tensor *   layer_first_tensor[MAX_NUM_LAYERS];

// SVM chunk globals (defined in ggml-opencl.cpp)
extern int                           last_chunk_idx;
extern std::vector<chunk_ptrs *>     g_svm_chunk_ptrs;
