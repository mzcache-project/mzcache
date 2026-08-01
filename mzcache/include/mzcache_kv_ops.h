#pragma once

// [mzcache] KV cache chunk helper functions.
// Extracted from llama-kv-cache-unified.cpp to separate mzCache chunk logic.

#include "mzcache_globals.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <utility>
#include <vector>

// Forward declarations
class llama_io_write_i;
class llama_io_read_i;

namespace mzcache {

// ---  KV tensor creation in chunked mode ---

// Create the base (chunk 0) KV tensors for a single layer in the constructor.
// Chunks beyond index 0 are materialized lazily at runtime by grow_layer_chunk()
// as the cache actually fills (state restore / decode), so a large n_ctx does
// not pre-commit SVM for tokens that are never reached.
// Sets ggml_tensor_ptrs on k and v.
void create_chunked_kv_tensors(
    ggml_context * ctx,
    ggml_type      type_k,
    ggml_type      type_v,
    uint32_t       n_embd_k_gqa,
    uint32_t       n_embd_v_gqa,
    uint32_t       kv_size,
    uint32_t       il,
    ggml_tensor *& k_out,
    ggml_tensor *& v_out,
    std::vector<ggml_tensor_ptrs> & layers_k_chunks,
    std::vector<ggml_tensor_ptrs> & layers_v_chunks);

// Materialize chunk `idx` (> 0) of layer `il` at runtime: allocates a zeroed
// SVM K+V pair, creates the chunk tensors in `ctx`, and publishes the pointers
// in layers_*_chunks and the GPU-visible g_svm_chunk_ptrs tables. The kv-cache
// keeps chunk growth in lockstep across layers (see
// llama_kv_cache_unified::mz_grow_chunks).
void grow_layer_chunk(
    ggml_context * ctx,
    uint32_t       il,
    uint32_t       idx,
    uint32_t       n_embd_k_gqa,
    std::vector<ggml_tensor_ptrs> & layers_k_chunks,
    std::vector<ggml_tensor_ptrs> & layers_v_chunks);

// Register SVM chunk offsets after buffer allocation.
void register_kv_chunk_offsets(
    uint32_t       n_layer,
    uint32_t       kv_size,
    const std::vector<ggml_tensor_ptrs> & layers_k_chunks,
    const std::vector<ggml_tensor_ptrs> & layers_v_chunks,
    std::vector<ggml_backend_buffer_ptr> & bufs);

// --- State I/O: chunk-wise write ---

// Write K data chunk-wise for given cell_ranges.
void state_write_k_chunked(
    llama_io_write_i & io,
    uint32_t il,
    uint64_t k_size_row,
    const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges);

// Write V data chunk-wise for given cell_ranges.
void state_write_v_chunked(
    llama_io_write_i & io,
    uint32_t il,
    uint64_t v_size_row,
    const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges);

// --- State I/O: chunk-wise read ---

// Read K data chunk-wise into chunk tensors.
void state_read_k_chunked(
    llama_io_read_i & io,
    uint32_t il,
    uint64_t k_size_row,
    uint32_t cell_count,
    uint32_t head,
    const std::vector<ggml_tensor_ptrs> & layers_k_chunks);

// Read V data chunk-wise into chunk tensors.
void state_read_v_chunked(
    llama_io_read_i & io,
    uint32_t il,
    uint64_t v_size_row,
    uint32_t cell_count,
    uint32_t head,
    const std::vector<ggml_tensor_ptrs> & layers_v_chunks);

} // namespace mzcache
