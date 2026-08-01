#include "mzcache_kv_ops.h"
#include "mzcache_globals.h"

#include "ggml.h"
#include "ggml-backend.h"

#include "llama-io.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>

#ifdef MZCACHE_USE_OPENCL
#include <CL/cl.h>
#include "ggml-opencl-extra.h" // g_opencl_context
#endif

namespace mzcache {

// ---------- chunk-wise I/O helper (shared by write_k, write_v) ----------

static void chunk_write(
    llama_io_write_i & io,
    int svm_chunk_slot, // index into g_svm_chunk_ptrs (2*il for K, 2*il+1 for V)
    uint64_t size_row,
    const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges)
{
    for (const auto & range : cell_ranges) {
        const int64_t start_token = range.first;
        const int64_t end_token   = range.second;

        if (start_token >= end_token) continue;

        const int first_chunk = (int)(start_token / TOKENS_PER_CHUNK);
        const int last_chunk  = (int)((end_token - 1) / TOKENS_PER_CHUNK);
        const size_t full_chunk_bytes = (size_t)TOKENS_PER_CHUNK * (size_t)size_row;

        for (int ci = first_chunk; ci <= last_chunk; ++ci) {
            void * chunk_ptr = g_svm_chunk_ptrs[svm_chunk_slot]->c[ci];

            size_t offset_bytes   = 0;
            size_t bytes_to_write = full_chunk_bytes;

            if (first_chunk == last_chunk) {
                offset_bytes   = (size_t)((start_token % TOKENS_PER_CHUNK) * size_row);
                bytes_to_write = (size_t)((end_token - start_token) * size_row);
            } else if (ci == first_chunk) {
                offset_bytes   = (size_t)((start_token % TOKENS_PER_CHUNK) * size_row);
                bytes_to_write = full_chunk_bytes - offset_bytes;
            } else if (ci == last_chunk) {
                // tokens in the last chunk = end_token - last_chunk*TOKENS_PER_CHUNK.
                // (end_token % TOKENS_PER_CHUNK) is wrong when end_token is an exact
                // multiple of TOKENS_PER_CHUNK — it yields 0 and drops a full chunk.
                bytes_to_write = (size_t)((end_token - (int64_t)last_chunk * TOKENS_PER_CHUNK) * size_row);
            }

            if (bytes_to_write == 0) continue;

            const uint8_t * src_ptr = (const uint8_t *)chunk_ptr + offset_bytes;
            io.write(src_ptr, bytes_to_write);
        }
    }
}

// ---------- chunk-wise I/O helper (shared by read_k, read_v) ----------

static void chunk_read(
    llama_io_read_i & io,
    int svm_chunk_slot, // index into g_svm_chunk_ptrs (2*il for K, 2*il+1 for V)
    uint64_t size_row,
    uint32_t cell_count,
    uint32_t head)
{
    const int64_t start_token = head;
    const int64_t end_token   = head + cell_count;

    const int first_chunk = start_token / TOKENS_PER_CHUNK;
    const int last_chunk  = (end_token - 1) / TOKENS_PER_CHUNK;
    const size_t full_chunk_bytes = TOKENS_PER_CHUNK * size_row;

    for (int ci = first_chunk; ci <= last_chunk; ++ci) {
        // Chunks are SVM (host-visible) memory; go through the pointer table —
        // like chunk_write — so runtime-grown chunks and chunks reallocated by
        // a swapin cycle are both covered.
        void * chunk_ptr = g_svm_chunk_ptrs[svm_chunk_slot]->c[ci];
        if (!chunk_ptr) {
            throw std::runtime_error("chunk_read: chunk " + std::to_string(ci) + " is not resident");
        }

        size_t offset_bytes  = 0;
        size_t bytes_to_read = full_chunk_bytes;

        if (first_chunk == last_chunk) {
            offset_bytes  = (size_t)((start_token % TOKENS_PER_CHUNK) * size_row);
            bytes_to_read = (size_t)(cell_count * size_row);
        } else if (ci == first_chunk) {
            offset_bytes  = (size_t)((start_token % TOKENS_PER_CHUNK) * size_row);
            bytes_to_read = full_chunk_bytes - offset_bytes;
        } else if (ci == last_chunk) {
            // See chunk_write: use end_token - last_chunk*TOKENS_PER_CHUNK, not
            // (end_token % TOKENS_PER_CHUNK), so an exact-multiple end_token reads
            // the full last chunk instead of 0 bytes (which misaligns the stream).
            bytes_to_read = (size_t)((end_token - (int64_t)last_chunk * TOKENS_PER_CHUNK) * size_row);
        }

        const void * data_ptr = io.read(bytes_to_read);
        memcpy((uint8_t *)chunk_ptr + offset_bytes, data_ptr, bytes_to_read);
    }
}

// ======================== Public API ========================

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
    std::vector<ggml_tensor_ptrs> & layers_v_chunks)
{
    uint32_t n_chunks = (kv_size + TOKENS_PER_CHUNK - 1) / TOKENS_PER_CHUNK;
    if (n_chunks > MAX_CHUNKS_PER_TENSOR) {
        throw std::runtime_error("kv cache size is too large for the number of chunks");
    }

    // Chunk 0: full 2D tensor (used as the "base" tensor for views).
    // Chunks 1.. are NOT created here: llama_kv_cache_unified::mz_grow_chunks
    // materializes them on demand as the cache fills, so n_ctx headroom for
    // generation costs no SVM up front.
    k_out = ggml_new_tensor_2d(ctx, type_k, n_embd_k_gqa, TOKENS_PER_CHUNK);
    v_out = ggml_new_tensor_2d(ctx, type_v, n_embd_v_gqa, TOKENS_PER_CHUNK);

    ggml_format_name(k_out, "cache_k_l%d", il);
    ggml_format_name(v_out, "cache_v_l%d", il);

    layers_k_chunks[il].c[0] = k_out;
    layers_v_chunks[il].c[0] = v_out;

    k_out->ggml_tensor_ptrs = &layers_k_chunks[il];
    v_out->ggml_tensor_ptrs = &layers_v_chunks[il];
}

void grow_layer_chunk(
    ggml_context * ctx,
    uint32_t       il,
    uint32_t       idx,
    uint32_t       n_embd_k_gqa,
    std::vector<ggml_tensor_ptrs> & layers_k_chunks,
    std::vector<ggml_tensor_ptrs> & layers_v_chunks)
{
#ifndef MZCACHE_USE_OPENCL
    (void) ctx; (void) il; (void) idx; (void) n_embd_k_gqa;
    (void) layers_k_chunks; (void) layers_v_chunks;
    throw std::runtime_error("grow_layer_chunk requires the OpenCL SVM backend");
#else
    if (idx == 0 || idx >= MAX_CHUNKS_PER_TENSOR) {
        throw std::runtime_error("grow_layer_chunk: chunk index out of range");
    }
    if (layers_k_chunks[il].c[idx] != nullptr) {
        return; // already materialized
    }

    const size_t chunk_elements = (size_t) TOKENS_PER_CHUNK * n_embd_k_gqa;
    const size_t chunk_bytes    = chunk_elements * sizeof(__fp16);

    // K and V as one SVM pair, exactly like kv_alloc_worker's swapin
    // reallocation (the compress path frees pairs when v == k + chunk).
    __fp16 * new_k = (__fp16 *) clSVMAlloc(
        g_opencl_context,
        CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
        chunk_bytes * 2, 0);
    if (!new_k) {
        throw std::runtime_error("grow_layer_chunk: clSVMAlloc failed");
    }
    __fp16 * new_v = new_k + chunk_elements;

    // The kq_mask excludes unwritten positions from attention, but the chunked
    // flash-attn kernel still reads them before masking — recycled SVM pages
    // could hold NaN bit patterns that would poison the softmax, so zero the
    // chunk (same guarantee the constructor-time buffer clear gives chunk 0).
    memset(new_k, 0, chunk_bytes * 2);

    ggml_tensor * k_chunk = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, chunk_elements);
    ggml_tensor * v_chunk = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, chunk_elements);

    ggml_format_name(k_chunk, "cache_k_c%d_l%d", idx, il);
    ggml_format_name(v_chunk, "cache_v_c%d_l%d", idx, il);

    // Bare tensors: chunk data lives outside the ggml buffer machinery (as it
    // does after any swapin reallocation) and is reached through the pointer
    // tables below, never as a direct ggml op source.
    k_chunk->data = new_k;
    v_chunk->data = new_v;

    layers_k_chunks[il].c[idx] = k_chunk;
    layers_v_chunks[il].c[idx] = v_chunk;

    // Publish to the GPU-visible tables (fine-grain SVM: host stores are
    // immediately visible to kernels launched afterwards).
    g_svm_chunk_ptrs[2 * il + 0]->c[idx] = new_k;
    g_svm_chunk_ptrs[2 * il + 1]->c[idx] = new_v;
#endif // MZCACHE_USE_OPENCL
}

void register_kv_chunk_offsets(
    uint32_t       n_layer,
    uint32_t       kv_size,
    const std::vector<ggml_tensor_ptrs> & layers_k_chunks,
    const std::vector<ggml_tensor_ptrs> & layers_v_chunks,
    std::vector<ggml_backend_buffer_ptr> & bufs)
{
    (void) kv_size;

    if (bufs.size() > 1) {
        MZ_LOG_ERROR("register_kv_chunk_offsets: expected 1 buffer, found %zu", bufs.size());
    }

    ggml_backend_buffer_t buf = bufs.front().get();

    // With lazy chunk creation each layer contributes exactly two child
    // buffers to the cache's multi-buffer (the K and V base tensors), so the
    // base tensor of layer il lives at child index il*2 (K) / il*2+1 (V).
    // The call order below also fixes the g_svm_chunk_ptrs layout: slot 2*il
    // is layer il's K table, slot 2*il+1 its V table.
    for (uint32_t il = 0; il < n_layer; ++il) {
        ggml_backend_buffer_set_chunk_offset(buf, layers_k_chunks[il].c[0], il * 2);
        ggml_backend_buffer_set_chunk_offset(buf, layers_v_chunks[il].c[0], il * 2 + 1);
    }
}

void state_write_k_chunked(
    llama_io_write_i & io,
    uint32_t il,
    uint64_t k_size_row,
    const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges)
{
    chunk_write(io, 2 * il, k_size_row, cell_ranges);
}

void state_write_v_chunked(
    llama_io_write_i & io,
    uint32_t il,
    uint64_t v_size_row,
    const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges)
{
    chunk_write(io, 2 * il + 1, v_size_row, cell_ranges);
}

void state_read_k_chunked(
    llama_io_read_i & io,
    uint32_t il,
    uint64_t k_size_row,
    uint32_t cell_count,
    uint32_t head,
    const std::vector<ggml_tensor_ptrs> & layers_k_chunks)
{
    (void) layers_k_chunks;
    chunk_read(io, 2 * il, k_size_row, cell_count, head);
}

void state_read_v_chunked(
    llama_io_read_i & io,
    uint32_t il,
    uint64_t v_size_row,
    uint32_t cell_count,
    uint32_t head,
    const std::vector<ggml_tensor_ptrs> & layers_v_chunks)
{
    (void) layers_v_chunks;
    chunk_read(io, 2 * il + 1, v_size_row, cell_count, head);
}

} // namespace mzcache
