# mzCache ↔ llama.cpp integration

mzCache lives in `mzcache/{include,src,kernels}` as a static library, but a system that
swaps tensors out from under a running inference engine cannot be a pure add-on: it needs
hooks inside the allocator, the KV cache, the model loader, and the GPU backend. This
document lists every change mzCache makes to the upstream tree (base:
[`fa4a9f2a`](https://github.com/ggml-org/llama.cpp/commit/fa4a9f2a1ccda2573189a9d4995bdf0bceb41156))
and why it exists.

**Everything below is compiled only when `MZCACHE_SVM_KV_CHUNK` is ON.** With the option
off, the tree builds as vanilla llama.cpp — the `mzcache/` directory is not even required
to configure.

## ggml

| File | Hook | Why |
|---|---|---|
| `ggml/include/ggml.h` | `struct ggml_tensor` gains `void * ggml_tensor_ptrs` and `int64_t is_weight` (replacing the padding bytes) | every KV tensor must reach its per-layer 256-token chunk table from inside backend kernels; weight tensors must be recognizable so the allocator can skip them |
| `ggml/include/ggml.h`, `ggml/src/ggml.c` | `ggml_flash_attn_ext(..., int32_t ne11)` variant | chunked K/V tensors no longer expose the true KV length through their shape, so it rides along in `op_params` |
| `ggml/src/ggml-alloc.c` | weight tensors return early from `ggml_gallocr_init_tensor`; context buffers are split one-buffer-per-tensor (`max_size = 0`); two new allocators (`ggml_backend_alloc_ctx_unloaded_tensors_from_buft`, `..._given_tensors_from_buft`) | unloading a weight layer frees its child buffers individually; reloading allocates one 4 KiB-padded SVM buffer per layer for O_DIRECT reads |
| `ggml/src/ggml-alloc.c` | graph realloc check compiled out | graph buffers are reserved once and reused across swapin cycles; a mid-decode realloc would race the overlapped restore |
| `ggml/src/ggml-backend.cpp` | multi-buffer context gains `cur_n_buffers` + a mutex and nullable child slots; `unload_buffers` / `unload_child_buffer` / `concat_buffer` APIs and vtable slots | weight layers are freed and re-inserted into the model's multi-buffer while graph compute runs on other layers |
| `ggml/src/ggml-backend.cpp` | `ggml_backend_buft_alloc_svm_buffer` dispatch | KV chunks and weight layers live in SVM so the CPU-side compressor/loader and the GPU see the same memory |
| `ggml/src/ggml-opencl/ggml-opencl.cpp` | defines `g_opencl_context/queue`, `g_svm_chunk_ptrs`, `last_chunk_idx`; calls `g_mzcache_core->wait_for_layer_sync()` before per-layer ops; the chunked `mzcache_flash_attn_*` kernels; `FLASH_ATTN_EXT` support | the backend must block a layer's compute until that layer's chunks are restored (layer sync), and the chunked flash-attention kernels gather K/V through the SVM chunk tables |
| `ggml/src/ggml-opencl/ggml-opencl-extra.h` | tensor extras carry `svm_chunk_ptrs`; shared CL context/queue externs | mzcache_kv_state allocates SVM outside the backend and needs the backend's `cl_context` |

## llama

| File | Hook | Why |
|---|---|---|
| `src/llama-model.{h,cpp}` | `load_tensors()` has an mzcache variant for the supported architectures that records per-tensor layout (`tensor_name_and_idx`, `layer_first_tensor`, `num_bytes_per_layer`, ...) while creating tensors; `alloc/read_weight_layer`, `unload_buffers`, `unload_layers_to_file` delegate to `mzcache_weight_ops`; the single weight ctx/buft/buf is captured | weight-layer swapout needs to know exactly which tensors form a layer, their sizes, and the buffer they live in |
| `src/llama-kv-cache-unified.{h,cpp}` | K/V tensors are created as per-layer chunk tables (`layers_k_chunks`/`layers_v_chunks`) via `mzcache::create_chunked_kv_tensors`; `state_write/read` walk chunks; `get_k/get_v` return chunk-local views; cache writes compute chunk-local offsets | the 256-token chunk granularity is established at cache creation and threaded through every KV access |
| `src/llama-context.cpp` | after `ggml_backend_sched_alloc_graph` succeeds, signals `g_graph_alloc_promise` | `swapin_generate` must not enqueue decompress/load jobs that touch graph memory until the decode's graph allocation is done — this promise is the prerequisite for overlapping restore with prefill |
| `src/llama-graph.{h,cpp}` | `build_attn_mha(..., n_kv)` | passes the true KV length down to the flash-attention op (see `ne11` above) |
| `include/llama.h` | `llama_model_get_n_tensors()` | used by the examples to size bookkeeping |

## Build wiring

- Root `CMakeLists.txt`: `option(MZCACHE_SVM_KV_CHUNK ... OFF)`; when ON, adds the global
  compile definition and the `mzcache/` subdirectory
- `src/CMakeLists.txt`: when ON, the `llama` target publicly exposes `src/`,
  `mzcache/include` and `ggml/src` (mz examples and the Android JNI wrapper include
  internal headers)
- `ggml/src/ggml-opencl/CMakeLists.txt`: when ON, links `mzcache` and registers the
  `mzcache_flash_attn_*` kernels
- `examples/CMakeLists.txt`: when ON, adds `mz_device_profile` (the mandatory
  profiling gate), `mz_dump_layers`, `mz_prefill`, `mz_prefill_power`,
  `mz_prefill_repeated`, `mz_save_state`, `mz_save_state_v2`, `mz_generate` and
  `mz_chat`, plus `exp1_android_swap`. The CUDA-side accuracy tools
  (`mz_accuracy`, `mz_load_state_accuracy`, `mz_gen_state`) hang off the separate
  `MZCACHE_BUILD_SERVER` option

Note: `ggml-opencl` and `mzcache` are mutually dependent (layer-sync calls one way, SVM
globals the other), which is one reason the Android app must build with
`BUILD_SHARED_LIBS=OFF`.

## Cross-library globals

Shared state that crosses the library boundary is declared centrally in
`mzcache/include/mzcache_globals.h`:

- weight bookkeeping (`mz_weight_ctx/buft/buf`, `num_bytes_per_layer`,
  `layer_first_tensor[]`, ...) — defined in `mzcache/src/mzcache_globals.cpp`, consumed by
  `ggml-alloc.c` and `llama-model.cpp`
- SVM chunk tables (`g_svm_chunk_ptrs`, `last_chunk_idx`) and the CL context/queue —
  defined in `ggml-opencl.cpp`, consumed by `mzcache_kv_state.cpp`
- the graph-alloc promise (`g_graph_alloc_promise`, `g_graph_alloc_signaled`) and
  `g_mzcache_core` — defined in `mzcache/src/mzcache_core.cpp`, consumed by
  `llama-context.cpp` and `ggml-opencl.cpp`

## Invariants worth knowing

1. mzCache file I/O (KV store files, arena swap files, weight layer dumps) is relative to
   the process CWD and uses O_DIRECT — run from a writable, O_DIRECT-capable filesystem
   (not FUSE-backed external storage on Android).
2. Weight layer dumps are generated on the first run while all layers are resident;
   deleting them breaks later weight reloads.
3. Chunks are materialized lazily: only chunk 0 exists at cache creation, and
   `llama_kv_cache_unified::mz_grow_chunks` (hooked into `apply_ubatch` and state
   restore) allocates the rest as the cache fills, so generation past the restored
   state grows the chunk set at runtime (`mzcache_core::on_kv_chunks_grown` keeps
   the swapout bookkeeping in step). A single KV-cache *write* still must not span
   two chunks — `mzcache::decode_chunk_aligned` (used by `swapin_generate` and the
   examples) transparently splits batches at 256-token boundaries.
4. All layers must live on a single device (the chunk offset registration and the
   one-multi-buffer weight capture assume this).
5. The chunked flash-attention kernel tiles queries over its launch grid
   (8 per work-group), so a single invocation handles any query count up to the
   ubatch ceiling. That ceiling is 256 — not a kernel limit but the KV-cache
   *write* constraint from invariant #3 — so keep `n_ubatch <= 256` in mzcache
   flows (see `examples/mz_generate`).
6. The chunked FA kernel is ONE source (`mzcache_flash_attn_f16_kvchunk.cl`)
   whose tuning constants are injected as build options: the loader compiles it
   twice per device (`-DD=64` for EXAONE4, `-DD=128` for Qwen3) with
   `-DNWARPS`/`-DKQ_TILE` taken from `get_kernel_config()` (Adreno 8xx = {128, 4},
   7xx = {64, 2}), so the kernel constants can never diverge from the host
   launch config — a mismatch (e.g. a 128/4 kernel launched with a {64, 2}
   work-group) leaves half the KQ tile's local memory unwritten and poisons the
   softmax with NaNs.
