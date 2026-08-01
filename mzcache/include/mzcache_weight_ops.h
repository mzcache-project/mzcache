#pragma once

// [mzcache] Weight layer management operations.
// These functions implement per-layer weight allocation, reading, and unloading,
// extracted from llama_model member functions for code separation.

#include "mzcache_globals.h"

#include <string>

namespace mzcache {

// Allocate buffer for a single layer's weight tensors.
bool alloc_weight_layer(int layer_idx);

// Read a layer's weight data from an open file descriptor (O_DIRECT).
// Closes fd on completion or error. Only available when MZCACHE_SVM_KV_CHUNK is defined.
bool read_weight_layer(int & fd, int layer_idx);

// Release a layer's weight tensors from GPU/device memory.
bool unload_buffers(int layer_idx);

// Save a layer's weight data to a binary file (page-aligned).
bool save_weights_in_layer_to_file(const std::string & file_path, int layer_idx);

// Save a single layer's weights to "<file_path>_layer_<idx>.bin".
bool unload_layers_to_file(const std::string & file_path, int layer_idx);

} // namespace mzcache
