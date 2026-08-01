#include "mzcache_weight_ops.h"
#include "mzcache_globals.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstring>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace mzcache {

bool alloc_weight_layer(int layer_idx) {
    int n_allocated = 0;
    struct ggml_tensor * first_tensor = layer_first_tensor[layer_idx];
    struct ggml_tensor * last_tensor  = layer_first_tensor[layer_idx + 1]; // nullptr if last layer

    return ggml_backend_alloc_ctx_given_tensors_from_buft(
        mz_weight_ctx, mz_weight_buf, mz_weight_buft,
        first_tensor, last_tensor, &n_allocated);
}

bool read_weight_layer(int & fd, int layer_idx) {
#ifdef MZCACHE_SVM_KV_CHUNK
    size_t bytes = padded_num_bytes_per_layer;
    if (bytes % 4096 != 0) {
        MZ_LOG_ERROR("read_weight_layer: bytes (%zu) is not multiple of 4096", bytes);
        close(fd);
        fd = -1;
        return false;
    }

    ssize_t read_bytes = read(fd, layer_first_tensor[layer_idx]->data, padded_num_bytes_per_layer);
    if (read_bytes < 0 || static_cast<size_t>(read_bytes) != (size_t)padded_num_bytes_per_layer) {
        close(fd);
        fd = -1;
        throw std::runtime_error(
            "Failed to read layer " + std::to_string(layer_idx) +
            ", read_bytes=" + std::to_string(read_bytes) +
            ", expected=" + std::to_string(padded_num_bytes_per_layer) +
            ", errno: " + std::strerror(errno));
    }

    close(fd);
    fd = -1;
    return true;
#else
    MZ_LOG_ERROR("read_weight_layer: not implemented for non-SVM build");
    close(fd);
    fd = -1;
    return false;
#endif
}

bool unload_buffers(int layer_idx) {
    int first_tensor_idx = layer_0_tensor_idx + num_tensors_per_layer * layer_idx;

    std::vector<ggml_tensor *> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(mz_weight_ctx);
         t != nullptr;
         t = ggml_get_next_tensor(mz_weight_ctx, t)) {
        tensors.push_back(t);
    }

    std::vector<ggml_backend_buffer_t> unique_buffers;
    for (int i = 0; i < num_tensors_per_layer; ++i) {
        int tensor_idx = first_tensor_idx + i;

        if (tensor_idx < 0 || tensor_idx >= (int)tensors.size()) {
            MZ_LOG_ERROR("unload_buffers: invalid tensor index %d", tensor_idx);
            continue;
        }

        ggml_tensor * t = tensors[tensor_idx];
        if (t != nullptr && t->buffer != nullptr &&
            std::find(unique_buffers.begin(), unique_buffers.end(), t->buffer) == unique_buffers.end()) {
            unique_buffers.push_back(t->buffer);
        }
    }

    for (ggml_backend_buffer_t buffer : unique_buffers) {
        if (!ggml_backend_buffer_unload_child_buffer(mz_weight_buf, buffer)) {
            MZ_LOG_ERROR("unload_buffers: failed to unload child buffer for layer %d", layer_idx);
            return false;
        }
    }

    for (int i = 0; i < num_tensors_per_layer; ++i) {
        int tensor_idx = first_tensor_idx + i;

        if (tensor_idx < 0 || tensor_idx >= (int)tensors.size()) {
            continue;
        }

        ggml_tensor * t = tensors[tensor_idx];
        if (t != nullptr) {
            t->buffer = nullptr;
            t->data   = nullptr;
#ifdef MZCACHE_SVM_KV_CHUNK
            t->is_weight = 1;
#endif
        }
    }

    return true;
}

bool save_weights_in_layer_to_file(const std::string & file_path, int layer_idx) {
    std::ofstream out_file(file_path, std::ios::binary);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + file_path);
    }

    constexpr size_t alignment = 4096;
    size_t total_size = 0;

    struct ggml_tensor * cur = layer_first_tensor[layer_idx];
    if (!cur) {
        out_file.close();
        throw std::runtime_error("No tensors found for layer " + std::to_string(layer_idx));
    }

    for (int t = 0; t < num_tensors_per_layer; ++t) {
        size_t n_size = ggml_nbytes(cur);
        std::vector<uint8_t> tmp_buf(n_size);
        ggml_backend_tensor_get(cur, tmp_buf.data(), 0, n_size);

        out_file.write(reinterpret_cast<char *>(tmp_buf.data()), n_size);
        total_size += n_size;

        cur = ggml_get_next_tensor(mz_weight_ctx, cur);
        if (!cur && t + 1 < num_tensors_per_layer) {
            out_file.close();
            throw std::runtime_error(
                "Insufficient tensors when saving layer " + std::to_string(layer_idx));
        }
    }

    size_t padding = (alignment - (total_size % alignment)) % alignment;
    if (padding > 0) {
        std::vector<char> pad_buf(padding, 0);
        out_file.write(pad_buf.data(), padding);
    }

    out_file.close();
    return true;
}

bool unload_layers_to_file(const std::string & file_path, int layer_idx) {
    if (layer_idx < 0) {
        throw std::runtime_error("unload_layers_to_file: layer_idx is invalid");
    }

    std::string layer_file_path = file_path + "_layer_" + std::to_string(layer_idx) + ".bin";
    return save_weights_in_layer_to_file(layer_file_path, layer_idx);
}

} // namespace mzcache
