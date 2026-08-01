#include "mzcache_globals.h"

// Weight context globals
ggml_context *              mz_weight_ctx  = nullptr;
ggml_backend_buffer_type_t  mz_weight_buft = nullptr;
ggml_backend_buffer_t       mz_weight_buf  = nullptr;

// Per-layer weight metadata
int                    layer_0_tensor_idx        = -1;
int                    num_tensors_per_layer      = -1;
int                    num_bytes_per_layer        = -1;
int                    padded_num_bytes_per_layer = -1;
std::vector<size_t>    layer_weight_sizes;
struct ggml_tensor *   layer_first_tensor[MAX_NUM_LAYERS] = {nullptr};
