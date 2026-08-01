#include <CL/cl.h>
#ifdef MZCACHE_SVM_KV_CHUNK
#include "mzcache_types.h"
#endif

//------------------------------------------------------------------------------
// Tensor extra management
//------------------------------------------------------------------------------
struct ggml_tensor_extra_cl {
    // The buffer object that holds the data.
    cl_mem data_device;
    // The offset into the buffer object. This is primarily for scratch buffer
    // and view operation.
    // NB: this offset no longer includes view offset (view_offs). Whenever this
    // offset is used, view_offs should be considered.
    cl_ulong offset;
    // The actual size of the cl_mem object. This is needed when returning the
    // block to the pool.
    size_t actual_size;

    void reset() {
        data_device = nullptr;
        offset = 0;
        actual_size = 0;
    }
#ifdef MZCACHE_SVM_KV_CHUNK
    cl_context clctx = nullptr;
    chunk_ptrs * svm_chunk_ptrs = nullptr;
#endif
};

#ifdef MZCACHE_SVM_KV_CHUNK
// mzcache: OpenCL context/queue shared with mzcache/src/mzcache_kv_state.cpp
// (SVM allocation happens outside the backend); defined in ggml-opencl.cpp
extern cl_context g_opencl_context;
extern cl_command_queue g_opencl_queue;
#endif
