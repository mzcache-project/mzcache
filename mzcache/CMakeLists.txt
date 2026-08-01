# Choices: FLEXGEN, CACHEGEN, NONE
set(MZCACHE_COMPRESSION "FLEXGEN_8BIT" CACHE STRING "Compression backend (FLEXGEN|FLEXGEN_8BIT|CACHEGEN|NONE)")
set_property(CACHE MZCACHE_COMPRESSION PROPERTY STRINGS FLEXGEN FLEXGEN_8BIT CACHEGEN NONE)

# Validate the value (define the list first)
set(_mzcache_comp_list FLEXGEN FLEXGEN_8BIT CACHEGEN NONE)
list(FIND _mzcache_comp_list "${MZCACHE_COMPRESSION}" _mzcache_comp_idx)
if(_mzcache_comp_idx EQUAL -1)
  message(FATAL_ERROR
    "Invalid MZCACHE_COMPRESSION='${MZCACHE_COMPRESSION}'. Choose one of: ${_mzcache_comp_list}")
endif()

# option(MZCACHE "Build with mzcache library" ON)
option(MZCACHE_USE_OPENCL "Build OpenCL backend" ON)
add_subdirectory(src)

# Per-backend macro: MZCACHE_COMPRESS_FLEXGEN or _CACHEGEN or _NONE
string(TOUPPER "${MZCACHE_COMPRESSION}" MZCACHE_COMP_UPPER)
target_compile_definitions(mzcache PRIVATE MZCACHE_COMPRESS_${MZCACHE_COMP_UPPER})



if(MZCACHE_SVM_KV_CHUNK)
    target_compile_definitions(mzcache PRIVATE MZCACHE_SVM_KV_CHUNK)
endif()

if(MZCACHE_USE_OPENCL)
    target_compile_definitions(mzcache PRIVATE MZCACHE_USE_OPENCL)
endif()

# Expose the headers to the entire llama project
target_include_directories(mzcache PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels           # mzcache/kernels
)
