# option(MZCACHE "Build with mzcache library" ON)
option(MZCACHE_USE_OPENCL "Build OpenCL backend" ON)

add_subdirectory(src)

if(MZCACHE_SVM_KV_CHUNK)
    target_compile_definitions(mzcache PRIVATE MZCACHE_SVM_KV_CHUNK)
endif()

if(MZCACHE_USE_OPENCL)
    target_compile_definitions(mzcache PRIVATE MZCACHE_USE_OPENCL)
endif()

# expose headers to the whole llama project
target_include_directories(mzcache PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
