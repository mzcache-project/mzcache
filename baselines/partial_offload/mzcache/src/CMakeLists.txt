llama_add_compile_flags()

# add all cpp files
file(GLOB_RECURSE MZCACHE_SOURCES *.cpp)
list(APPEND MZCACHE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/compress.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/decompress.cpp
)

# create static library
add_library(mzcache STATIC ${MZCACHE_SOURCES})

target_compile_features(mzcache PUBLIC cxx_std_17)

# Same scheme as the main tree: mzcache's headers include <CL/cl.h>, so declare
# the OpenCL dependency (the build script points find_package at opencl_sdk/)
# instead of relying on CL headers injected into the NDK sysroot.
find_package(OpenCL REQUIRED)
target_link_libraries(mzcache PUBLIC OpenCL::OpenCL)

# add header paths visible to mzcache
target_include_directories(mzcache PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include             # mzcache/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common          # llama.cpp/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include    # llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include         # llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src             # llama.cpp/src
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernels            # mzcache/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-opencl # ggml-opencl (to include ggml_opencl_extra.h)
)


target_include_directories(mzcache PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/
)

target_compile_options(mzcache PRIVATE -march=armv8.2-a+fp16)
