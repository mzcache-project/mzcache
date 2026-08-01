llama_add_compile_flags()

# Add all cpp files
file(GLOB_RECURSE MZCACHE_SOURCES *.cpp)
list(APPEND MZCACHE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/compress.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/decompress.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/cachegen_utils.cpp
)

# Build the static library
add_library(mzcache STATIC ${MZCACHE_SOURCES})

target_compile_features(mzcache PUBLIC cxx_std_17)

# mzcache calls the OpenCL runtime directly (clSVMAlloc/clSVMFree/kernels) and
# its public headers include <CL/cl.h>, so declare the dependency here instead
# of relying on CL headers injected into the NDK sysroot. PUBLIC propagates the
# include dirs to every consumer (the mz_* examples, the Android JNI wrapper).
# The build scripts point find_package(OpenCL) at the local opencl_sdk/ prefix
# (vendored Khronos headers + the device's own libOpenCL.so).
find_package(OpenCL REQUIRED)
target_link_libraries(mzcache PUBLIC OpenCL::OpenCL)

# Header search paths for mzcache
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
