llama_add_compile_flags()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache-unified.cpp
            llama-kv-cache-unified-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampling.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            )

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()

if (MZCACHE_SVM_KV_CHUNK)
    # mzcache: consumers (mz_* examples, the Android JNI wrapper) reach llama
    # internal headers (src/) and mzcache headers through the llama target;
    # llama-model.cpp itself needs ggml-backend-impl.h from ggml/src
    target_include_directories(llama PUBLIC . ../mzcache/include ../ggml/src)
    if (ANDROID)
        # mzcache_types.h routes MZ_LOG_INFO/ERROR to __android_log_print on
        # Android (commit 2d8c8b9a); that symbol lives in liblog. llama-model.cpp
        # (the mzcache metadata loader) and the mz_* examples emit MZ_LOG, so link
        # liblog PUBLIC to propagate onto every executable that links llama.
        target_link_libraries(llama PUBLIC log)
    endif()
endif()
