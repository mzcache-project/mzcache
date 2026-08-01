# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# examples

if (EMSCRIPTEN)
else()
    add_subdirectory(batched)
    add_subdirectory(embedding)
    add_subdirectory(eval-callback)

    add_subdirectory(gguf-hash)
    add_subdirectory(gguf)
    add_subdirectory(gritlm)
    add_subdirectory(lookahead)
    add_subdirectory(lookup)
    add_subdirectory(parallel)
    add_subdirectory(passkey)
    add_subdirectory(retrieval)
    add_subdirectory(save-load-state)
    add_subdirectory(simple)
    add_subdirectory(simple-chat)
    add_subdirectory(speculative)
    add_subdirectory(speculative-simple)
    add_subdirectory(gen-docs)
    add_subdirectory(training)

    # Deprecated examples
    add_subdirectory(weight_load_test) # mzcache


    add_subdirectory(mz_comp_decomp_test) # mzcache
    add_subdirectory(mz_kv_profile) # mzcache
    add_subdirectory(mz_kv_store_load_test) # mzcache
    add_subdirectory(mz_load_state_and_inference) # mzcache
    # add_subdirectory(mz_profile) # mzcache
    add_subdirectory(mz_save_layers_to_files) # mzcache
    # add_subdirectory(mz_weight_test) # mzcache
    add_subdirectory(mz_baseline_swap)
    add_subdirectory(exp1_partial_offload)
    if (NOT GGML_BACKEND_DL)
        add_subdirectory(convert-llama2c-to-ggml)
        # these examples use the backends directly and cannot be built with dynamic loading
        if (GGML_SYCL)
            add_subdirectory(sycl)
        endif()
    endif()
endif()
