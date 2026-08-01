# dependencies

find_package(Threads REQUIRED)

# flags

llama_add_compile_flags()

# examples

if (NOT EMSCRIPTEN)
    # mzcache examples: on-device prefill-state creation and swapout/swapin cycles
    if (MZCACHE_SVM_KV_CHUNK)
        add_subdirectory(mz_device_profile)
        add_subdirectory(mz_dump_layers)
        add_subdirectory(mz_prefill)
        add_subdirectory(mz_prefill_power)
        add_subdirectory(mz_prefill_repeated)
        add_subdirectory(mz_save_state)
        add_subdirectory(mz_save_state_v2)
        add_subdirectory(mz_generate)
        add_subdirectory(mz_chat)
    endif()

    # GPU/server-side tools (CUDA build, MZCACHE_SVM_KV_CHUNK off):
    #  - mz_accuracy / mz_load_state_accuracy: F1-accuracy pipeline
    #    (see docs/reproduce_f1_accuracy.md)
    #  - mz_gen_state: prefill-state (.kv) generator for the on-device
    #    experiments (see EVALUATION.md Section 1.2, scripts/setup/gen_prefill_states.sh)
    if (MZCACHE_BUILD_SERVER)
        add_subdirectory(mz_accuracy)
        add_subdirectory(mz_load_state_accuracy)
        add_subdirectory(mz_gen_state)
        # Stock llama.cpp quantizer, used host-side to turn the downloaded BF16
        # GGUFs into the FP16 GGUFs the experiments use. CPU-only, so it builds
        # on both the GPU server and the phone (adb) host — see EVALUATION.md Section 1.2.
        add_subdirectory(quantize)
    endif()

    # Figure 9 reproduction baseline (stock CPU llama.cpp Android swap).
    add_subdirectory(exp1_android_swap)
endif()
