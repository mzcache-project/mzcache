set(TARGET mz_gen_state)
add_executable(${TARGET} mz_gen_state.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
# mz_gen_state reaches into internal llama headers (llama-model.h) for the
# arch/type-based state file naming, so expose src/ like mz_accuracy does.
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
