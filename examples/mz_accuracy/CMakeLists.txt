set(TARGET mz_accuracy)
add_executable(${TARGET} mz_accuracy.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
# mz_accuracy reaches into internal llama headers (llama-model.h) and the vendored
# single-header JSON, so expose src/ and vendor/nlohmann on top of what `common`
# provides transitively.
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor/nlohmann)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
