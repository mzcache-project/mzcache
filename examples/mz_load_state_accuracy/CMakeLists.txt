set(TARGET mz_load_state_accuracy)
add_executable(${TARGET} mz_load_state_accuracy.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
# Reaches into internal llama headers (llama-model.h) + vendored single-header JSON.
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor/nlohmann)
target_compile_features(${TARGET} PRIVATE cxx_std_17)