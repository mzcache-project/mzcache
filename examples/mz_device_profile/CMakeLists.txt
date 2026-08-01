set(TARGET mz_device_profile)
add_executable(${TARGET} mz_device_profile.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama mzcache ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../mzcache/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../mzcache/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
target_compile_options(${TARGET} PRIVATE -march=armv8.2-a+fp16)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
