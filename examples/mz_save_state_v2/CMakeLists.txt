set(TARGET mz_save_state_v2)
add_executable(${TARGET} mz_save_state_v2.cpp)
install(TARGETS ${TARGET} RUNTIME)

string(TOUPPER "${MZCACHE_COMPRESSION}" MZCACHE_COMP_UPPER)
target_compile_definitions(${TARGET} PRIVATE MZCACHE_COMPRESS_${MZCACHE_COMP_UPPER})

target_link_libraries(${TARGET} PRIVATE common llama mzcache ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)