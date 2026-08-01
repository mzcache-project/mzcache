set(TARGET mz_profile_test)
add_executable(${TARGET} mz_profile_test.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ../../vendor/nlohmann)
# target_link_libraries(${TARGET} PRIVATE mzcache common llama ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${TARGET} PRIVATE common llama mzcache ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
