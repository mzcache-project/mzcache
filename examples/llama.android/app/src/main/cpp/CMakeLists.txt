cmake_minimum_required(VERSION 3.18.1)

project(psiwatcher)

add_library(psiwatcher SHARED
    psi_watcher.cpp
)

find_library(log-lib log)

target_link_libraries(psiwatcher
    ${log-lib}
)
