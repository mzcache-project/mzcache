# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("llama-android")

#include(FetchContent)
#FetchContent_Declare(
#        llama
#        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
#        GIT_TAG        master
#)

# Also provides "common"
#FetchContent_MakeAvailable(llama)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#

#load local llama.cpp
add_subdirectory(../../../../../../ build-llama)

# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.
add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        llama-android.cpp)

# --------------- OpenCL (libOpenCL.so pulled from the device) ---------------
# The GGML OpenCL backend needs the vendor OpenCL library. Place the device's
# libOpenCL.so under ../jniLibs/<ABI>/ and the CL headers under include/CL.
# (gradle also pre-seeds find_package(OpenCL) via -DOpenCL_LIBRARY/-DOpenCL_INCLUDE_DIR)
# The cpu-swap baseline build (-PmzCpuSwap → GGML_OPENCL=OFF) skips all of it.
if(GGML_OPENCL)
    include_directories(${CMAKE_SOURCE_DIR}/include/CL)
    set(EXT_LIBS_DIR ${CMAKE_SOURCE_DIR}/../jniLibs)
    add_library(OpenCL SHARED IMPORTED)
    set_target_properties(OpenCL PROPERTIES
            IMPORTED_LOCATION ${EXT_LIBS_DIR}/${ANDROID_ABI}/libOpenCL.so)
endif()
# -----------------------------------------------------------------------------

# The root CMakeLists' add_compile_definitions() does not reach this sibling
# directory, and the mzcache members of llama's private headers (e.g.
# llama_kv_cache_unified::layers_k_chunks) are #ifdef-guarded, so the JNI
# must be compiled with the same definition as the llama/mzcache targets.
if(MZCACHE_SVM_KV_CHUNK)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE MZCACHE_SVM_KV_CHUNK)
endif()

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
# mzcache/OpenCL exist only in the respective build flavors.
set(MZ_EXTRA_LIBS "")
if(MZCACHE_SVM_KV_CHUNK)
    list(APPEND MZ_EXTRA_LIBS mzcache)
endif()
if(GGML_OPENCL)
    list(APPEND MZ_EXTRA_LIBS OpenCL)
endif()
target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        llama
        common
        ${MZ_EXTRA_LIBS}
        android
        log)
