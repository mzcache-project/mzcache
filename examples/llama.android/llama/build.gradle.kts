plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "android.llama.cpp"
    compileSdk = 34

    packaging {
        jniLibs {
            // The device's own libOpenCL.so is dlopen'ed at runtime
            // (uses-native-library); do not package the local copy.
            excludes += "/lib/**/libOpenCL.so"
        }
    }

    defaultConfig {
        minSdk = 33

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
        ndk {
            // Add NDK properties if wanted, e.g.
            abiFilters += listOf("arm64-v8a")
        }
        externalNativeBuild {
            cmake {
                arguments += "-DLLAMA_CURL=OFF"
                arguments += "-DLLAMA_BUILD_COMMON=ON"
                arguments += "-DGGML_LLAMAFILE=OFF"
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                // Match mzcache_build.sh: static ggml/llama/mzcache so that the
                // mzcache globals referenced from ggml-alloc.c (num_bytes_per_layer,
                // layer_first_tensor, ...) resolve at the final libllama-android.so
                // link instead of failing libggml-base.so's --no-undefined check.
                arguments += "-DBUILD_SHARED_LIBS=OFF"
                if ((project.findProperty("mzCpuSwap")?.toString() ?: "false") == "true") {
                    // cpu-swap baseline (./gradlew assembleDebug -PmzCpuSwap=true):
                    // vanilla CPU llama.cpp — weights mmapped (kernel-reclaimable),
                    // KV cache anonymous memory (swaps out to zram). No OpenCL, no
                    // mzcache. NOTE: switching flavors changes these cmake args —
                    // delete llama/.cxx first.
                    arguments += "-DGGML_OPENCL=OFF"
                    arguments += "-DMZCACHE_SVM_KV_CHUNK=OFF"
                } else {
                    arguments += "-DGGML_OPENCL=ON"
                    arguments += "-DMZCACHE_SVM_KV_CHUNK=ON"
                    // Match mzcache_build.sh (desktop/NDK build) - default is FLEXGEN
                    arguments += "-DMZCACHE_COMPRESSION=FLEXGEN_8BIT"
                    // Pre-seed find_package(OpenCL REQUIRED) in ggml-opencl with the
                    // device libOpenCL.so / vendored CL headers (see cpp/CMakeLists.txt).
                    arguments += "-DOpenCL_LIBRARY=${projectDir}/src/main/jniLibs/arm64-v8a/libOpenCL.so"
                    arguments += "-DOpenCL_INCLUDE_DIR=${projectDir}/src/main/cpp/include"
                }
                cppFlags += listOf()
                arguments += listOf()

                cppFlags("")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {

    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}
