#!/bin/bash

# Usage: ./build_and_push.sh /absolute/path/to/install_dir

set -e  # Exit on any error

cmake -DLLAMA_CURL=OFF -DGGML_OPENMP=ON -B build 

cmake --build build --config Release -j$(nproc)
