#!/bin/bash
# get_models.sh — obtain the FP16 GGUF models used by the experiments.
#
# Downloads the upstream BF16 GGUFs and quantizes them to FP16 with
# llama-quantize (CPU-only — no GPU backend needed, so this runs on either the
# GPU server or the phone/adb host). The FP16 output is bit-identical to the
# GGUF used in the paper.
#
# Usage (from the repository root):
#   ./scripts/setup/get_models.sh [out_dir]      # default out_dir: current dir
#
# Requires llama-quantize. Build it first with either:
#   ./scripts/compression/accuracy/build_server_cuda.sh      # GPU server (CUDA build)
#   ./scripts/setup/build_quantize.sh            # phone host (CPU-only build)
#
# Produces in <out_dir>:
#   Qwen3-0.6B-FP16.gguf
#   EXAONE-4.0-1.2B-FP16.gguf

set -eu

cd "$(dirname "$0")/../.."
ROOT=$(pwd)
OUT=${1:-.}
mkdir -p "$OUT"

# Locate llama-quantize (CUDA build dir or the CPU-only build dir).
QUANT=""
for c in cuda_build/bin/llama-quantize build_quantize/bin/llama-quantize; do
    if [ -x "$ROOT/$c" ]; then QUANT="$ROOT/$c"; break; fi
done
[ -n "$QUANT" ] || { echo "ERROR: llama-quantize not found — build it first (see header)" >&2; exit 1; }

fetch() {  # fetch <url> <dst>
    [ -f "$2" ] && { echo "[skip] $2 already present"; return; }
    echo "[get ] $1"
    if command -v curl >/dev/null 2>&1; then
        curl -fL "$1" -o "$2"
    else
        wget -O "$2" "$1"
    fi
}

# model_tag | BF16 download URL | FP16 output name
download_and_quantize() {
    local url=$1 bf16=$2 fp16=$3
    fetch "$url" "$OUT/$bf16"
    echo "[quant] $bf16 -> $fp16 (f16)"
    "$QUANT" "$OUT/$bf16" "$OUT/$fp16" f16
}

download_and_quantize \
    "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-BF16.gguf" \
    "Qwen3-0.6B-BF16.gguf" "Qwen3-0.6B-FP16.gguf"

download_and_quantize \
    "https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF/resolve/main/EXAONE-4.0-1.2B-BF16.gguf" \
    "EXAONE-4.0-1.2B-BF16.gguf" "EXAONE-4.0-1.2B-FP16.gguf"

echo ""
echo "--- FP16 models ready ---"
ls -lh "$OUT/Qwen3-0.6B-FP16.gguf" "$OUT/EXAONE-4.0-1.2B-FP16.gguf"
