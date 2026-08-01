#!/bin/bash
# gen_prefill_states.sh — generate the prefill-state (.kv) files on the GPU server.
#
# Runs cuda_build/bin/mz_gen_state over the full experiment grid:
#   model x context {8193, 16385, 32700} x {flash-attn ON, OFF}
# and writes the 12 files the on-device runners expect into ./states/:
#   <model_tag>_fa_<ctx>.kv   flash-attn ON  -> mzCache binaries
#   <model_tag>_<ctx>.kv      flash-attn OFF -> partial_offload, OS-paging baseline
# with <model_tag> derived from the model (qwen3_0.6B, exaone4_1.2B).
#
# The decode corpus is the vendored scripts/setup/wikitext.txt (wikitext-2 test
# split, ~262k tokens); the decode config is pinned (n_ctx/n_batch 65536,
# n_ubatch 512), so re-runs on the same GPU reproduce bit-identical states.
#
# Usage (from the repository root, after scripts/compression/accuracy/build_server_cuda.sh):
#   ./scripts/setup/gen_prefill_states.sh <Qwen3-0.6B-FP16.gguf> [EXAONE-4.0-1.2B-FP16.gguf]
#
# Then push to the device with:
#   ./scripts/setup/push_files.sh .        # picks up ./states/*.kv

set -eu

cd "$(dirname "$0")/../.."

BIN=cuda_build/bin/mz_gen_state
CORPUS=scripts/setup/wikitext.txt
CONTEXTS="8193 16385 32700"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model1.gguf> [model2.gguf ...]" >&2
    echo "Example: $0 Qwen3-0.6B-FP16.gguf EXAONE-4.0-1.2B-FP16.gguf" >&2
    exit 1
fi
[ -x "$BIN" ] || { echo "ERROR: $BIN not found — run ./scripts/compression/accuracy/build_server_cuda.sh first" >&2; exit 1; }
[ -f "$CORPUS" ] || { echo "ERROR: corpus $CORPUS missing" >&2; exit 1; }

for GGUF in "$@"; do
    [ -f "$GGUF" ] || { echo "ERROR: model not found: $GGUF" >&2; exit 1; }
    for ctx in $CONTEXTS; do
        for fa in "" fa; do
            echo ""
            echo "=== $GGUF ctx=$ctx fa='${fa:-off}' ==="
            "$BIN" "$GGUF" "$CORPUS" "$ctx" $fa
        done
    done
done

echo ""
echo "--- generated states ---"
ls -lh states/*_8193.kv states/*_16385.kv states/*_32700.kv
