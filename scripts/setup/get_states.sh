#!/bin/bash
# get_states.sh — download the pre-generated prefill-state (.kv) bundle.
#
# The experiments restore from a prefilled KV cache. Instead of regenerating the
# 12 states on a GPU server (scripts/setup/gen_prefill_states.sh), download the
# author-hosted bundle and unpack it into ./states/. This removes the GPU-server
# dependency from setup (EVALUATION.md Section 1.2, Option 1).
#
# Usage (from the repository root):
#   ./scripts/setup/get_states.sh
#
# Downloads states.zip (~17 GB) from Google Drive and unpacks ./states/, which
# holds the 12 files the on-device runners expect:
#   <model_tag>_fa_<ctx>.kv   flash-attn ON  -> mzCache
#   <model_tag>_<ctx>.kv      flash-attn OFF -> partial_offload, OS-paging
# with <model_tag> in {qwen3_0.6B, exaone4_1.2B}, <ctx> in {8193, 16385, 32700}.
set -eu

# Author-hosted bundle (Google Drive). Share page:
#   https://drive.google.com/file/d/14vm4jhWsH7b0dsZdDoJPyt4McZGPHNUb/view
FILE_ID="14vm4jhWsH7b0dsZdDoJPyt4McZGPHNUb"
SHA256=""   # optional integrity check (sha256 of states.zip); empty = skip

# Check the 12 expected files BY NAME, never "any states/*.kv": Section 6.3 writes
# its own per-document states, and matching those would silently skip this download
# (or later pass a bundle that is actually incomplete).
expected_states() {
    local tag ctx
    for tag in qwen3_0.6B exaone4_1.2B; do
        for ctx in 8193 16385 32700; do
            echo "states/${tag}_fa_${ctx}.kv"
            echo "states/${tag}_${ctx}.kv"
        done
    done
}
missing_states() { expected_states | while read -r f; do [ -f "$f" ] || echo "$f"; done; }

if [ -z "$(missing_states)" ]; then
    echo "./states/ already has all 12 expected .kv files — nothing to do."
    exit 0
fi

URL="https://drive.usercontent.google.com/download?id=${FILE_ID}&export=download&confirm=t"
ZIP="states.zip"

# A complete zip may already be here (a manual download from the share link, or a
# previous run that unpacked partially) — only fetch when it is missing/damaged.
if [ -f "$ZIP" ] && unzip -t "$ZIP" >/dev/null 2>&1; then
    echo "$ZIP is already here and intact — skipping the download."
else
    echo "downloading $ZIP (~17 GB) from Google Drive (resumable — just re-run to continue)..."
    curl -fL -C - -o "$ZIP" "$URL"
    # Drive answers an exceeded quota / consent interstitial with an HTML page and
    # HTTP 200, which curl happily saves as states.zip. Catch it here, not hours later.
    unzip -t "$ZIP" >/dev/null 2>&1 || {
        echo "ERROR: $ZIP is not a valid zip (Google Drive likely returned a quota or consent page)." >&2
        echo "       Download it by hand from the share link above, drop it in this directory, and re-run." >&2
        exit 1
    }
fi

if [ -n "$SHA256" ]; then
    echo "$SHA256  $ZIP" | sha256sum -c -
fi

echo "unpacking to ./states/ ..."
unzip -q -o "$ZIP" -d .
rm -f "$ZIP"

miss=$(missing_states)
[ -z "$miss" ] || {
    echo "ERROR: the bundle is missing expected state files:" >&2
    echo "$miss" | sed 's/^/  /' >&2
    exit 1
}
echo "done: all 12 expected state files are in ./states/"
