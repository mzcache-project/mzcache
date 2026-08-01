# Reproducing the F1-accuracy experiment (Figure 14, accuracy)

This experiment measures how much the on-device KV-cache **compression** degrades
answer quality. For 30 long TriviaQA documents we compare the model's token-level
**F1** when it answers from:

- the **original** (uncompressed) KV cache, versus
- a KV cache that was **compressed and then decompressed on the phone** with one of
  the mzcache backends (`FLEXGEN_8BIT` or `CACHEGEN`).

The gap between the two is the accuracy cost of the compression. **A large gap is
not automatically a failure** — reproducing the paper means landing near its
numbers, and one of its cells is deliberately low:

| Qwen3-0.6B | original | CacheGen | 8-bit |
|---|---|---|---|
| paper F1 (Fig. 14) | 37.2 | **18.7** | 37.3 |

| EXAONE-4.0-1.2B | original | CacheGen | 8-bit |
|---|---|---|---|
| paper F1 (Fig. 14) | 37.5 | 37.2 | 38.2 |

CacheGen roughly halving F1 on Qwen3 is the paper's own result (§6.5: 8-bit
preserves accuracy better on Qwen3, likely due to its larger KV hidden dimension
and smaller model size), while both backends stay comparable on EXAONE. So the
pass criterion is closeness to the table, not smallness of the gap.

## Two machines, two roles

The experiment is split across two machines because prefilling long contexts and
scoring need an NVIDIA GPU, while the compression roundtrip must run on the phone.
**The same repository is cloned on both**; each machine only builds and runs its
half. Moving the `.kv` files between the machines is done **by hand** (`scp`,
`adb pull/push`) — no script crosses the machine boundary.

| | **GPU server** (NVIDIA + CUDA) | **Phone server** (this host: Linux + adb + rooted Qualcomm phone) |
|---|---|---|
| Builds | `mz_accuracy`, `mz_load_state_accuracy` (CUDA, `MZCACHE_BUILD_SERVER=ON`) | `mz_save_state_v2` (Android, per compression backend) |
| Does | ① prep dataset → ② prefill each doc into a `.kv` + score **original** F1 → ⑤ score **compressed** F1 | ④ compress+decompress each `.kv` on the phone |
| Hardware | ~8 GB VRAM is enough for Qwen3-0.6B / EXAONE-1.2B | rooted phone with a supported SoC (EVALUATION.md Section 0), governor pinned |

```
 GPU server                                   Phone server
 ───────────                                   ────────────
 ① prepare_triviaqa.py  ->  test.json
 ② mz_accuracy          ->  states/raw/*.kv  --(scp)-->  push to phone raw/
     (also: original F1)                          ④ roundtrip_phone.sh (per backend,
                                                      pins the reference profile)
                                                      raw/*.kv -> full_swap_*/*.kv
 ⑤ mz_load_state_accuracy  <--(scp)-------------  pull compressed *.kv
     (compressed F1)
```

The `.kv` files are named `<arch>_<type>_fa_<mzcache_id>.kv` (e.g.
`qwen3_0.6B_fa_14.kv`) on **every** step, so a document is tracked by its
`mzcache_id` (1..30) end to end — no renaming between tools.

---

## Prerequisites

**Both machines:** clone this branch.

```bash
git clone -b mzcache_v0.9_ae <this-repo-url> mzcache && cd mzcache
```

**GPU server:** CUDA toolkit + NVIDIA driver, cmake, a C++17 compiler, Python 3
with `datasets` (`pip install datasets`), and the model GGUF(s)
(`Qwen3-0.6B-FP16.gguf`, `EXAONE-4.0-1.2B-FP16.gguf` — obtain via
`scripts/setup/get_models.sh`, see EVALUATION.md Section 1.2).

**Phone server:** Android NDK, `adb`, a rooted phone with a supported SoC
(Snapdragon 8 Elite or 8 Gen 3 — see EVALUATION.md Section 0; the paper's Galaxy S25+
is strongly recommended since the reference profile was measured on it) at
`/data/local/tmp/mzcache`, and the same GGUF(s). See the top-level `README` /
`mzcache_build.sh` for the Android build environment. The full swap-out in
step ④ also unloads weight layers, so the per-layer weight files must exist on
the device first — generate them once with
`scripts/setup/gen_weight_layers.sh` (EVALUATION.md Section 1.2); the roundtrip
script checks and aborts with that hint if they are missing.

Pick one model + one backend and run the whole flow; repeat for the others. The
examples below use **Qwen3-0.6B** and the **CacheGen** backend.

---

## ① GPU server — prepare the dataset (once)

```bash
python3 scripts/compression/accuracy/prepare_triviaqa.py rc triviaqa_30docs_seed1234.json
```

Downloads TriviaQA, deterministically (seed 1234) selects 30 long documents, and
assigns each a stable `mzcache_id`. Every later step keys off this file, so
generate it once and reuse the same copy on both machines if you like (it is small).

## ② GPU server — build the CUDA tools

```bash
./scripts/compression/accuracy/build_server_cuda.sh          # -> cuda_build/bin/{mz_accuracy,mz_load_state_accuracy,mz_gen_state}
```

(These are plain llama.cpp state tools; they do not need the mzcache OpenCL core,
so the build uses `GGML_CUDA=ON MZCACHE_BUILD_SERVER=ON MZCACHE_SVM_KV_CHUNK=OFF`.)

## ② GPU server — prefill each document → raw `.kv` + original F1

`mz_accuracy` always writes its states into `./states/` relative to the working
directory, so run it from a **dedicated directory** — the repo's own `./states/`
holds the 12 prefill states from EVALUATION.md Section 1.2 and the two sets must
not mix:

```bash
mkdir -p f1_work && cd f1_work
../cuda_build/bin/mz_accuracy ../Qwen3-0.6B-FP16.gguf ../triviaqa_30docs_seed1234.json 64
```

For every document this prefills the context, greedily generates an answer, scores
its F1 against the gold answers (this is the **original / uncompressed** number),
and saves the KV cache to `f1_work/states/qwen3_0.6B_fa_<mzcache_id>.kv`. The
per-document original F1 is written to `f1_work/mz_accuracy_results.json`. `64` is
the max generated tokens per question.

**Disk**: these raw states are large and one is written per document — roughly
2 GB on average (up to ~3.6 GB at 32k tokens), so all 30 need **~60 GB**, on top of
everything in EVALUATION.md Section 0. The roundtrip in step ④ consumes them one at
a time, so you can delete each `f1_work/states/<doc>.kv` once its compressed
counterpart exists, or run the pipeline on a subset of documents.

## ③ Transfer raw `.kv` → phone server (manual)

On the GPU server, copy `f1_work/states/qwen3_0.6B_fa_*.kv` to the phone server
(e.g. `scp`) into a directory of your choice, say `./raw_qwen3/` — again **not**
the phone server's `./states/`, which holds the 12 prefill states. The roundtrip script
in step ④ pushes them to the phone one document at a time, so the phone only
ever needs ~one document (≤ 3 GB) of free space — not all 30.

## Reference device profile — pinned during step ④ (reproducibility)

mzcache schedules its swap-out/swap-in from a measured **device profile**
(`mzcache_device_profile_<BACKEND>.txt`, produced once by `mz_device_profile`).
The profile sets `decomp_load_ratio = decomp_gbps / (decomp_gbps + kv_read_gbps)`
— the fraction of KV chunks that are **lossy-compressed** at swap-out; the rest
are stored **losslessly**. That split is exactly what the accuracy experiment
measures, so **F1 is itself a function of the profile**: run the same roundtrip
with a profile measured on a different phone and you get a (legitimately)
different F1.

For the performance experiments (Figures 9–13), using *your* device's profile is
the point. For the accuracy experiment, however, the paper's numbers are only
reproducible under the paper's chunk split. The roundtrip script therefore
**pins the authors' reference profiles** — captured on the paper's Galaxy S25+
and vendored at `scripts/compression/accuracy/profiles/` — on the device for the duration of
step ④, and **restores your own profile on exit** (including Ctrl-C). Your
Figure-9–13 setup is untouched afterwards.

If you run step ④ by hand instead of via the script, do the same swap yourself:
back up `/data/local/tmp/mzcache/mzcache_device_profile_<BACKEND>.txt`, push the
matching file from `scripts/compression/accuracy/profiles/`, and restore your backup when
done.

## ④ Phone server — build + compress/decompress roundtrip

Build `mz_save_state_v2` for the backend you are testing (each backend is a
separate compile-time build installed to its own device dir):

```bash
# CacheGen
MZCACHE_COMPRESSION=CACHEGEN     ./mzcache_build.sh mzcache_install_cachegen
# or FLEXGEN-8bit
MZCACHE_COMPRESSION=FLEXGEN_8BIT ./mzcache_build.sh mzcache_install_flexgen8bit
```

Run the roundtrip for all 30 documents with the wrapper script:

```bash
ADB_SERIAL=<serial> ./scripts/compression/accuracy/roundtrip_phone.sh \
    qwen3 cachegen ./raw_qwen3 ./compressed_cg mzcache_install_cachegen
```

The script pins the reference device profile (previous section), then per
document: pushes the raw `.kv` → runs `mz_save_state_v2 <gguf> <mzcache_id> 0`
(full swap-out — compress + store — then swap-in — restore + decompress;
`remaining = 0` means fully evict) → pulls the reconstructed state into
`./compressed_cg/` → deletes the on-device copies. On the device the outputs
pass through `accuracy_test/full_swap_cg/` (CacheGen) or
`accuracy_test/full_swap_fg8bit/` (FLEXGEN-8bit). The script is resume-safe —
documents whose output already exists are skipped — so an interrupted run can
simply be re-invoked, and it restores your own device profile on exit.

## ⑤ Transfer compressed `.kv` back + score compressed F1 (GPU server)

The reconstructed states are already on the phone server (`./compressed_cg/`
from step ④). Copy them to the GPU server, e.g. into `states_cg/`:

```bash
# phone server
scp ./compressed_cg/*.kv gpu-server:mzcache/states_cg/
```

On the GPU server, score F1 over the same 30 documents using the compressed states:

```bash
./cuda_build/bin/mz_load_state_accuracy Qwen3-0.6B-FP16.gguf ./states_cg triviaqa_30docs_seed1234.json 64
```

It scans the directory for `<...>_<mzcache_id>.kv`, loads each state, generates an
answer, and scores F1 against the same gold answers — this is the **compressed**
number. Compare it to the original F1 from step ②: the gap is the accuracy cost of
the compression backend (Figure 14, accuracy panel).

---

## Notes

- **Backends and models are independent builds.** Repeat steps ②/④/⑤ per
  `(model, backend)` cell you want in the figure. FLEXGEN-8bit writes to
  `full_swap_fg8bit/`, CacheGen to `full_swap_cg/`; keep them in separate GPU-side
  directories (`states_fg8bit/`, `states_cg/`) so `mz_load_state_accuracy` scores
  one backend at a time.
- **KV precision.** The experiment uses fp16 KV (flash-attention on), so the state
  files are named `..._fa_<id>.kv` with no quantization suffix. All three tools
  agree on this name, which is why a document is addressed by `mzcache_id` alone.
- **`mzcache_id` is the join key.** Do not renumber or reshuffle it — the phone
  names outputs by the id it was given, and the GPU scorer matches states back to
  documents by the same id.
- **Context size.** All three tools use a 32k context (`n_ctx = 32768`), which
  covers the longest test documents. For Qwen3-0.6B that KV cache is ~3.6 GB, so
  an 8 GB GPU is enough on the server side.
- **Transfers are manual by design.** Any `scp`/`rsync`/`adb` path works; the
  scripts here never reach across machines.
