# mzCache — Artifact for "mzCache: On-Device LLM Memory Management under Multitasking" (MobiCom 2026)

mzCache is a restoration-oriented memory management system for on-device LLM
inference under mobile multitasking. Under memory pressure it elastically
evicts LLM memory (KV cache compressed/stored in fine-grained chunks, weight
layers unloaded); on the next inference request it restores everything
overlapped with GPU prefill, achieving 2.1–5.5× lower time-to-first-token
than a storage-backed partial-offload baseline.

This repository is the **artifact-evaluation package**: the mzCache engine
(a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), upstream
`fa4a9f2a`), the two baselines, and scripts that reproduce the paper's key
figures end-to-end and render paper-style plots.

The models are downloaded and quantized by `scripts/setup/get_models.sh`; the
prefill states come from the bundle linked in
[EVALUATION.md Section 1.2](EVALUATION.md#12-deployment-files-onto-the-device).

## Start here

**[EVALUATION.md](EVALUATION.md)** — step-by-step guide, organized by topic
(setup → TTFT → overlap breakdown → power & energy → compression → real-world
app), with troubleshooting at the end.

```bash
export ANDROID_NDK=<ndk-r25+> ADB_SERIAL=<device-serial>
./mzcache_build.sh mzcache_install_flexgen8bit new       # mzCache (OpenCL)
./exp1_partial_offload_build.sh exp1_partial_offload new # baseline: partial offload
./cpu_build.sh exp1_android_swap new                     # baseline: OS paging
./scripts/setup/get_models.sh                            # FP16 models
./scripts/setup/get_states.sh                            # prefill states (~17 GB)
./scripts/setup/push_files.sh                            # models + states -> device
./scripts/setup/gen_weight_layers.sh                     # per-layer weight files
./scripts/setup/device_setup.sh                          # governor + zram
./scripts/setup/run_device_profile.sh                    # one-time gate (~3 min)
./scripts/ttft/run_mzcache.sh                            # first experiment (TTFT)
python3 scripts/plot/plot_ttft.py                        # paper-style plot
```

## Repository layout

```
EVALUATION.md               reviewer guide (setup, TTFT, overlap breakdown, power & energy,
                            compression + F1 accuracy, real-world app)
mzcache/                    mzCache core library (compression kernels, chunked KV state,
                            weight unload/reload, device-profile-derived scheduling)
src/ ggml/ include/ common/ llama.cpp with #ifdef MZCACHE_SVM_KV_CHUNK hooks
                            (docs/INTEGRATION.md lists every hook)
examples/
├── mz_prefill*             swap-in TTFT benchmarks (single / repeated / power-sampled)
├── mz_save_state           prefill-state (.kv) creation
├── mz_device_profile       one-time device profiler (mandatory before mz_* runs)
├── mz_chat, mz_generate    functionality demos (interactive swap ladder; chunk growth)
├── exp1_android_swap       OS-paging baseline (CPU llama.cpp + token_embd mlock)
└── llama.android           Android app (Fig. 15): mzCache and OS-paging (cpu-swap) variants
                            (demo: real_world_demo_30x.mov; driver + live trace: scripts/mzcache/)
baselines/partial_offload/  vendored partial-offload baseline (self-contained tree)
scripts/
├── setup/                  device prep, file push, device profile
├── ttft/                   TTFT runners (mzCache / partial-offload / OS-paging) + memstress
├── overlap_breakdown/      per-stage breakdown + overlap ablation runner
├── power_energy/           power-trace runner
├── compression/            space/TTFT runner + accuracy/ (F1 roundtrip, dataset, profiles)
├── mzcache/                live app memory tracers (mem_trace*.py) + device helpers
└── plot/                   paper-style plot scripts (PNG out)
results/                    experiment output (gitignored)
Dockerfile / .dockerignore  reproducible host build env (EVALUATION.md Section 1.1)
mzcache_build.sh            mzCache OpenCL build + device push
exp1_partial_offload_build.sh / cpu_build.sh   baseline builds + push
```

## Requirements (summary)

Linux host with adb + Android NDK r25+; a rooted smartphone with a
**Snapdragon 8 Elite** (SM8750, Adreno 830) or **Snapdragon 8 Gen 3** (SM8650,
Adreno 750) SoC — support is per chipset, but for exact reproduction the
paper's devices are strongly recommended (OnePlus 12 and Galaxy S25+, 12 GB
RAM, UFS 4.0); Qwen3-0.6B /
EXAONE-4.0-1.2B in FP16 GGUF. Details in
[EVALUATION.md](EVALUATION.md#0-hardware).

## Vanilla build

All mzCache functionality is gated behind the CMake option
`MZCACHE_SVM_KV_CHUNK` (default **OFF**): without it the tree configures and
builds as vanilla llama.cpp
(`cmake -B build -DLLAMA_CURL=OFF && cmake --build build` — upstream's
`LLAMA_CURL` defaults to ON and needs `libcurl4-openssl-dev`, which the AE image
does not ship).

## License

MIT, following llama.cpp (base commit
`fa4a9f2a1ccda2573189a9d4995bdf0bceb41156`). Vendored Khronos OpenCL headers
under `examples/llama.android/.../CL/` are Apache-2.0.
