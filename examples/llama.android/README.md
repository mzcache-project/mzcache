# llama.android + mzCache

Android app that restores a 32700-token context-aware prefill state and
evicts it under memory pressure using mzCache.

- `onTrimMemory` / `onLowMemory` → `handleMemoryPressureNative` →
  `mzcache_core::swapout(cur_ratio - step)` (compress + store + close +
  unload weight layers). Each pressure event lowers the resident ratio by
  `step` — 0.15 by default (1.0 → 0.85 → 0.70 → ...), settable at runtime with
  `adb shell setprop debug.mzcache.step <frac>` (read once per app start).
- **Send** → `chat_completion_init` → `mzcache_core::swapin_generate(...)` when the
  KV was swapped out (restore overlapped with prefilling the new turn), then the
  ratio resets to 1.0. Repeated cycles are supported (see
  `examples/mz_prefill_repeated`).

## Prerequisites

1. **OpenCL library from the device** (not tracked in git). The GGML OpenCL
   backend links against the vendor driver:
   ```
   adb pull /vendor/lib64/libOpenCL.so llama/src/main/jniLibs/arm64-v8a/libOpenCL.so
   ```
   The library is used at build time only (`packaging` excludes it from the
   APK; the app loads the device's own copy via
   `<uses-native-library android:name="libOpenCL.so"/>`).

2. **OpenCL headers**: the minimal set the build reaches (`cl.h` plus its
   transitive includes `cl_platform.h` and `cl_version.h`, Apache-2.0
   Khronos headers) is vendored under `llama/src/main/cpp/include/CL/`.

3. **Prefilled KV state file** (not tracked in git) created by `examples/mz_prefill` /
   `examples/mz_save_state` on the device, e.g.
   `/data/local/tmp/mzcache/states/qwen3_0.6B_fa_32700.kv` (~3.7 GB for
   Qwen3-0.6B). On non-rooted devices the app cannot read
   `/data/local/tmp`; copy the file into the app-private fallback instead:
   ```
   adb push qwen3_0.6B_fa_32700.kv /data/local/tmp/
   adb shell run-as com.example.llama sh -c 'mkdir -p files/states && cp /data/local/tmp/qwen3_0.6B_fa_32700.kv files/states/'
   ```

4. **Model gguf** (not tracked in git): pushed to `/data/local/tmp/gguf/`, e.g.
   `/data/local/tmp/gguf/Qwen3-0.6B-FP16.gguf` — the app reads it directly from
   there, like the state file above. Obtain it from HuggingFace
   (`Qwen3-0.6B-FP16`) or quantize it from the checkpoint. `mzcache_core` only
   knows the per-layer weight sizes of `Qwen3-0.6B` and `EXAONE-4.0-1.2B` (see
   `mzcache_core.cpp`).

5. **Root (optional)**: GPU memory sampling
   (`/sys/devices/virtual/kgsl/kgsl/proc/<pid>/gpumem_mapped`) uses libsu and
   requires a rooted device; without root the CSV column is `-1`.

Weight layer dumps (`files/layers/Qwen3-0.6B_layer_<i>.bin`, ~880 MB total)
are generated automatically on the first model load.

## Notes

- mzCache performs O_DIRECT file I/O relative to the CWD; the JNI
  `mz_init()` chdir()s into the app's **internal** `filesDir` (external
  storage is FUSE-backed and rejects O_DIRECT).
- **Send** is the swap-in trigger: it continues the loaded context (it does not
  clear the KV cache), which is exactly what the Fig. 15 workload measures.
- Build variants: default = mzCache, `-PmzCpuSwap=true` = CPU-only OS-paging
  baseline (its own package, `com.example.llama.cpuswap`). `-PmzBaseline=true`
  still exists for a GPU-resident/no-trim control build, but it keeps the
  **`com.example.llama` application id**, so installing it silently replaces the
  mzCache app — rebuild the default variant afterwards.
- Building needs the Android SDK (`ANDROID_HOME` or `local.properties`) with
  cmake 3.22.1 installed, plus JDK 17 — the mzCache Docker image does not carry
  the Gradle/SDK stack.
- A 100 ms memory CSV sampler writes to the app's external files dir
  (`mem_<timestamp>.csv`) for evaluation.
- The PSI watcher (`libpsiwatcher`) is included but its startup block in
  `MainActivity.onCreate` is commented out; enable it to trigger eviction
  from PSI stall events instead of `onTrimMemory`.
