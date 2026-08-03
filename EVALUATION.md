# mzCache — Artifact Evaluation Guide

This guide reproduces the core experiments of the paper on commodity
smartphones. Each section is named by topic, with the paper figure in
parentheses:

| Section | Paper figure | Claim exercised | Automated? |
|---|---|---|---|
| [0. Hardware](#0-hardware) | — | phone host, smartphone (by SoC), optional GPU server | — |
| [1. Setup](#1-setup) | — | build (Docker or from scratch) + deploy models/states/weights | scripts |
| [2. Offline device profile](#2-offline-device-profile-mandatory-gate) | — | one-time per-device, per-backend gate | script |
| [3. TTFT](#3-ttft-paper-fig-9) | Fig. 9 | mzCache restores with substantially lower TTFT than Partial Offload; OS paging is far worse | scripts + plot |
| [4. Overlap breakdown](#4-overlap-breakdown-paper-fig-10--11a) | Fig. 10, 11(a) | per-stage wins (alloc/reload/prefill) and the overlap ladder | script + plots |
| [5. Power & energy](#5-power--energy-paper-fig-13) | Fig. 13 | shorter swap-in ⇒ less energy despite higher peak power | script + plot |
| [6. Compression algorithm](#6-compression-algorithm-paper-fig-14--f1-accuracy) | Fig. 14 | CacheGen vs 8-bit: space savings, TTFT, **and F1 accuracy** | scripts + plot (F1: numbers only) |
| [7. Real-world app](#7-real-world-android-app-paper-fig-15) | Fig. 15 | app under real multitasking pressure (eviction + seamless restore) | app builds + driver script |
| [8. Notes](#8-measurement-notes--troubleshooting) | — | pitfalls that matter for reproduction | — |

Sections 3-6 write their CSVs under `results/` and each has a plot script that
renders a paper-style PNG next to them. Section 7 is a recorded demo instead — it
produces a video and a survived/killed verdict, not a CSV.

---

## 0. Hardware

Up to **three machines**: a phone host and a smartphone (both required), plus a
GPU server used only by the F1-accuracy panel (Section 6.3).

**Phone host** — an x86-64 Linux box that builds the binaries and drives the
phone over USB:
- Ubuntu 22.04 or similar, with `adb` (Android platform tools) installed and the
  phone visible in `adb devices`
- **~70 GB free disk**: NDK (~7 GB unpacked) + build trees + models (~7 GB, BF16
  downloads included) + the state bundle, which needs ~38 GB transiently while the
  ~18 GB zip is unpacked beside itself. The optional F1 experiment (Section 6.3)
  needs a further ~60 GB for its per-document states
- the phone connected by USB; every on-device step goes through adb
- Section 7 additionally needs a **graphical desktop session** on this machine
  (scrcpy mirror + live plot are screen-recorded)

**Smartphone** — a **rooted** Android phone with one of the two supported SoCs,
detected automatically from `ro.soc.model`:

- **Snapdragon 8 Elite** (SM8750\*, 2 prime + 6 performance cores, Adreno 830)
- **Snapdragon 8 Gen 3** (SM8650\*, 1 + 5 + 2 cores, Adreno 750)

Any other SoC only prints a warning and falls back to the 8 Gen 3 core placement,
so it still produces numbers — do not report those as reproductions.

`su` must work from `adb shell` (governor/THP/zram control, battery sysfs, mlock
beyond `RLIMIT_MEMLOCK`), and the phone needs **~30 GB free**: models + states
(~25 GB), the per-layer weight files (~2.9 GB), the pushed install dirs, and a
512 MB profiling scratch file (+7.5 GB if you run the optional memstress files).

For **exact reproduction of the paper's numbers we strongly recommend the paper's
devices**: the **Galaxy S25+** (SM-S936N, 8 Elite, 12 GB, UFS 4.0) is the main
device used across all experiments, and the **OnePlus 12** (8 Gen 3, 12 GB, UFS
4.0) is additionally used for the TTFT experiment.

**GPU server (Section 6.3 only).** An NVIDIA GPU that prefills and scores the F1
panel; we used an **RTX 3090** (~8 GB VRAM suffices, and matching the GPU helps —
F1 shifts slightly with GPU architecture). If the phone host has an NVIDIA GPU,
use it directly; no separate machine is needed.

---

## 1. Setup

Run every host command from the repository root. Set the device serial first —
every on-device script reads it:

```bash
export ADB_SERIAL=<device-serial>     # see `adb devices`
```

The remaining knobs (defaults in `scripts/common.sh`) only matter when you
deviate from the defaults:

| Variable | Default | Needed when |
|---|---|---|
| `ANDROID_NDK` | — | every shell that builds (Section 1.1) |
| `MZ_INSTALL` | `mzcache_install_flexgen8bit` | you rename the install dir, or select the CacheGen build |
| `MODEL_TAG` + `GGUF` | `qwen3_0.6B` + `/data/local/tmp/gguf/Qwen3-0.6B-FP16.gguf` | **running Sections 3-5 on EXAONE instead of Qwen3** |

```bash
# example: the same TTFT sweep on the second model
MODEL_TAG=exaone4_1.2B GGUF=/data/local/tmp/gguf/EXAONE-4.0-1.2B-FP16.gguf \
    ./scripts/ttft/run_mzcache.sh
```

Everything on the device lives under two directories:

```
/data/local/tmp/mzcache/<install-dir>/bin/   binaries (pushed by mzcache_build.sh)
/data/local/tmp/mzcache/states/              prefill .kv states
/data/local/tmp/mzcache/layers/              per-layer weight files
/data/local/tmp/gguf/                        FP16 models
```

### 1.1 Building (host)

Build inside the provided **Docker image** (recommended — the whole toolchain is
pinned) or install the toolchain yourself. Either way the same build scripts run
afterwards.

**Option A — Docker (recommended).** The image carries the CLI toolchain (NDK,
`adb`, plotting env) and reaches your phone through the host's adb server:

```bash
adb start-server                             # the container talks to THIS server
docker pull appleyu1/mzcache-ae:v0.9
docker run --rm -it --network host -v "$PWD:/workspace" \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e ADB_SERIAL="$ADB_SERIAL" appleyu1/mzcache-ae:v0.9
adb devices                                  # sanity check: your phone must be listed
# ANDROID_NDK is already set; continue with "Build mzCache" below.
```

The container has no USB access of its own: it reaches the phone through the
host's adb server over `--network host` (Linux-only), so leave that server
running — an empty `adb devices` inside the container almost always means the
host server is not up. `--user` keeps everything the build writes into the repo
owned by you instead of root. **Section 7 cannot run in the container**: the
Android app needs the Gradle/SDK stack and the demo needs a desktop session, so
build and record that one on the host.

**Option B — build from scratch.** On a clean Ubuntu (22.04/24.04):

```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake ninja-build \
                     git curl unzip adb file ca-certificates libcurl4-openssl-dev \
                     python3 python3-matplotlib python3-numpy    # last two only for the plots
./scripts/setup/get_ndk.sh                     # downloads + verifies NDK r29 (~780 MB)
export ANDROID_NDK=$PWD/android-ndk-r29        # the export line the script prints
```

**Build mzCache.** Built once per compression backend (each gets its own profile,
Section 2). Build the 8-bit backend now; add CacheGen only if you'll run Section 6:

```bash
MZCACHE_COMPRESSION=FLEXGEN_8BIT ./mzcache_build.sh mzcache_install_flexgen8bit new
MZCACHE_COMPRESSION=CACHEGEN     ./mzcache_build.sh mzcache_install_cachegen new   # optional, for Section 6
```

Keep the phone connected: the build pushes the binaries to the device and, the
first time, pulls the phone's `libOpenCL.so` into `opencl_sdk/`. That copy is
reused afterwards — **delete `opencl_sdk/lib/libOpenCL.so` before building for a
different phone**. The two baselines build later (Section 3), the accuracy tools
on the GPU server (Section 6.3).

### 1.2 Deployment (files onto the device)

Run every command below **on the phone host**, except the one block marked
"on the GPU server" (states Option 2).

**Models (FP16 GGUF).** Quantize from the upstream BF16 (CPU-only, no GPU — and no
phone either, so this can run while you are still setting the device up):

```bash
./scripts/setup/build_quantize.sh          # build llama-quantize once (skip if you built the CUDA tools)
./scripts/setup/get_models.sh              # download BF16 + quantize -> ./Qwen3-0.6B-FP16.gguf + ./EXAONE-4.0-1.2B-FP16.gguf
```

The BF16 downloads stay next to the FP16 output; only the FP16 models are pushed
to the device.

**Prefill states (`.kv`).** 12 files: 2 models × ctx {8193, 16385, 32700} ×
flash-attn {on, off}. Either download or regenerate:

- **Option 1 — download the bundle (recommended; on the phone host, no GPU).**
  Fetch the ~18 GB bundle from
  [Google Drive](https://drive.google.com/file/d/14vm4jhWsH7b0dsZdDoJPyt4McZGPHNUb/view)
  and unpack it into `./states/` (~20 GB unpacked):
  ```bash
  ./scripts/setup/get_states.sh
  ```
  Expect ~25 min on a good link. The script then verifies the archive before
  unpacking, which on an 18 GB zip takes a few minutes with no output — it has not
  hung.
- **Option 2 — regenerate on a GPU server** (deterministic, seed-pinned). Run this
  whole block **on the GPU server** — a separate machine with an NVIDIA GPU and
  this repo checked out — then copy the states back to the phone host:
  ```bash
  # on the GPU server:
  ./scripts/compression/accuracy/build_server_cuda.sh    # CUDA prefill tool (+ llama-quantize)
  ./scripts/setup/get_models.sh                          # the gguf models to prefill from
  ./scripts/setup/gen_prefill_states.sh Qwen3-0.6B-FP16.gguf EXAONE-4.0-1.2B-FP16.gguf   # -> ./states/*.kv
  scp -r ./states <user>@<phone-host>:/path/to/mzcache-ae/   # into the phone host's repo root
  ```

Either way `./states/` must end up with **12 `.kv` files** in the repo root, next
to the `./*.gguf` models. Push both to the device — no argument needed:

```bash
./scripts/setup/push_files.sh   # pushes ./*.gguf + ./states/*.kv, then verifies both models
```

The closing checklist is the setup gate: every line must read `[ok]` before you
continue.

**Per-layer weight files (once per model).** Required before any swap-out
experiment — without them the engine silently skips weight unloading. Generate
on the device:

```bash
./scripts/setup/gen_weight_layers.sh    # both models, ~840 MB + ~2 GB on device
```

The runners check for these and abort if missing. The app (Section 7) makes its own.

**Device preparation.** Re-run after every reboot — one command applies the
performance governor and the zram (zstd/8 GiB) setting:

```bash
./scripts/setup/device_setup.sh
```

---

## 2. Offline device profile (mandatory gate)

The experiment binaries refuse to start without a device profile — one per
device, per compression backend. Run it for each backend you built (Section 1.1):

```bash
./scripts/setup/run_device_profile.sh                                        # 8-bit build, ~3 min
MZ_INSTALL=mzcache_install_cachegen ./scripts/setup/run_device_profile.sh    # CacheGen, if built (Section 6)
```

It ends by printing the profile it just wrote, and leaves a 512 MB scratch file
on the device. To re-profile, delete the profile first:

```bash
adb -s $ADB_SERIAL shell 'rm /data/local/tmp/mzcache/mzcache_device_profile_*.txt'
```

The two derived constants can be overridden at **experiment** run time (they are
read by the experiment binaries, not by this script):
`MZCACHE_PER_LAYER_BALANCE=<int> MZCACHE_DECOMP_LOAD_RATIO=<0..1>`.

---

## 3. TTFT (paper Fig. 9)

The headline experiment: time-to-first-token when three systems each restore
inference state from a given remaining-in-memory level and serve an 8-token
query — the metric is TTFT *including* restoration. mzCache (built in Section 1.1)
is compared against two storage-backed baselines, each built in its own
subsection below.

### 3.1 mzCache

```bash
./scripts/ttft/run_mzcache.sh                       # full sweep: 3 ctx × 5 levels × 3 cycles
# subset example: -c "16385" -f "0 0.5" -r 3        # -o <dir> to keep it in its own CSV
```

Per (ctx, fp) the device runs `mz_prefill_repeated`: load state →
`swapout(1-fp)` → `swapin_generate` (restoration overlapped with the 8-token
prefill). Output: `results/ttft/ttft_mzcache_<device>_<model>.csv`.

> **Each invocation rewrites its CSV from scratch** (both this runner and 3.2), so
> splitting a sweep into subsets keeps only the last one — give each subset its own
> `-o <dir>`. The OS-paging runner in 3.3 is the exception: it *appends*, so
> re-running the same context duplicates rows in the same file.

### 3.2 Partial Offload

Build the vendored baseline (GPU), then run the sweep:

```bash
./exp1_partial_offload_build.sh exp1_partial_offload new
./scripts/ttft/run_partial_offload.sh
```

It maps (ctx, ratio) to a (weight_layers, kv_layers) combination internally and
prints `TTFT: ... ms` including KV + weight reload from storage. Storage reads are
measured cold (`drop_caches` per rep by default; `-d once|never` to change).
Output: `results/ttft/ttft_partial_offload_<device>_<model>.csv`.

### 3.3 OS Paging (zRAM)

Build the CPU-only baseline (stock llama.cpp + `token_embd` mlock):

```bash
./cpu_build.sh exp1_android_swap new
```

Resident points (remaining = 100%) are fully automatic:

```bash
./scripts/ttft/run_os_paging.sh                     # -m resident (default)
```

Output (both modes, appended): `results/ttft/ttft_os_paging_<device>_<model>.csv`.

The **full-eviction endpoint is fully automatic** (`evict-full`): it applies
memory pressure in two stages — `mmap_touch` (file-cache) first, then a ramped
`anon_hog` (dirty-anonymous, capped at `-A` MiB ≈ MemTotal − 2 GiB) — until the
process's resident weight + KV Rss falls to `-T` percent of the warm baseline
(default 10), then measures TTFT there. This mode has **no**
`MemAvailable`/`SwapFree` guard: it deliberately pushes RSS toward 0 and accepts
the reboot/OOM risk.

```bash
./scripts/ttft/memstress/setup_memstress.sh   # once: needs $ANDROID_NDK; writes ~7.5 GB of stress files
./scripts/ttft/run_os_paging.sh -m evict-full -c 16385        # one context per invocation (-r 3 default)
```

If the phone reboots mid-run, lower the anonymous-pressure cap — `-A 5120` is
known to complete all three contexts on a 12 GB device. A rep whose eviction does
not reach the target is reported and **skipped** rather than recorded.

> **Two different "remaining" numbers are printed.** The ramp reports its stop
> metric (`-T`, plain resident Rss, e.g. `-> remaining ~10%`); the measurement
> reports the plotted x-axis value (`at remaining 48%`), which excludes the mlocked
> embedding and counts the KV still held compressed in zram. Both are correct —
> see Section 8 for the formula.

> Anonymous pressure is far more aggressive than file pressure; on large-KV
> contexts it can briefly stall the device (a transient adb drop). Drive each
> context — ideally each rep — in its own invocation and re-run any that the
> host loses, so a stall costs only that measurement.

Intermediate points are **operator-guided** (`-m evict`): after each pressure
step the script prints the resident weight/KV Rss and remaining-% and asks
whether to measure, continue, or quit. It reads that answer from the terminal, so
this mode must run in a real shell (not backgrounded). It uses file pressure only
and keeps the `MemAvailable`/`SwapFree` guard, which stops the ramp before
over-pressure can reboot the device.

### 3.4 Plot

```bash
python3 scripts/plot/plot_ttft.py       # -> results/ttft/ttft_<device>_<model>.png
```

One figure per (device, model): TTFT vs remaining-% per context, three marked
series, and a red dash-dot line at OS paging's full-eviction point. The expected
ordering at every point is **mzCache < Partial Offload ≪ OS Paging (evicted)**.

---

## 4. Overlap breakdown (paper Fig. 10 & 11a)

Per-stage swap-in breakdown (alloc / reload / prefill) and the overlap ablation.
Paper setting: Galaxy S25+, Qwen3-0.6B, remaining = 0.5. **Requires the Partial
Offload baseline from Section 3.2** — the runner measures it alongside mzCache and
aborts if it is not on the device.

```bash
./scripts/overlap_breakdown/run_overlap.sh   # 1 discarded warm-up + 3 reps per config
                                             # -c <ctx> -R <remaining> -r <reps> -o <dir>
python3 scripts/plot/plot_breakdown.py       # -> results/overlap_breakdown/breakdown.png
python3 scripts/plot/plot_overlap.py         # -> results/overlap_breakdown/overlap.png
```

Output: `results/overlap_breakdown/overlap_mzcache.csv` (one row per
`ctx,type,rep` with `alloc_ms,restore_ms,prefill_ms,total_ms`) and
`breakdown_po.csv`. The runner sweeps all four overlap combinations itself; in the
CSV, `type` 3 = `mz_NO` (nothing overlapped), 2 = `mz_AR`, 1 = `mz_RP`,
0 = `mz_ARP` (= mzCache, fully overlapped).

Partial Offload's reload time is the most run-to-run variable number here, so
judge the ordering and the compute stages rather than its absolute value. The
script discards a cold warm-up per config; if late runs look inflated, cool down
~5 min.

---

## 5. Power & energy (paper Fig. 13)

Power trace during swap-in: a shorter swap-in draws higher peak power but less
total energy. Paper setting: Galaxy S25+, Qwen3-0.6B, ctx 32700, remaining =
0.5. It samples the battery gauge every 100 ms around the swap-in (needs `su`),
and — like Section 4 — **requires the Partial Offload baseline from Section 3.2**.

**The phone must be physically unplugged — over USB the gauge reads ~0**, because
the phone runs off VBUS and `current_now` stays near zero. Switch to wireless adb
before measuring:

```bash
adb -s $ADB_SERIAL shell ip route get 1.1.1.1     # note the phone's "src" IP
adb -s $ADB_SERIAL tcpip 5555                     # still on USB
# unplug the cable now
adb connect <phone-ip>:5555
export ADB_SERIAL=<phone-ip>:5555
adb -s $ADB_SERIAL shell dumpsys battery | grep -E 'USB powered|status'   # expect false / 3 (discharging)
```

The phone and the host must be on the **same subnet** (a phone on 192.168.1.x
cannot be reached from a host on 192.168.0.x). Rebooting the phone drops tcpip
mode — re-enable it over USB.

```bash
./scripts/power_energy/run_power.sh        # -x <ctx> -R <remaining> -o <dir>
python3 scripts/plot/plot_power_energy.py  # -> results/power_energy/power_energy.png
```

Output: `results/power_energy/power_mzcache.csv` and `power_partial_offload.csv`.
Compare the two traces only within one back-to-back session (absolute wattage
drifts with charge state) — the shape is the result, not the absolute watts: on
the paper's Galaxy S25+, Partial Offload ramps ~8-15 W for ~3.0-3.2 s while
mzCache plateaus higher (~16-19 W) for only ~0.8 s, i.e. a higher peak over a
much smaller area. If `Power_W` comes out implausibly small
(~0.003 W), your gauge reports mA rather than µA — multiply by 1000; the shape and
duration are unaffected (Section 8).

---

## 6. Compression algorithm (paper Fig. 14 + F1 accuracy)

Compares the two KV-compression backends, **CacheGen** and **FLEXGEN-8bit**,
along three axes: **space savings** and **TTFT** (Section 6.2, on the phone) and
**F1 accuracy** (Section 6.3, phone + GPU server). All three are the same compression
question, so they live together under `scripts/compression/`.

### 6.1 Prerequisites

Both compression backends must be **built** (Section 1.1) and **profiled** (Section 2):
`mzcache_install_flexgen8bit` and `mzcache_install_cachegen`, each with its own
`mzcache_device_profile_<COMPRESSION>.txt`. If you only built and profiled the
8-bit backend earlier, do CacheGen now:

```bash
MZCACHE_COMPRESSION=CACHEGEN ./mzcache_build.sh mzcache_install_cachegen new
MZ_INSTALL=mzcache_install_cachegen ./scripts/setup/run_device_profile.sh
```

The space-savings panel covers **both models**, so EXAONE also needs its gguf,
its 16k states, and its per-layer weight files on the device (Section 1.2).
Paper setting: Galaxy S25+, 16k context.

### 6.2 Space savings and TTFT

```bash
./scripts/compression/run_compression.sh all   # space savings (Qwen3+EXAONE) + TTFT (Qwen3)
python3 scripts/plot/plot_compression.py       # -> results/compression/compression.png
```

The mode (`space` | `ttft` | `all`) is the **first** argument, before any flag:

```bash
./scripts/compression/run_compression.sh ttft -T exaone4_1.2B -o results/compression_exaone
```

- **Space savings** (left panel): `mz_prefill` at full eviction (fp=0) prints
  `[SpaceSavings] ... space_savings_pct=<x>` per backend →
  `results/compression/space_savings.csv`.
- **TTFT** (right panel): same `mz_prefill` `Prefill time` at remaining
  {0, 25, 50}% for both backends → `results/compression/ttft.csv` (Qwen3 by
  default; re-running with `-T exaone4_1.2B` overwrites it unless you pass `-o`).
  The plot's y-axis auto-scales to the measured values.

Paper values for the space-savings panel (Fig. 14), for comparison:

| Space savings | CacheGen | 8-bit |
|---|---|---|
| Qwen3-0.6B | 81.3% | 46.9% |
| EXAONE-4.0-1.2B | 71.8% | 46.9% |

8-bit is a fixed ratio by construction; CacheGen is data-dependent, so it moves a
little with the document set.

### 6.3 F1 accuracy (rightmost panel of paper Fig. 14)

The third axis of the compression comparison: how much the lossy KV
compression degrades answer quality (token-level F1 over 30 long TriviaQA
documents). This one needs an **NVIDIA GPU server** in addition to the phone
(the phone runs the compress/decompress roundtrip, the GPU server prefills and
scores), so it has its own step-by-step guide:

**→ [docs/reproduce_f1_accuracy.md](docs/reproduce_f1_accuracy.md)**

Paper values (Fig. 14). Note that **CacheGen roughly halving F1 on Qwen3 is the
expected result**, not a failed run — the paper's §6.5 attributes it to Qwen3's
larger KV hidden dimension and smaller model size, while both backends stay
comparable on EXAONE:

| F1 | original | CacheGen | 8-bit |
|---|---|---|---|
| Qwen3-0.6B | 37.2 | **18.7** | 37.3 |
| EXAONE-4.0-1.2B | 37.5 | 37.2 | 38.2 |

Budget **~60 GB** for the raw per-document states this produces (≈2 GB each, on
top of the Section 0 figures); they can be deleted per document once compressed.
There is no plot script for this panel — compare the printed F1 numbers directly
(`plot_compression.py` draws only the space-savings and TTFT panels).

The phone-side tooling lives under `scripts/compression/accuracy/`
(`prepare_triviaqa.py`, `roundtrip_phone.sh`, `profiles/`). For reproducibility
the roundtrip **pins the authors' reference device profile**
(`scripts/compression/accuracy/profiles/`) during the on-phone compression —
the profile decides the lossy/lossless chunk split, and therefore the F1
itself — and restores your own profile afterwards.

---

## 7. Real-world Android app (paper Fig. 15)

mzCache's real-world benefit — surviving heavy multitasking pressure by evicting
KV (and weights) on `onTrimMemory` and restoring on return, where the OS-paging
baseline is LMK-killed and cold-starts — is captured in the demo video below
(`real_world_demo_30x.mp4`, 30× speed). It stacks the mzCache and OS-paging
variants under the same app-switching workload with a live memory trace: mzCache
steps its GPU working set down on each backgrounding and swaps back in on return
(fast first token), while the OS-paging app is LMK-killed and reloads cold.

https://github.com/user-attachments/assets/3925f5db-1e82-4838-b98f-e9cf8accd564

This section documents how to **build the app** (Section 7.1) and **reproduce the
app-switch workload** yourself (Section 7.2).

### 7.1 Build variants (`examples/llama.android`)

Build **on the host, not in the Docker image** — this needs the Android SDK
(`ANDROID_HOME`, or `sdk.dir` in `examples/llama.android/local.properties`) with
cmake 3.22.1 installed, plus JDK 17. Build and install both variants:

```bash
cd examples/llama.android
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export ANDROID_HOME=$HOME/Android/Sdk           # wherever your SDK lives
rm -rf llama/.cxx app/.cxx    # REQUIRED when switching variants (cmake args change)
./gradlew assembleDebug [variant flag]
adb -s $ADB_SERIAL install -r app/build/outputs/apk/debug/app-debug.apk
```

| Variant | Flag | Package | Meaning |
|---|---|---|---|
| **mzCache** | *(none, default)* | `com.example.llama` | OpenCL + SVM-chunked KV + FLEXGEN-8bit; `onTrimMemory` → swapout ladder, return → `swapin_generate` |
| **OS paging** | `-PmzCpuSwap=true` | `com.example.llama.cpuswap` | CPU-only stock llama.cpp; weights mmapped, KV anonymous (zram); trim signals ignored |

The two variants have different package names, so both can stay installed.

Prerequisites — the last three are already satisfied if you ran Section 1.2;
`push_files.sh` places and `chmod`s them:
- `llama/src/main/jniLibs/arm64-v8a/libOpenCL.so` — pull once from the device
  (`adb -s $ADB_SERIAL pull /vendor/lib64/libOpenCL.so llama/src/main/jniLibs/arm64-v8a/`);
  not distributed in the repo. Required by the mzCache variant (the `-PmzCpuSwap`
  build is CPU-only and links fine without it)
- the 32700-token states in `/data/local/tmp/mzcache/states/`: mzCache loads
  `qwen3_0.6B_fa_32700.kv`, the cpu-swap variant the non-fa `qwen3_0.6B_32700.kv`
- the model gguf in `/data/local/tmp/gguf/`
- for the OS-paging variant, zram set to zstd via `scripts/setup/device_setup.sh`

The mzCache eviction step is a runtime property, read once per app start (the
Section 7.2 driver sets it for you — this is only for driving the app by hand):

```bash
adb -s $ADB_SERIAL shell setprop debug.mzcache.step 0.15   # %p per trim signal
adb -s $ADB_SERIAL shell am force-stop com.example.llama   # restart to pick it up
# verify in logcat: "mz_init: ... swapout step = 0.15"
```

### 7.2 Reproducing the app-switch workload (`scripts/mzcache/record_demo_rounds.sh`)

Reproducing the paper's real-world experiment as published takes ~10 app-switch
rounds and roughly an hour of continuous, by-hand interaction. We reduced it to
**3 rounds** and recorded ourselves running those 3 rounds; that recording, sped
up 30×, is the **`real_world_demo_30x.mp4`** embedded above.

Run the 3-round workload yourself — once per variant, ~14 min each:

```bash
./scripts/mzcache/record_demo_rounds.sh mzcache     # or: baseline
```

It drives the app-switch gauntlet on the phone, mirrors the screen (scrcpy) next
to a live memory trace (`mem_trace*.py`), and screen-records the pair to
`/tmp/mz_rounds_<variant>_<ts>.mp4`. The trace marks the moment the process is
LMK-killed; timing knobs (rounds, dwell, fps) are the variables at the top of the
script. The side-by-side video above was assembled from the two clips afterwards.

**The verdict is the last line the script prints**, and it is the Fig. 15 claim:

```
RESULT: SURVIVED all 3 rounds — pid 12345 unchanged     # expected for mzcache
RESULT: LLM app restarted — was pid 12345, now 23456    # expected for baseline
```

It reproduces the workload faithfully **only** under these conditions:

- **All ten workload apps are installed** (and signed in, where they need an
  account). A missing app is skipped with a one-line log, which lowers the memory
  pressure and changes the result. Region- and vendor-specific ones are the usual
  trap:

  | App | Package | App | Package |
  |---|---|---|---|
  | Instagram | `com.instagram.android` | Roblox | `com.roblox.client` |
  | Facebook | `com.facebook.katana` | PUBG Mobile | `com.pubg.krmobile` (**KR** build) |
  | YouTube | `com.google.android.youtube` | Camera | `com.sec.android.app.camera` (**Samsung**) |
  | TikTok | `com.ss.android.ugc.trill` (**not** `…musically`) | SNOW | `com.campmobile.snow` |
  | Netflix | `com.netflix.mediaclient` | Chrome | `com.android.chrome` |

- **Run on a Galaxy S25+**, with `ADB_SERIAL` exported and `su` available. Every UI
  tap is hard-coded to the 1080×2340 layout; on another resolution the script warns
  and then mis-taps its way to a meaningless `SURVIVED`.
- **Run from the phone host's desktop session** — the script opens a scrcpy mirror
  and a matplotlib window and screen-records a 2880 px-wide region, so it needs a
  live `$DISPLAY` on a monitor at least that wide. Not over SSH, not in Docker.

**Host environment.** Beyond the build toolchain (Section 1):

```bash
sudo apt-get install -y ffmpeg python3-matplotlib python3-pyqt5
```

plus **scrcpy** for the mirror. The script defaults to the scrcpy 3.x flag
`--capture-orientation=@0`; on scrcpy 2.x pass the older equivalent instead:

```bash
SCRCPY_LOCK=--lock-video-orientation=0 ./scripts/mzcache/record_demo_rounds.sh mzcache
```

If scrcpy is not on `PATH`, point the script at it with `SCRCPY_BIN=/path/to/scrcpy`.

---

## 8. Measurement notes & troubleshooting

- **Device-profile gate**: `[MZCACHE][FATAL] device profile ... not found`
  → run `scripts/setup/run_device_profile.sh`. A profile taken under a
  different `MZCACHE_COMPRESSION` build is rejected; delete
  `/data/local/tmp/mzcache/mzcache_device_profile_*.txt` to force re-profiling.
- **Warm-up runs**: first run after push/boot pays cold `clSVMAlloc`
  (hundreds of ms). All runners discard a warm-up where it matters.
- **Thermal drift**: +10–20% on late runs in long batches; runners insert
  cooldown sleeps, but if results look inflated, pause 5 min and re-run.
- **Known issue** — allocation does not parallelize on the 8 Gen 3. The paper's
  Figure 10 is measured on the Galaxy S25+ (8 Elite). On SM8650 (OnePlus 12) the
  allocation stage does not scale with added parallelism and can come out slower.
- **Governor/THP reset on reboot**: re-run `scripts/setup/device_setup.sh`.
- **Everything runs from `/data/local/tmp/mzcache`**: weight-layer files
  (`layers/<model>_layer_<i>.bin`, dumped on first run) and O_DIRECT I/O are
  cwd-relative. Running from elsewhere aborts loudly rather than deadlocking.
- **su domains**: battery sysfs (Fig 13), governor/THP/zram, drop_caches, and
  the OS-paging baseline (mlock > `RLIMIT_MEMLOCK`) all need root.
- **Restore fidelity check (mzCache)**: `MZCACHE_VERIFY_RESTORE=1` verifies stored
  chunks bit-exactly against a never-evicted reference. It is read by the device
  binary, so run it there — prefixing the Section 3.1 runner does nothing:
  ```bash
  adb -s $ADB_SERIAL shell "cd /data/local/tmp/mzcache && MZCACHE_VERIFY_RESTORE=1 \
      ./$MZ_INSTALL/bin/mz_prefill_repeated /data/local/tmp/gguf/Qwen3-0.6B-FP16.gguf 32700 0.5 3"
  ```
- **OS-paging baseline stability**: use the *second* decode after a warm-up
  input for resident TTFT; the "first decode" at 32k is unstable (KV loading
  evicts weights → re-faults, 5–20 s spread).
- **OS-paging remaining-% floors well above 0 — that is expected, not a bug.**
  The x-axis counts the swapped KV's compressed footprint still held in zram and
  excludes the mlocked `token_embd`, and fp16 KV barely compresses, so full
  eviction still reads ~48–75% (Qwen3-0.6B) / ~18–45% (EXAONE-4.0-1.2B). The
  runner stores the raw components per measurement (`w_rss_kb`, `kv_rss_kb`,
  `emb_kb`, the `base_*` baselines, `comp`), and `scripts/plot/plot_ttft.py`
  computes

  ```
  ( (w_rss - emb) + kv_rss + (base_kv - kv_rss)*comp )
  ----------------------------------------------------  x 100
             ( (base_w - emb) + base_kv )
  ```

  so `comp` can be retuned at plot time without re-measuring. Regions are
  classified from `/proc/<pid>/smaps` by `scripts/ttft/region_rss.awk`.
- **Power gauge units**: the samplers read `current_now` as µA (the Linux
  power_supply convention, what the Galaxy S25+ reports → the paper's 8-21 W).
  Some vendor gauges report mA instead — a OnePlus 12 logs `current_now=760`, so
  `Power_W` lands 1000× low. Multiply by 1000; the trace shape is unaffected.
- **Section 7 demo won't start**: check `/tmp/mz_scrcpy.log`, `/tmp/mz_trace.log`
  and `/tmp/mz_ffmpeg.log` — in that order they cover a broken/too-old scrcpy, a
  missing PyQt5 or root, and a `$DISPLAY`/`XAUTHORITY` that does not point at the
  desktop session actually on screen.
