#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <math.h>
#include <string>
#include <unistd.h>
#include <sys/system_properties.h>
#include <sys/stat.h>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <vector>

#include "llama.h"
#include "common.h"
#include "ggml.h"

#ifdef MZCACHE_SVM_KV_CHUNK
// llama's src/ headers are exposed PUBLIC only in mzcache builds (see
// src/CMakeLists.txt); the cpu-swap baseline compiles against llama.h alone.
#include "llama-model.h"
#include "llama-kv-cache-unified.h"
#include "mzcache_types.h"
#include "mzcache_kv_state.h"
#include "mzcache_weight.h"
#include "mzcache_core.h"
#include "mzcache_profile.h"  // device_profile_path()
#endif

// Write C++ code here.
//
// Do not forget to dynamically load the C++ library into your application.
//
// For instance,
//
// In MainActivity.java:
//    static {
//       System.loadLibrary("llama-android");
//    }
//
// Or, in MainActivity.kt:
//    companion object {
//      init {
//         System.loadLibrary("llama-android")
//      }
//    }

#define TAG "llama-android.cpp"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// The mzCache experiment restores a prefilled KV state of this many tokens
// (created offline by examples/mz_prefill on the device) and swaps it in/out.
#define MZ_CONTEXT_TOKENS 32700
#define MZ_DECODE_TOKENS  8
// Extra KV headroom above the loaded state so multi-turn chat has room to grow
// after the restored context (lazy chunks make unused headroom nearly free).
// The experiment flows only ever append MZ_DECODE_TOKENS, so this is inert for
// them; it just lets the chat UI run several turns instead of overflowing.
#define MZ_CHAT_HEADROOM  4096

// Compute threads. The app builds common_params directly instead of going
// through common_params_parse(), so postprocess_cpu_params() — the only caller
// that turns the struct default n_threads = -1 into a real core count — never
// runs, and ggml silently clamps -1 to GGML_DEFAULT_N_THREADS (4). Under full
// eviction the swap-in TTFT is very sensitive to this (measured on SM8650, Qwen
// 32k: 4 threads ~9 s, 6 ~4-20 s, 8 ~40 s, because llama.cpp waits for the
// slowest thread each ubatch and the little cores drag), so pin it explicitly
// rather than leaving it to an accidental default. Fixed at 6 to match the CLI
// baseline (scripts/ttft/run_os_paging.sh -t 6); measured in-app at 32k, median
// of 3: 6 threads 1.23 s resident / 3.27 s evicted, 8 threads 1.31 s / 3.96 s.
#define MZ_N_THREADS      6

static const std::string STATES_DIR = "/data/local/tmp/mzcache/states/";
static int state_size = 0;

// Set once llama_state_load_file succeeds in new_context; every mzCache
// operation positions itself relative to this many restored tokens.
static int n_past_loaded = 0;

// Fraction of the full working set (weights + KV) currently resident in
// memory. Each memory-pressure event lowers the target by the swapout step; a
// successful swapin (mz_decode) restores everything and resets it to 1.0.
// The step is overridable at runtime for experiments via
// `adb shell setprop debug.mzcache.step <float>` (read once in mz_init), so
// swapout aggressiveness can be varied without rebuilding the APK.
// A step of exactly 0 is a special case: the app keeps its whole working set
// resident and ignores every pressure callback. That models an LLM engine with
// no memory-pressure response at all — the same mzCache build, but unable to
// give memory back, so the OS is left to reclaim or kill it.
#define MZ_SWAPOUT_STEP_DEFAULT 0.15f
static float mz_swapout_step = MZ_SWAPOUT_STEP_DEFAULT;
static float cur_ratio = 1.0f;

// True once every chunk is compressed+stored and every layer unloaded; passed
// through to swapin_generate as in examples/mz_prefill.
static bool offload_compressed_kv = false;

// App-writable directory the JNI chdir()s into (mzcache writes KV_*.bin,
// mzcache_arenas_*.swap and reads layers/*.bin relative to the CWD).
static std::string mz_files_dir;

jclass la_int_var;
jmethodID la_int_var_value;
jmethodID la_int_var_inc;

std::string cached_token_chars;

// ---- Multi-turn chat state (mirrors examples/mz_chat) ----
// n_chat_cur is the KV position of the next token to write. It starts at
// n_past_loaded (right after the restored .kv context) and accumulates across
// turns, so the conversation grows in place instead of resetting to 0 like the
// single-turn send() path. g_chat_messages/g_chat_prev_len drive the
// incremental chat template: each turn's prompt is the template diff, which
// carries the previous assistant turn's closing markers. g_chat_response
// accumulates the in-flight assistant reply so it can be appended at EOG.
static int n_chat_cur = 0;
static std::vector<llama_chat_message> g_chat_messages;
static int g_chat_prev_len = 0;
static std::string g_chat_response;

// Detokenized text of the restored .kv context, so the UI can show the loaded
// context as the initial chat history (filled in new_context after the state
// loads; empty when no state was restored).
static std::string g_loaded_context_text;

// Qwen3 emits a <think>...</think> reasoning block. We disable it per turn with
// the /no_think soft switch and additionally hide the block (markers + any
// content) from the streamed text: g_in_think tracks whether we are inside the
// block, g_started_visible trims the leading whitespace of the visible answer.
// The block still lands in the KV and in g_chat_response so the chat-template
// bookkeeping stays aligned with the actual decoded positions.
static bool g_in_think = false;
static bool g_started_visible = false;

// TTFT breakdown instrumentation: time the prefill in chat_completion_init and,
// in the loop, how many tokens (including hidden <think> ones) are generated
// before the first VISIBLE token — so a long "Send -> first token" gap can be
// attributed to prep / prefill / hidden-token generation rather than guessed.
static int64_t g_chat_prefill_done_us = 0;
static int     g_chat_gen_count = 0;
static bool    g_chat_first_visible_logged = false;

// Reset the running conversation to a fresh state that continues after
// `start_pos` tokens (called from new_context once the .kv state is loaded).
static void reset_chat_state(int start_pos) {
    for (auto & m : g_chat_messages) {
        free(const_cast<char *>(m.content)); // role is a literal; only content is strdup'd
    }
    g_chat_messages.clear();
    g_chat_prev_len = 0;
    g_chat_response.clear();
    n_chat_cur = start_pos;
}

static std::string set_state_name(llama_model * model, common_params & params, int context_size_of_states) {
    std::string state_name = STATES_DIR;

#ifdef MZCACHE_SVM_KV_CHUNK
    switch (model->arch) {
        case LLM_ARCH_LLAMA:
            state_name += "llama3";
            break;
        case LLM_ARCH_QWEN3:
            state_name += "qwen3";
            break;
        case LLM_ARCH_EXAONE4:
            state_name += "exaone4";
            break;
        default:
            state_name += "unknown";
            break;
    }

    state_name += "_" + model->type_name();
#else
    // cpu-swap baseline: model internals (arch/type_name) are not visible
    // without the mzcache PUBLIC src/ include path, and the auto flow only
    // ever downloads/loads Qwen3-0.6B — fix the prefix.
    (void) model;
    state_name += "qwen3_0.6B";
#endif

    if (params.cache_type_k == GGML_TYPE_BF16) {
        state_name += "_bf16";
    } else if (params.cache_type_k == GGML_TYPE_Q8_0) {
        state_name += "_q8_0";
    } else if (params.cache_type_k == GGML_TYPE_Q4_0) {
        state_name += "_q4_0";
    }

    if (params.flash_attn) {
        state_name += "_fa";
    }

    if (context_size_of_states == 2049) {
        state_name += "_2049";
        state_size = 2081;
    } else if (context_size_of_states == 4097) {
        state_name += "_4097";
        state_size = 4129;
    } else if (context_size_of_states == 8193) {
        state_name += "_8193";
        state_size = 9225;
    } else if (context_size_of_states == 16385) {
        state_name += "_16385";
        state_size = 16417;
    } else if (context_size_of_states == 32700) {
        state_name += "_32700";
        state_size = 32732;
    } else if (context_size_of_states == 65500) {
        state_name += "_65500";
        state_size = 65532;
    }

    state_name += ".kv";
    return state_name;
}

// Drop any tokens decoded past the restored state so the KV cache is back to
// exactly the prefilled n_past tokens (see examples/mz_prefill_repeated).
static bool cleanup_appended_range(llama_context * ctx, int n_past) {
    if (n_past <= 0) {
        return false;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, n_past, -1);

    const llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
    if (pos_max != (llama_pos) n_past - 1) {
        LOGe("cleanup_appended_range: unexpected seq_pos_max %d (expected %d)", (int) pos_max, n_past - 1);
        return false;
    }

    return true;
}

bool is_valid_utf8(const char * string) {
    if (!string) {
        return true;
    }

    const unsigned char * bytes = (const unsigned char *)string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }

    return true;
}

static void log_callback(ggml_log_level level, const char * fmt, void * data) {
    if (level == GGML_LOG_LEVEL_ERROR)     __android_log_print(ANDROID_LOG_ERROR, TAG, fmt, data);
    else if (level == GGML_LOG_LEVEL_INFO) __android_log_print(ANDROID_LOG_INFO, TAG, fmt, data);
    else if (level == GGML_LOG_LEVEL_WARN) __android_log_print(ANDROID_LOG_WARN, TAG, fmt, data);
    else __android_log_print(ANDROID_LOG_DEFAULT, TAG, fmt, data);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_mz_1init(JNIEnv *env, jobject, jstring jfiles_dir) {
    const char * files_dir = env->GetStringUTFChars(jfiles_dir, 0);
    mz_files_dir = files_dir;
    env->ReleaseStringUTFChars(jfiles_dir, files_dir);

    // mzcache stores KV chunks (KV_<model>_L<i>.bin), arena swap files and
    // weight layer dumps (layers/*.bin) relative to the CWD. An Android app
    // starts with CWD "/", which is not writable, so move to filesDir first.
    if (chdir(mz_files_dir.c_str()) != 0) {
        LOGe("mz_init: chdir(%s) failed: %s", mz_files_dir.c_str(), strerror(errno));
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "mz_init: chdir failed");
        return;
    }

    ::mkdir("layers", 0755);
    ::mkdir("states", 0755);

#ifdef MZCACHE_SVM_KV_CHUNK
    // Apps don't run the offline profiler; ship the pre-measured device profile
    // (Galaxy S25+/SM-S936N, full swap-in emulation, FLEXGEN_8BIT — see
    // examples/mz_device_profile) so mzcache_core can derive its scheduling
    // constants. Written once into the app's CWD (filesDir).
    // Per-compression filename so it matches what mzcache_core reads (the app is
    // built FLEXGEN_8BIT -> mzcache_device_profile_FLEXGEN_8BIT.txt).
    const std::string mz_profile_path = mzcache::device_profile_path();
    if (!std::filesystem::exists(mz_profile_path)) {
        std::ofstream f(mz_profile_path);
        f << "# bundled device profile (Galaxy S25+ / SM-S936N, SM8750, full swap-in emulation)\n"
             "version=1\n"
             "compression=FLEXGEN_8BIT\n"
             "soc=SM8750\n"
             "decomp_gbps=6.60\n"
             "kv_read_gbps=3.06\n"
             "weight_read_gbps=3.81\n";
        LOGi("mz_init: wrote bundled device profile (%s)", mz_profile_path.c_str());
    }
#endif

    // Experiment knob: swapout step per pressure event. 0 is accepted and means
    // "never respond to pressure" (see mz_swapout_step above). Parsing is strict
    // so a typo cannot land on strtof's 0 and silently disable the response —
    // a non-numeric value is rejected and the default step stays in force.
    {
        char prop[PROP_VALUE_MAX] = {0};
        if (__system_property_get("debug.mzcache.step", prop) > 0) {
            char * end = nullptr;
            const float v = strtof(prop, &end);
            if (end == prop) {
                LOGe("mz_init: ignoring debug.mzcache.step='%s' (not a number)", prop);
            } else if (v == 0.0f || (v > 0.009f && v <= 1.0f)) {
                mz_swapout_step = v;
            } else {
                LOGe("mz_init: ignoring debug.mzcache.step='%s' (out of range)", prop);
            }
        }
    }

    LOGi("mz_init: working directory set to %s, swapout step = %.2f%s",
         mz_files_dir.c_str(), (double) mz_swapout_step,
         mz_swapout_step == 0.0f ? " (memory-pressure response DISABLED)" : "");
}

// Experiment knob: override flash attention at runtime without a rebuild —
//   adb shell setprop debug.mzcache.fa 0   # force FA off (loads the non-fa .kv)
//   adb shell setprop debug.mzcache.fa 1   # force FA on  (loads the _fa .kv)
// Unset keeps the per-build default (mzCache: on, cpu-swap: off). Read on every
// load_model / new_context so it takes effect on the next model Load. Because
// set_state_name() appends "_fa" only when flash_attn is set, toggling this also
// selects the matching .kv state file.
static bool resolve_flash_attn(bool default_on) {
    char prop[PROP_VALUE_MAX] = {0};
    if (__system_property_get("debug.mzcache.fa", prop) > 0) {
        if (prop[0] == '0' || prop[0] == 'f' || prop[0] == 'F') {
            LOGi("debug.mzcache.fa='%s' -> flash_attn OFF", prop);
            return false;
        }
        if (prop[0] == '1' || prop[0] == 't' || prop[0] == 'T') {
            LOGi("debug.mzcache.fa='%s' -> flash_attn ON", prop);
            return true;
        }
        LOGe("ignoring debug.mzcache.fa='%s' (expected 0/1/true/false)", prop);
    }
    return default_on;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_load_1model(JNIEnv *env, jobject, jstring filename) {
    common_params params;
    params.sampling.seed  = 1234;
    params.n_ctx         = MZ_CONTEXT_TOKENS + 32;
    params.n_predict     = 32;
    params.n_batch       = MZ_CONTEXT_TOKENS + 32;
    params.n_ubatch      = 16;
    params.warmup        = false;
    params.cpuparams.n_threads = MZ_N_THREADS;
#ifdef MZCACHE_SVM_KV_CHUNK
    params.flash_attn    = true;
    params.use_mmap      = false;  // weight layers are reloaded from file after unload
    params.n_gpu_layers  = 100;
#else
    // cpu-swap baseline: weights stay file-backed (mmap) so the kernel can
    // drop and refault them under memory pressure; the KV cache is plain
    // anonymous memory that swaps out to zram. No GPU backend is built in.
    // Flash attention off -> standard CPU attention; loads the non-fa state.
    params.flash_attn    = false;
    params.use_mmap      = true;
    params.n_gpu_layers  = 0;
#endif
    params.flash_attn = resolve_flash_attn(params.flash_attn);

    auto path_to_model = env->GetStringUTFChars(filename, 0);
    LOGi("Loading model from %s", path_to_model);

    auto mparams = common_model_params_to_llama(params);
    auto model = llama_model_load_from_file(path_to_model, mparams);
    env->ReleaseStringUTFChars(filename, path_to_model);

    if (!model) {
        LOGe("load_model() failed");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "load_model() failed");
        return 0;
    }

    return reinterpret_cast<jlong>(model);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1model(JNIEnv *, jobject, jlong model) {
    llama_model_free(reinterpret_cast<llama_model *>(model));
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1context(JNIEnv *env, jobject, jlong jmodel) {
    auto model = reinterpret_cast<llama_model *>(jmodel);

    if (!model) {
        LOGe("new_context(): model cannot be null");
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Model cannot be null");
        return 0;
    }

    common_params params;
    params.sampling.seed  = 1234;
    params.n_ctx         = MZ_CONTEXT_TOKENS + MZ_CHAT_HEADROOM;  // prefilled state + chat headroom
    params.n_predict     = 32;
    params.n_batch       = MZ_CONTEXT_TOKENS + MZ_CHAT_HEADROOM;
    params.n_ubatch      = 16;
    params.warmup        = false;
    params.cpuparams.n_threads = MZ_N_THREADS;   // decode threads (see MZ_N_THREADS)
#ifdef MZCACHE_SVM_KV_CHUNK
    params.flash_attn    = true;
    params.use_mmap      = false;
    params.n_gpu_layers  = 100;
#else
    params.flash_attn    = false;  // cpu-swap baseline: standard attention, non-fa state
    params.use_mmap      = true;   // cpu-swap baseline (see load_model)
    params.n_gpu_layers  = 0;
#endif
    params.flash_attn = resolve_flash_attn(params.flash_attn);

    auto cparams = common_context_params_to_llama(params);
    llama_context * context = llama_init_from_model(model, cparams);

    if (!context) {
        LOGe("llama_init_from_model() returned null)");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "llama_init_from_model() returned null)");
        return 0;
    }
    LOGi("new_context: n_threads=%d n_threads_batch=%d (MZ_N_THREADS=%d)",
         cparams.n_threads, cparams.n_threads_batch, MZ_N_THREADS);

    // Restore the prefilled KV state created offline by examples/mz_prefill.
    std::string state_name = set_state_name(model, params, MZ_CONTEXT_TOKENS);
    if (!std::filesystem::exists(state_name) && !mz_files_dir.empty()) {
        // Fall back to the app-private copy (CWD is mz_files_dir after mz_init).
        std::string fallback = "states/" + std::filesystem::path(state_name).filename().string();
        LOGi("state file not found at %s, trying %s", state_name.c_str(), fallback.c_str());
        state_name = fallback;
    }

    size_t n_token_count = 0;
    if (std::filesystem::exists(state_name)) {
        std::vector<llama_token> tmp(params.n_ctx);
        if (!llama_state_load_file(context, state_name.c_str(), tmp.data(), tmp.size(), &n_token_count)) {
            LOGe("llama_state_load_file() failed for %s", state_name.c_str());
            env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                          "llama_state_load_file() failed");
            return 0;
        }
        n_past_loaded = (int) n_token_count;
        LOGi("Loaded %zu tokens from state file: %s", n_token_count, state_name.c_str());

        // Show only a short preview of the restored context — the first and
        // last PREVIEW_EDGE tokens with an ellipsis between — so the UI never
        // has to render the full (huge) context and stays lightweight.
        g_loaded_context_text.clear();
        const size_t PREVIEW_EDGE = 30;
        if (n_token_count <= 2 * PREVIEW_EDGE) {
            for (size_t i = 0; i < n_token_count; ++i) {
                g_loaded_context_text += common_token_to_piece(context, tmp[i]);
            }
        } else {
            for (size_t i = 0; i < PREVIEW_EDGE; ++i) {
                g_loaded_context_text += common_token_to_piece(context, tmp[i]);
            }
            g_loaded_context_text += "\n\n ... (skipped " +
                std::to_string(n_token_count - 2 * PREVIEW_EDGE) + " tokens) ... \n\n";
            for (size_t i = n_token_count - PREVIEW_EDGE; i < n_token_count; ++i) {
                g_loaded_context_text += common_token_to_piece(context, tmp[i]);
            }
        }

        cleanup_appended_range(context, n_past_loaded);
    } else {
        g_loaded_context_text.clear();
        LOGe("No KV state file found at %s - mzCache swapout/swapin will be unavailable", state_name.c_str());
    }

    // Multi-turn chat continues right after the restored context (n_past_loaded
    // is 0 when no state loaded, so a plain chat still starts at position 0).
    reset_chat_state(n_past_loaded);

    return reinterpret_cast<jlong>(context);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1context(JNIEnv *, jobject, jlong context) {
    llama_free(reinterpret_cast<llama_context *>(context));
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1mzcache(JNIEnv *env, jobject, jlong context_pointer, jlong model_pointer) {
#ifdef MZCACHE_SVM_KV_CHUNK
    auto * ctx = reinterpret_cast<llama_context *>(context_pointer);
    auto * model = reinterpret_cast<llama_model *>(model_pointer);

    if (!ctx || !model) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "new_mzcache: context/model cannot be null");
        return 0;
    }

    if (n_past_loaded <= 0) {
        LOGe("new_mzcache: no prefilled KV state was loaded (place the .kv file first)");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "new_mzcache: no prefilled KV state was loaded");
        return 0;
    }

    auto * kv_cache = static_cast<llama_kv_cache_unified *>(llama_get_memory(ctx));
    const int n_layers = kv_cache->layers_k_chunks.size();
    const int hdim = model->hparams.n_embd_k_gqa(n_layers - 1);
    const int n_chunks_per_tensor = (n_past_loaded + TOKENS_PER_CHUNK - 1) / TOKENS_PER_CHUNK;

    // Weight unload reloads layers from layers/<name>_layer_<i>.bin; generate
    // them on first run while all layers are still resident (mz_weight_test
    // does the same offline). Missing files would abort a thread-pool worker.
    std::string sanitized_name = model->name;
    std::replace(sanitized_name.begin(), sanitized_name.end(), ' ', '_');
    const std::string layer_prefix = "layers/" + sanitized_name;

    for (int i = 0; i < n_layers; ++i) {
        const std::string layer_file = layer_prefix + "_layer_" + std::to_string(i) + ".bin";
        if (std::filesystem::exists(layer_file)) {
            continue;
        }
        LOGi("new_mzcache: generating %s", layer_file.c_str());
        if (!model->unload_layers_to_file(layer_prefix, i)) {
            LOGe("new_mzcache: unload_layers_to_file failed for layer %d", i);
            env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                          "new_mzcache: failed to generate weight layer file");
            return 0;
        }
    }

    LOGi("Initializing mzcache_core with %d layers, hidden dimension %d, %d chunks per tensor",
         n_layers, hdim, n_chunks_per_tensor);

    auto * core = new mzcache_core(n_layers, hdim, n_chunks_per_tensor, model);

    if (core->weight_layer_bytes == 0) {
        LOGe("new_mzcache: unsupported model name '%s' (weight_layer_bytes not configured in mzcache_core)",
             model->name.c_str());
        delete core;
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "new_mzcache: unsupported model for mzcache");
        return 0;
    }

    cur_ratio = 1.0f;
    offload_compressed_kv = false;

    LOGi("mzcache_core initialized successfully");
    return reinterpret_cast<jlong>(core);
#else
    (void) context_pointer; (void) model_pointer;
    // cpu-swap baseline: no mzcache core. Return a dummy non-zero handle so
    // the Kotlin load sequence proceeds unchanged (free_mzcache is a no-op in
    // this build, and the baseline auto flow never calls mz_decode or
    // handleMemoryPressure).
    LOGi("new_mzcache: built without MZCACHE_SVM_KV_CHUNK (cpu-swap baseline), returning dummy handle");
    return 1;
#endif
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1mzcache(JNIEnv *, jobject, jlong mzcache_core_pointer) {
#ifdef MZCACHE_SVM_KV_CHUNK
    delete reinterpret_cast<mzcache_core *>(mzcache_core_pointer);
#else
    (void) mzcache_core_pointer;
#endif
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_handleMemoryPressureNative(JNIEnv *env, jobject, jlong context_pointer, jlong model_pointer, jlong mzcache_core_pointer) {
#ifdef MZCACHE_SVM_KV_CHUNK
    LOGi("Memory pressure signal detected, native function called.");

    // step 0: no pressure response at all. Return before touching the core so
    // the working set stays exactly where it is — nothing compressed, stored or
    // unloaded, and cur_ratio left at whatever it was.
    if (mz_swapout_step == 0.0f) {
        LOGi("handleMemoryPressureNative: swapout step = 0, pressure response disabled — ignoring");
        return;
    }

    auto * core = reinterpret_cast<mzcache_core *>(mzcache_core_pointer);
    auto * ctx = reinterpret_cast<llama_context *>(context_pointer);

    if (!core || !ctx || n_past_loaded <= 0) {
        LOGe("handleMemoryPressureNative: mzcache is not ready, ignoring");
        return;
    }

    mzcache_kv_state * kv_state_ptr = core->get_kv_state();
    mzcache_weight * weight_ptr = core->get_weight();

    auto * kv_cache = static_cast<llama_kv_cache_unified *>(llama_get_memory(ctx));
    const int n_layers = kv_cache->layers_k_chunks.size();

    const float target_ratio = std::max(0.0f, cur_ratio - mz_swapout_step);
    LOGi("swapout: cur_ratio %.3f -> target %.3f", cur_ratio, target_ratio);

    auto [achieved_ratio, n_comp_chunks, n_store_chunks, n_unload_layers] = core->swapout(target_ratio);

    // swapout's returned ratio is cumulative (measured freed bytes tracked in
    // kv_state across calls), so it is the resident ratio to base the next
    // pressure step on.
    cur_ratio = achieved_ratio;

    // Also offload weight layers proportionally to the ladder. The core swapout
    // ladder only unloads weights at very deep targets (its Phase 2 runs after
    // ~all KV is compressed), so at the demo's moderate steps the ~0.84 GB of
    // weights never leave GPU memory. Unload enough layers that the offloaded
    // fraction tracks the resident ratio; swapin_generate restores them via
    // mzcache_reload_w_layers() (per-layer sync makes the decode wait for each).
    // Layer files were generated in new_mzcache; mzcache_unload_layers returns
    // false (and we stop) if one is missing, so this degrades gracefully. This is
    // an app-demo add-on and does not touch core->swapout (the experiment path).
    const int want_unload = std::min(n_layers, (int) (n_layers * (1.0f - cur_ratio) + 0.5f));
    int extra_unload = 0;
    while (weight_ptr->n_cur_unload_layers < want_unload && weight_ptr->mzcache_unload_layers(1)) {
        ++extra_unload;
    }
    if (extra_unload > 0) {
        LOGi("swapout: +weight unload %d (total %d/%d, want %d for ratio %.3f)",
             extra_unload, weight_ptr->n_cur_unload_layers, n_layers, want_unload, (double) cur_ratio);
    }

    offload_compressed_kv =
        kv_state_ptr->n_cur_comp_chunks == kv_state_ptr->max_comp_chunks &&
        kv_state_ptr->n_cur_store_chunks == kv_state_ptr->max_store_chunks &&
        weight_ptr->n_cur_unload_layers == n_layers;

    LOGi("swapout done: achieved %.3f, +comp %d (total %d/%d), +store %d (total %d/%d), +unload %d (total %d/%d), offload_compressed_kv=%d",
         achieved_ratio,
         n_comp_chunks, kv_state_ptr->n_cur_comp_chunks, kv_state_ptr->max_comp_chunks,
         n_store_chunks, kv_state_ptr->n_cur_store_chunks, kv_state_ptr->max_store_chunks,
         n_unload_layers, weight_ptr->n_cur_unload_layers, n_layers,
         (int) offload_compressed_kv);
#else
    (void) context_pointer; (void) model_pointer; (void) mzcache_core_pointer;
    LOGe("handleMemoryPressureNative: built without MZCACHE_SVM_KV_CHUNK");
#endif
}

// Build the 8-token "Hello ..." follow-up batch positioned right after the
// restored state, exactly like examples/mz_prefill (lines 340-388). Shared by
// mz_decode (swapin path) and baseline_decode (plain decode path). Returns
// false (with nothing to free) if the batch cannot be built.
static bool build_follow_batch(llama_context * ctx, llama_model * model, llama_batch & batch_out, const char * who) {
#ifdef MZCACHE_SVM_KV_CHUNK
    if ((n_past_loaded % TOKENS_PER_CHUNK) + MZ_DECODE_TOKENS > TOKENS_PER_CHUNK) {
        // No longer fatal: decodes are split at chunk boundaries internally
        // (mzcache::decode_chunk_aligned), and the crossed-into chunk is grown
        // on demand (llama_kv_cache_unified::mz_grow_chunks).
        LOGi("%s: follow-up prompt crosses a chunk boundary (n_past %% %d = %d) — will be split",
             who, TOKENS_PER_CHUNK, n_past_loaded % TOKENS_PER_CHUNK);
    }
#endif

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string follow_prompt = "Hello";
    const int NCOLS = MZ_DECODE_TOKENS;

    int n_tok = -llama_tokenize(
        vocab,
        follow_prompt.c_str(),
        follow_prompt.size(),
        nullptr,
        0,
        true,
        true);

    LOGi("%s: token count of follow prompt: %d", who, n_tok);
    for (int i = n_tok; i < NCOLS; i++) {
        follow_prompt += " Hello";
    }

    std::vector<llama_token> follow_tokens(NCOLS);
    llama_tokenize(
        vocab,
        follow_prompt.c_str(),
        follow_prompt.size(),
        follow_tokens.data(),
        follow_tokens.size(),
        true,
        true);

    llama_batch batch2 = llama_batch_init(follow_tokens.size(), 0, 1);
    int32_t pos = n_past_loaded;
    for (size_t i = 0; i < follow_tokens.size(); ++i, ++pos) {
        common_batch_add(batch2, follow_tokens[i], pos, {0}, false);
    }

    const int n_ctx = llama_n_ctx(ctx);
    const int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (n_ctx_used + batch2.n_tokens > n_ctx) {
        LOGe("%s: context size exceeded (%d + %d > %d)", who, n_ctx_used, batch2.n_tokens, n_ctx);
        llama_batch_free(batch2);
        return false;
    }

    batch2.logits[batch2.n_tokens - 1] = true;
    batch_out = batch2;
    return true;
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_mz_1decode(JNIEnv *env, jobject, jlong context_pointer, jlong model_pointer, jlong mzcache_core_pointer) {
#ifdef MZCACHE_SVM_KV_CHUNK
    LOGi("mz_decode: swapin + prefill requested.");

    auto * core = reinterpret_cast<mzcache_core *>(mzcache_core_pointer);
    auto * ctx = reinterpret_cast<llama_context *>(context_pointer);
    auto * model = reinterpret_cast<llama_model *>(model_pointer);

    if (!core || !ctx || !model || n_past_loaded <= 0) {
        LOGe("mz_decode: mzcache is not ready, ignoring");
        return;
    }

    // Drop tokens appended by a previous decode so the follow-up prompt
    // always starts right after the restored state and, in particular, never
    // crosses a chunk boundary (see examples/mz_prefill_repeated).
    cleanup_appended_range(ctx, n_past_loaded);

    llama_batch batch2;
    if (!build_follow_batch(ctx, model, batch2, "mz_decode")) {
        return;
    }

    // Swapin (arena reload + decompress + load + weight reload) overlapped
    // with the prefill decode; equivalent to mz_prefill.cpp:507.
    long ttft = core->swapin_generate(ctx, batch2, offload_compressed_kv);

    LOGi("mz_decode: swapin_generate completed, TTFT = %ld us (%.2f ms)", ttft, (double) ttft / 1000.0);

    llama_batch_free(batch2);

    // Everything is resident again; rewind to the pristine prefilled state so
    // the next swapout/swapin cycle starts from the same point.
    cleanup_appended_range(ctx, n_past_loaded);
    cur_ratio = 1.0f;
    offload_compressed_kv = false;
#else
    (void) context_pointer; (void) model_pointer; (void) mzcache_core_pointer;
    LOGe("mz_decode: built without MZCACHE_SVM_KV_CHUNK");
#endif
}

// Baseline counterpart of mz_decode: the same 8-token follow-up prefill but a
// plain llama_decode with no mzcache swapin. Used by the -PmzBaseline app
// build, where nothing is ever swapped out so all chunks are resident.
extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_baseline_1decode(JNIEnv *, jobject, jlong context_pointer, jlong model_pointer) {
    LOGi("baseline_decode: plain 8-token prefill requested.");

    auto * ctx = reinterpret_cast<llama_context *>(context_pointer);
    auto * model = reinterpret_cast<llama_model *>(model_pointer);

    if (!ctx || !model || n_past_loaded <= 0) {
        LOGe("baseline_decode: state not loaded, ignoring");
        return;
    }

    cleanup_appended_range(ctx, n_past_loaded);

    llama_batch batch2;
    if (!build_follow_batch(ctx, model, batch2, "baseline_decode")) {
        return;
    }

    const int64_t t0 = ggml_time_us();
#ifdef MZCACHE_SVM_KV_CHUNK
    // SVM-chunked KV imposes the chunk-boundary batching invariant.
    const int ret = mzcache::decode_chunk_aligned(ctx, batch2);
#else
    // cpu-swap baseline: plain contiguous KV, a straight decode suffices.
    const int ret = llama_decode(ctx, batch2);
#endif
    const int64_t ttft = ggml_time_us() - t0;

    if (ret != 0) {
        LOGe("baseline_decode: decode failed, ret=%d", ret);
    }
    LOGi("baseline_decode: prefill completed, TTFT = %lld us (%.2f ms)",
         (long long) ttft, (double) ttft / 1000.0);

    llama_batch_free(batch2);

    cleanup_appended_range(ctx, n_past_loaded);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_backend_1free(JNIEnv *, jobject) {
    llama_backend_free();
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_log_1to_1android(JNIEnv *, jobject) {
    llama_log_set(log_callback, NULL);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_bench_1model(
        JNIEnv *env,
        jobject,
        jlong context_pointer,
        jlong model_pointer,
        jlong batch_pointer,
        jint pp,
        jint tg,
        jint pl,
        jint nr
        ) {
    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto model = reinterpret_cast<llama_model *>(model_pointer);
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);

    const int n_ctx = llama_n_ctx(context);

    LOGi("n_ctx = %d", n_ctx);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp)");

        common_batch_clear(*batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(*batch, 0, i, { 0 }, false);
        }

        batch->logits[batch->n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(context, *batch) != 0) {
            LOGi("llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg)");

        llama_memory_clear(llama_get_memory(context), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {

            common_batch_clear(*batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(*batch, 0, i, { j }, true);
            }

            LOGi("llama_decode() text generation: %d", i);
            if (llama_decode(context, *batch) != 0) {
                LOGi("llama_decode() failed during text generation");
            }
        }

        const auto t_tg_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(context), false);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));

    const auto model_size     = double(llama_model_size(model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(model)) / 1e9;

    const auto backend    = "(Android)"; // TODO: What should this be?

    std::stringstream result;
    result << std::setprecision(2);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | " << backend << " | pp " << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | " << backend << " | tg " << tg << " | " << tg_avg << " ± " << tg_std << " |\n";

    return env->NewStringUTF(result.str().c_str());
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1batch(JNIEnv *, jobject, jint n_tokens, jint embd, jint n_seq_max) {

    // Source: Copy of llama.cpp:llama_batch_init but heap-allocated.

    llama_batch *batch = new llama_batch {
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
    };

    if (embd) {
        batch->embd = (float *) malloc(sizeof(float) * n_tokens * embd);
    } else {
        batch->token = (llama_token *) malloc(sizeof(llama_token) * n_tokens);
    }

    batch->pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens);
    batch->n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens);
    batch->seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        batch->seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch->logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens);

    return reinterpret_cast<jlong>(batch);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1batch(JNIEnv *, jobject, jlong batch_pointer) {
    //llama_batch_free(*reinterpret_cast<llama_batch *>(batch_pointer));
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);
    delete batch;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1sampler(JNIEnv *, jobject) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    return reinterpret_cast<jlong>(smpl);
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_free_1sampler(JNIEnv *, jobject, jlong sampler_pointer) {
    llama_sampler_free(reinterpret_cast<llama_sampler *>(sampler_pointer));
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_backend_1init(JNIEnv *, jobject) {
    llama_backend_init();
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_system_1info(JNIEnv *env, jobject) {
    return env->NewStringUTF(llama_print_system_info());
}

extern "C"
JNIEXPORT jint JNICALL
Java_android_llama_cpp_LLamaAndroid_completion_1init(
        JNIEnv *env,
        jobject,
        jlong context_pointer,
        jlong batch_pointer,
        jstring jtext,
        jboolean format_chat,
        jint n_len
    ) {

    cached_token_chars.clear();

    const auto text = env->GetStringUTFChars(jtext, 0);
    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto batch = reinterpret_cast<llama_batch *>(batch_pointer);

    bool parse_special = (format_chat == JNI_TRUE);
    const auto tokens_list = common_tokenize(context, text, true, parse_special);

    auto n_ctx = llama_n_ctx(context);
    auto n_kv_req = tokens_list.size() + n_len;

    LOGi("n_len = %d, n_ctx = %d, n_kv_req = %d", n_len, n_ctx, n_kv_req);

    if (n_kv_req > n_ctx) {
        LOGe("error: n_kv_req > n_ctx, the required KV cache size is not big enough");
    }

    for (auto id : tokens_list) {
        LOGi("token: `%s`-> %d ", common_token_to_piece(context, id).c_str(), id);
    }

    common_batch_clear(*batch);

    // evaluate the initial prompt
    for (auto i = 0; i < tokens_list.size(); i++) {
        common_batch_add(*batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch->logits[batch->n_tokens - 1] = true;

    if (llama_decode(context, *batch) != 0) {
        LOGe("llama_decode() failed");
    }

    env->ReleaseStringUTFChars(jtext, text);

    return batch->n_tokens;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_completion_1loop(
        JNIEnv * env,
        jobject,
        jlong context_pointer,
        jlong batch_pointer,
        jlong sampler_pointer,
        jint n_len,
        jobject intvar_ncur
) {
    const auto context = reinterpret_cast<llama_context *>(context_pointer);
    const auto batch   = reinterpret_cast<llama_batch   *>(batch_pointer);
    const auto sampler = reinterpret_cast<llama_sampler *>(sampler_pointer);
    const auto model = llama_get_model(context);
    const auto vocab = llama_model_get_vocab(model);

    if (!la_int_var) la_int_var = env->GetObjectClass(intvar_ncur);
    if (!la_int_var_value) la_int_var_value = env->GetMethodID(la_int_var, "getValue", "()I");
    if (!la_int_var_inc) la_int_var_inc = env->GetMethodID(la_int_var, "inc", "()V");

    // sample the most likely token
    const auto new_token_id = llama_sampler_sample(sampler, context, -1);

    const auto n_cur = env->CallIntMethod(intvar_ncur, la_int_var_value);
    if (llama_vocab_is_eog(vocab, new_token_id) || n_cur == n_len) {
        return nullptr;
    }

    auto new_token_chars = common_token_to_piece(context, new_token_id);
    cached_token_chars += new_token_chars;

    jstring new_token = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        new_token = env->NewStringUTF(cached_token_chars.c_str());
        LOGi("cached: %s, new_token_chars: `%s`, id: %d", cached_token_chars.c_str(), new_token_chars.c_str(), new_token_id);
        cached_token_chars.clear();
    } else {
        new_token = env->NewStringUTF("");
    }

    common_batch_clear(*batch);
    common_batch_add(*batch, new_token_id, n_cur, { 0 }, true);

    env->CallVoidMethod(intvar_ncur, la_int_var_inc);

    if (llama_decode(context, *batch) != 0) {
        LOGe("llama_decode() returned null");
    }

    return new_token;
}

// ==================== Multi-turn chat (mzcache-aware) ====================
//
// chat_completion_init / chat_completion_loop are the multi-turn counterpart of
// completion_init / completion_loop. Differences:
//   * the user turn is prefilled at the running position n_chat_cur (which
//     started at n_past_loaded), not at position 0, so the loaded .kv context
//     and every prior turn stay in the KV cache — the conversation accumulates;
//   * the prompt is formatted through the model's chat template (incremental
//     diff), so an instruct model actually answers;
//   * when the KV was swapped out (cur_ratio < 1.0), the first prefill after a
//     swapout is routed through core->swapin_generate, which restores the
//     swapped-out KV/weights concurrently with the prefill (as in mz_decode);
//   * decodes go through mzcache::decode_chunk_aligned so turns of any length
//     split at 256-token chunk boundaries and chunks grow on demand;
//   * the KV is NOT cleared afterwards.
extern "C"
JNIEXPORT jint JNICALL
Java_android_llama_cpp_LLamaAndroid_chat_1completion_1init(
        JNIEnv *env,
        jobject,
        jlong context_pointer,
        jlong model_pointer,
        jlong mzcache_core_pointer,
        jlong batch_pointer,
        jstring jtext
    ) {
    cached_token_chars.clear();
    g_chat_response.clear();
    g_in_think = false;
    g_started_visible = false;
    g_chat_gen_count = 0;
    g_chat_first_visible_logged = false;
    const int64_t t_init0 = ggml_time_us();

    auto * ctx   = reinterpret_cast<llama_context *>(context_pointer);
    auto * model = reinterpret_cast<llama_model *>(model_pointer);
    auto * batch = reinterpret_cast<llama_batch *>(batch_pointer);
    const auto text = env->GetStringUTFChars(jtext, 0);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    LOGi("chat_completion_init: n_chat_cur=%d cur_ratio=%.3f", n_chat_cur, (double) cur_ratio);

    // Minimal prompt: NO chat template — just the user text followed by Qwen3's
    // CLOSED empty reasoning block. Dropping the <|im_start|>user/assistant
    // scaffolding cuts the prefill from ~20 tokens to ~6 ("Summarize context" ->
    // "Summarize context<think>\n\n</think>\n\n"), which roughly halves the
    // prefill time at a 32k context. The closed <think></think> both stops the
    // model from generating a reasoning block AND cues it to answer directly, so
    // even a short instruction still yields a plausible response. (Multi-turn
    // chat-template bookkeeping is dropped with it — the demo is single-shot.)
    std::string prompt = std::string(text) + "<think>\n\n</think>\n\n";
    env->ReleaseStringUTFChars(jtext, text);

    // Only the very first turn of an empty context adds BOS; otherwise we are
    // continuing an existing sequence (loaded .kv and/or previous turns).
    const bool add_bos = (n_chat_cur == 0);
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, add_bos, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (n_prompt <= 0 ||
        llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(),
                       prompt_tokens.size(), add_bos, true) < 0) {
        LOGe("chat_completion_init: tokenize failed");
        return n_chat_cur;
    }

    // The persistent batch is allocated with 512 slots (see ensureLoaded).
    const int batch_cap = 512;
    if (n_prompt > batch_cap) {
        LOGe("chat_completion_init: prompt %d tokens exceeds batch capacity %d — clamping",
             n_prompt, batch_cap);
        n_prompt = batch_cap;
    }
    if (n_chat_cur + n_prompt + 1 > (int) llama_n_ctx(ctx)) {
        LOGe("chat_completion_init: context size exceeded");
        return n_chat_cur;
    }

    common_batch_clear(*batch);
    for (int i = 0; i < n_prompt; ++i) {
        common_batch_add(*batch, prompt_tokens[i], n_chat_cur + i, {0}, false);
    }
    batch->logits[batch->n_tokens - 1] = true;

    const int64_t t_prep = ggml_time_us();   // template + tokenize + batch build
    const char * prefill_path = "decode";
#ifdef MZCACHE_SVM_KV_CHUNK
    auto * core = reinterpret_cast<mzcache_core *>(mzcache_core_pointer);
    if (core && cur_ratio < 1.0f) {
        // Restore the swapped-out KV/weights concurrently with this prefill.
        prefill_path = "swapin";
        const long ttft = core->swapin_generate(ctx, *batch, offload_compressed_kv);
        LOGi("chat_completion_init: swapin_generate TTFT %ld us (%.2f ms, %d prompt tokens)",
             ttft, (double) ttft / 1000.0, n_prompt);
        cur_ratio = 1.0f;
        offload_compressed_kv = false;
    } else if (mzcache::decode_chunk_aligned(ctx, *batch) != 0) {
        LOGe("chat_completion_init: prompt decode failed");
    }
#else
    (void) mzcache_core_pointer;
    if (llama_decode(ctx, *batch) != 0) {
        LOGe("chat_completion_init: llama_decode failed");
    }
#endif

    const int64_t t_prefill = ggml_time_us();
    g_chat_prefill_done_us = t_prefill;
    LOGi("chat TTFT breakdown: prep=%.1fms prefill(%s)=%.1fms n_prompt=%d n_chat_cur=%d",
         (t_prep - t_init0) / 1000.0, prefill_path, (t_prefill - t_prep) / 1000.0,
         n_prompt, n_chat_cur);

    n_chat_cur += n_prompt;
    return n_chat_cur;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_chat_1completion_1loop(
        JNIEnv *env,
        jobject,
        jlong context_pointer,
        jlong model_pointer,
        jlong mzcache_core_pointer,
        jlong batch_pointer,
        jlong sampler_pointer,
        jint n_len,
        jobject intvar_ncur
    ) {
    (void) mzcache_core_pointer;
    auto * ctx     = reinterpret_cast<llama_context *>(context_pointer);
    auto * model   = reinterpret_cast<llama_model *>(model_pointer);
    auto * batch   = reinterpret_cast<llama_batch *>(batch_pointer);
    auto * sampler = reinterpret_cast<llama_sampler *>(sampler_pointer);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (!la_int_var) la_int_var = env->GetObjectClass(intvar_ncur);
    if (!la_int_var_value) la_int_var_value = env->GetMethodID(la_int_var, "getValue", "()I");
    if (!la_int_var_inc) la_int_var_inc = env->GetMethodID(la_int_var, "inc", "()V");

    const auto new_token_id = llama_sampler_sample(sampler, ctx, -1);
    const auto n_gen = env->CallIntMethod(intvar_ncur, la_int_var_value);

    if (llama_vocab_is_eog(vocab, new_token_id) || n_gen >= n_len ||
        n_chat_cur + 1 > (int) llama_n_ctx(ctx)) {
        // End of this turn. The minimal prompt uses no chat template, so there is
        // no next-turn diff to accumulate — just stop.
        return nullptr;
    }

    g_chat_gen_count++;   // this is a real (non-EOG) generated token

    const auto piece = common_token_to_piece(ctx, new_token_id);
    g_chat_response += piece; // raw, keeps the chat-template bookkeeping aligned

    // Hide the <think>...</think> block from the streamed text (markers and any
    // content); the block stays in g_chat_response / the KV above.
    std::string vis;
    if (piece == "<think>") {
        g_in_think = true;
    } else if (piece == "</think>") {
        g_in_think = false;
    } else if (!g_in_think) {
        vis = piece;
        if (!g_started_visible) {
            const size_t p = vis.find_first_not_of(" \n\r\t");
            if (p == std::string::npos) {
                vis.clear();            // still only whitespace after the block
            } else {
                vis = vis.substr(p);    // trim the leading whitespace once
                g_started_visible = true;
            }
        }
    }

    if (!g_chat_first_visible_logged && !vis.empty()) {
        g_chat_first_visible_logged = true;
        LOGi("chat first-visible token: after %d generated tokens (%d hidden), %.1fms since prefill done",
             g_chat_gen_count, g_chat_gen_count - 1,
             (ggml_time_us() - g_chat_prefill_done_us) / 1000.0);
    }

    jstring out = nullptr;
    if (!vis.empty()) {
        cached_token_chars += vis;
        if (is_valid_utf8(cached_token_chars.c_str())) {
            out = env->NewStringUTF(cached_token_chars.c_str());
            cached_token_chars.clear();
        } else {
            out = env->NewStringUTF("");
        }
    } else {
        out = env->NewStringUTF("");
    }

    common_batch_clear(*batch);
    common_batch_add(*batch, new_token_id, n_chat_cur, {0}, true);
    env->CallVoidMethod(intvar_ncur, la_int_var_inc);

#ifdef MZCACHE_SVM_KV_CHUNK
    if (mzcache::decode_chunk_aligned(ctx, *batch) != 0) {
        LOGe("chat_completion_loop: decode failed at pos %d", n_chat_cur);
    }
#else
    if (llama_decode(ctx, *batch) != 0) {
        LOGe("chat_completion_loop: llama_decode failed");
    }
#endif
    n_chat_cur++;

    return out;
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_kv_1cache_1clear(JNIEnv *, jobject, jlong context) {
    llama_memory_clear(llama_get_memory(reinterpret_cast<llama_context *>(context)), true);
}

// The detokenized text of the restored .kv context (empty when no state was
// loaded), so the UI can seed the chat history with the loaded context.
extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_loaded_1context_1text(JNIEnv *env, jobject) {
    return env->NewStringUTF(g_loaded_context_text.c_str());
}
