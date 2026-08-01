// mz_dump_layers — one-time on-device setup tool.
//
// mzCache's weight swap-out unloads transformer layers from GPU memory and
// later reloads them from ./layers/<model>_layer_<i>.bin (cwd-relative, see
// mzcache_weight.cpp). Since commit 9399d127 the engine refuses to unload a
// layer whose reload file is missing (graceful degradation to compression-only
// swap-out), so these files MUST exist before any weight-unloading experiment.
// The Android app generates them on first launch (llama-android.cpp); this
// tool is the CLI equivalent: load the model once, dump every layer, exit.
//
// Usage (from /data/local/tmp/mzcache):
//   ./mz_dump_layers <model.gguf>
//
// Idempotent: layers that already have a file are skipped.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-model.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include <sys/stat.h>

static char * copy_str(const char * s) {
    char * d = new char[strlen(s) + 1];
    strcpy(d, s);
    return d;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>\n"
                  << "Dumps every transformer layer to ./layers/<model>_layer_<i>.bin\n"
                  << "(run from /data/local/tmp/mzcache).\n";
        return 1;
    }
    const std::string gguf = argv[1];

    common_params params;
    params.sampling.seed = 1234;
    params.use_mmap      = false;
    params.n_ctx         = 2080;   // minimal context: only the model load matters
    params.n_batch       = 2080;
    params.n_ubatch      = 16;
    params.warmup        = false;

    int    fixed_argc = 6;
    char ** fixed_argv = new char *[fixed_argc];
    fixed_argv[0] = copy_str(argv[0]);
    fixed_argv[1] = copy_str("-m");
    fixed_argv[2] = copy_str(gguf.c_str());
    fixed_argv[3] = copy_str("-ngl");
    fixed_argv[4] = copy_str("100");
    fixed_argv[5] = copy_str("-fa");
    if (!common_params_parse(fixed_argc, fixed_argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();

    auto init_res = common_init_from_params(params);
    llama_model * model = init_res.model.get();
    if (!model || !init_res.context.get()) {
        std::cerr << "ERROR: model/context initialization failed\n";
        return 1;
    }

    // Same prefix convention as the app and mzcache_weight.cpp.
    std::string sanitized = model->name;
    std::replace(sanitized.begin(), sanitized.end(), ' ', '_');
    const std::string prefix = "layers/" + sanitized;
    ::mkdir("layers", 0755);

    const int n_layers = (int) model->hparams.n_layer;
    int generated = 0, skipped = 0;
    for (int i = 0; i < n_layers; ++i) {
        const std::string layer_file = prefix + "_layer_" + std::to_string(i) + ".bin";
        if (std::filesystem::exists(layer_file)) {
            skipped++;
            continue;
        }
        if (!model->unload_layers_to_file(prefix, i)) {
            std::cerr << "ERROR: failed to dump layer " << i << "\n";
            return 1;
        }
        std::cout << "wrote " << layer_file << " ("
                  << std::filesystem::file_size(layer_file) << " bytes)\n";
        generated++;
    }
    std::cout << "done: " << n_layers << " layers (" << generated
              << " written, " << skipped << " already present) under ./"
              << prefix << "_layer_<i>.bin\n";
    return 0;
}
