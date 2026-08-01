/*
  mzCache Minsung Note:
  mzCache unloader implementation
*/
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"

#include "llama-kv-cache-unified.h"
#include "llama-kv-cache-unified-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

// #include "ggml-cpp.h"
// #include "ggml-backend-impl.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace mzCache{
  class mzCacheUnloader{
    public:
      mzCacheUnloader();
      mzCacheUnloader(llama_model * model);
      ~mzCacheUnloader();

      void init();
      void unload_tensors();

      // Unload tensors in layers in the model to file.
      bool unload_layers_to_file(const std::string &file_path,
                                 int layer_idx);

    private:
      llama_model * _model;
  };
}