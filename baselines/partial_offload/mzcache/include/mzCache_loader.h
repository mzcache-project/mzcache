/*
  mzCache Minsung Note:
  mzCache loader implementation
  must run run() in seperate thread
  Todo: consider weight load
*/

#include "llama-model.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"

#include "llama-kv-cache-unified.h"
#include "llama-kv-cache-unified-iswa.h"
#include "mzCache_threadpool.h"

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
#include <fstream>

namespace mzCache{
  class mzCacheLoader{
    public:
      mzCacheLoader();
      mzCacheLoader(llama_model * model);
      ~mzCacheLoader();

      void init();
      bool load_weights_from_files_pipelined(llama_model_loader *& ml,
                                    std::vector<int>& tensors_to_reload,
                                    double& alloc_time, double& read_time, double& copy_time);
    private:
      llama_model * _model;
  };
}