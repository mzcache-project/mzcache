/*
  mzCache Minsung Note:
  This file contains the declarations of overloaded functions in llama.cpp for mzCache.
*/

#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "log.h"
#include "llama.h"
#include "llama-impl.h"
#include "llama-cpp.h"
#include "llama-model-loader.h"
#include "llama-chat.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-saver.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <cmath>

#pragma once

namespace mzCache{
    /*
      mzCache Minsung Note:
      Struct for maintaining model metadata
      - model name (file)
      - Layer-tensor mappings
      - Load / decompress profile
      - Execution plan (layer wise)
      - misc settings
      - ...
    */
    struct mz_model_metadata{
      std::string name;
      std::pair<int, std::vector<int>> layer_tensor_pair;
    };


    /*
      mzCache Minsung Note:
      Overloaded common_init_result for mzCache.
      (contians model_loader)
    */
    struct common_init_result {
        llama_model_ptr   model;
        llama_context_ptr context;
        llama_model_loader* model_loader;
        std::vector<llama_adapter_lora_ptr> lora;
    };

    /*
      mzCache Minsung Note:
      Parameter struct
    */
    struct mz_parameter{
      // maybe parse from yaml file?
      int dummy;
    };

    struct mzCache::common_init_result mz_common_init_from_params(common_params & params);


    llama_model_loader* mz_llama_model_load(const std::string & fname,
                                            std::vector<std::string> & splits,
                                            llama_model & model, llama_model_params & params);

    static struct llama_model * mz_llama_model_load_from_file_impl(
            const std::string & path_model,
            std::vector<std::string> & splits,
            struct llama_model_params params,
            llama_model_loader *& ml);

    struct llama_model * mz_llama_model_load_from_file(
            const char * path_model,
            struct llama_model_params params,
            llama_model_loader *& ml);

    bool mz_llama_model_load_buffers(struct llama_model * model,
                                     llama_model_loader *& ml,
                                     std::vector<int>& tensors_to_reload,
                                     double & weight_alloc_time,
                                     double & weight_read_time);


    // Parameter parser class for future features
    // e.g., yaml format or..
    class ParameterParser{
      public:
        ParameterParser();
        ~ParameterParser();

        const mz_parameter & get_initial_param() const;
      private:
        mz_parameter initial_parameter;
      
    };

} // namespace mzCache
