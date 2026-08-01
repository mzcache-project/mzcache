/*
  mzCache Minsung Note:
  This file contains the declarations of overloaded functions in llama.cpp for mzCache.
*/

#include "mzCache_common.h"

namespace mzCache{
 
 // Todo: change return type for model loader.
struct mzCache::common_init_result mz_common_init_from_params(common_params & params) {
    mzCache::common_init_result iparams;
    auto mparams = common_model_params_to_llama(params);
    llama_model_loader * model_loader = nullptr;

    auto t_model_start = std::chrono::high_resolution_clock::now();
    llama_model * model = \
        mzCache::mz_llama_model_load_from_file(params.model.path.c_str(), mparams, model_loader);
    if (model == NULL) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return iparams;
    }
    /*
      mzCache Minsung Note:
      Todo: Better consider safer implementation for carrying out model_loader.
    */ 
    iparams.model_loader = model_loader;

    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto cparams = common_context_params_to_llama(params);

    llama_context * lctx = llama_init_from_model(model, cparams);

    if (lctx == NULL) {
        LOG_ERR("%s: failed to create context with model '%s'\n", __func__, params.model.path.c_str());
        llama_model_free(model);
        return iparams;
    }

    if (params.ctx_shift && !llama_memory_can_shift(llama_get_memory(lctx))) {
        LOG_WRN("%s: KV cache shifting is not supported for this context, disabling KV cache shifting\n", __func__);
        params.ctx_shift = false;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_model_n_layer(model);

        const auto cvec = common_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            llama_free(lctx);
            llama_model_free(model);

            return iparams;
        }

        int err = llama_apply_adapter_cvec(
                lctx,
                cvec.data.data(),
                cvec.data.size(),
                cvec.n_embd,
                params.control_vector_layer_start,
                params.control_vector_layer_end);
        if (err) {
            llama_free(lctx);
            llama_model_free(model);

            return iparams;
        }
    }

    if (llama_pooling_type(lctx) == LLAMA_POOLING_TYPE_RANK) {
        bool ok = true;

        if (llama_vocab_bos(vocab) == LLAMA_TOKEN_NULL) {
            LOG_WRN("%s: warning: vocab does not have a  BOS token, reranking will not work\n", __func__);
            ok = false;
        }

        bool has_eos = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;
        bool has_sep = llama_vocab_sep(vocab) != LLAMA_TOKEN_NULL;

        if (!has_eos && !has_sep) {
            LOG_WRN("%s: warning: vocab does not have an EOS token or SEP token, reranking will not work\n", __func__);
            ok = false;
        } else if (!has_eos) {
            LOG_WRN("%s: warning: vocab does not have an EOS token, using SEP token as fallback\n", __func__);
        } else if (!has_sep) {
            LOG_WRN("%s: warning: vocab does not have a SEP token, reranking will not work\n", __func__);
            ok = false;
        }

        if (!ok) {
            llama_free(lctx);
            llama_model_free(model);

            return iparams;
        }
    }

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        llama_adapter_lora_ptr lora;
        lora.reset(llama_adapter_lora_init(model, la.path.c_str()));
        if (lora == nullptr) {
            LOG_ERR("%s: failed to apply lora adapter '%s'\n", __func__, la.path.c_str());
            llama_free(lctx);
            llama_model_free(model);
            return iparams;
        }

        la.ptr = lora.get();
        iparams.lora.emplace_back(std::move(lora)); // copy to list of loaded adapters
    }

    if (!params.lora_init_without_apply) {
        common_set_adapter_lora(lctx, params.lora_adapters);
    }

    if (params.sampling.ignore_eos && llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
        LOG_WRN("%s: warning: vocab does not have an EOS token, ignoring --ignore-eos\n", __func__);
        params.sampling.ignore_eos = false;
    }

    if (params.sampling.ignore_eos) {
        for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
            if (llama_vocab_is_eog(vocab, i)) {
                LOG_INF("%s: added %s logit bias = %f\n", __func__, common_token_to_piece(lctx, i).c_str(), -INFINITY);
                params.sampling.logit_bias.push_back({i, -INFINITY});
            }
        }
    }

    if (params.sampling.penalty_last_n == -1) {
        LOG_INF("%s: setting penalty_last_n to ctx_size = %d\n", __func__, llama_n_ctx(lctx));
        params.sampling.penalty_last_n = llama_n_ctx(lctx);
    }

    if (params.sampling.dry_penalty_last_n == -1) {
        LOG_INF("%s: setting dry_penalty_last_n to ctx_size = %d\n", __func__, llama_n_ctx(lctx));
        params.sampling.dry_penalty_last_n = llama_n_ctx(lctx);
    }

    if (params.warmup) {
        LOG_WRN("%s: warming up the model with an empty run - please wait ... (--no-warmup to disable)\n", __func__);

        llama_set_warmup(lctx, true);

        std::vector<llama_token> tmp;
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);

        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        if (llama_model_has_encoder(model)) {
            llama_encode(lctx, llama_batch_get_one(tmp.data(), tmp.size()));
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch)));
        }
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_synchronize(lctx);
        llama_perf_context_reset(lctx);
        llama_set_warmup(lctx, false);
    }

    iparams.model.reset(model);
    iparams.context.reset(lctx);

    return iparams;
}

bool mz_llama_model_load_buffers(struct llama_model * model, llama_model_loader *& ml,
                                 std::vector<int>& tensors_to_reload, double & weight_alloc_time,
                                 double & weight_read_time){
  try{
    if(model->load_buffers(ml, tensors_to_reload, weight_alloc_time, weight_read_time)){
        return true;
    }
    fprintf(stderr, "%s: load_buffers function is deprecated. %s\n", __func__, "use alloc_buffers_layerwise_sync() instead");
  }
  catch(const std::exception& e){
    LOG_ERR("%s: failed to load buffers from model loader: %s\n", __func__, e.what());
  }
  return false;
}


/*
mzCache Minsung Note:
This function loads and allocates tensors for given model and returns llama_model_loader.
(vanilla llama.cpp::llama_model_load() only returns inteager status code)
*/
llama_model_loader* mz_llama_model_load(const std::string & fname,
                                        std::vector<std::string> & splits,
                                        llama_model & model, llama_model_params & params){
// Todo: fill the body
    // loading time will be recalculated after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = 0;
    time_meas tm(model.t_load_us);

    model.t_start_us = tm.t_start_us;
    llama_model_loader* ml_ = new llama_model_loader(fname, splits, params.use_mmap,
           params.check_tensors, params.kv_overrides, params.tensor_buft_overrides);
    try {
        // ml_->print_info();

        model.hparams.vocab_only = params.vocab_only;

        try {
            model.load_arch(*ml_);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            model.load_hparams(*ml_);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            model.load_vocab(*ml_);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model.load_stats(*ml_);
        // model.print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0;
        }

        if (!model.load_tensors(*ml_)) {
            return nullptr;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return nullptr;
    }
    std::cout << "Model loaded successfully from " << fname << "\n";
    return ml_;
}


/*
  mzCache Minsung Note:
  Load model from file and returns model_loader*.
*/
static struct llama_model * mz_llama_model_load_from_file_impl(
        const std::string & path_model,
        std::vector<std::string> & splits,
        struct llama_model_params params,
        llama_model_loader *& ml) {
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    llama_model * model = new llama_model(params);

    // create list of devices to use with this model
    if (params.devices) {
        for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
            model->devices.push_back(*dev);
        }
    } else {
        std::vector<ggml_backend_dev_t> rpc_servers;
        // use all available devices
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            switch (ggml_backend_dev_type(dev)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:
                case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                    // skip CPU backends since they are handled separately
                    break;

                case GGML_BACKEND_DEVICE_TYPE_GPU:
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                    if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                        rpc_servers.push_back(dev);
                    } else {
                        model->devices.push_back(dev);
                    }
                    break;
            }
        }
        // add RPC servers at the front of the list
        if (!rpc_servers.empty()) {
            model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
        if (params.main_gpu < 0) {
            model->devices.clear();
        } else {
            if (params.main_gpu >= (int)model->devices.size()) {
                LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, params.main_gpu, model->devices.size());
                llama_model_free(model);
                return nullptr;
            }
            ggml_backend_dev_t main_gpu = model->devices[params.main_gpu];
            model->devices.clear();
            model->devices.push_back(main_gpu);
        }
    }

    for (auto * dev : model->devices) {
        size_t free, total; // NOLINT
        ggml_backend_dev_memory(dev, &free, &total);
        LLAMA_LOG_INFO("%s: using device %s (%s) - %zu MiB free\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), free/1024/1024);
    }

    ml = mz_llama_model_load(path_model, splits, *model, params);
    return model;
}

struct llama_model * mz_llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params,
        llama_model_loader *& ml) {
    std::vector<std::string> splits = {};
    return mz_llama_model_load_from_file_impl(path_model, splits, params, ml);
}



}  // mzCache