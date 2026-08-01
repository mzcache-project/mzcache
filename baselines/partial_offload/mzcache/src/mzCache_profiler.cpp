/*
  mzCache Minsung Note:
  This file contains the profiler functions for mzCache.
*/
#include "mzCache_profiler.h"

namespace mzCache{
  mzProfiler::mzProfiler(){

  }

  mzProfiler::mzProfiler(llama_model* model){
    std::cout << "[mzProfiler] Setup model: " << model->name << "\n";
    _model = model;
  }

  mzProfiler::~mzProfiler(){

  }

  std::vector<int> mzProfiler::get_weights(){
    std::vector<int> attention_tensor_indices;
    int tensor_idx = 0;
    // std::cout << "get attention tensor to unload idx : ";
    for (const auto & name_tensor : _model->tensor_name_and_idx) {
        if (name_tensor.first == "layer") {
            // std::cout << tensor_idx << " ";
            attention_tensor_indices.push_back(tensor_idx);
        }
        tensor_idx++;
    }
    // std::cout << "\n";
    return attention_tensor_indices;
  }
  
  std::vector<int> mzProfiler::get_weights_from_layers(std::vector<int>& layers){
    std::vector<int> attention_tensor_indices;
    int tensor_idx = 0;

    for (const auto & name_tensor : _model->tensor_name_and_idx) {
        const std::string& type = name_tensor.first;
        int layer_idx = name_tensor.second;

        if (type == "layer" &&
            std::find(layers.begin(), layers.end(), layer_idx) != layers.end()) {
            attention_tensor_indices.push_back(tensor_idx);
        }
        tensor_idx++;
    }
    return attention_tensor_indices;  
  }

  void  mzProfiler::print_model_info(){
    int tensor_idx = 0;
    std::cout << "Print model " << _model->name << " tensors and indices" << "\n";
    std::cout << "Total number of weight tensors: " << _model->tensor_name_and_idx.size() << "\n";
    for(auto & name_tensor : _model->tensor_name_and_idx){
      std::cout << "tensor [" << tensor_idx << "] layer: " << name_tensor.second << \
                   " type: " << name_tensor.first << "\n";
      tensor_idx++;
    }
  }

  struct mz_model_metadata mzProfiler::get_model_metadata(){
    struct mz_model_metadata new_metadata;
    // how to get layer and tensor mappings?

    /*
      mzCache Minsung Note:
      Todo: avoid getting tensor metadata from ggml.
            (For now, there's absolutely no way to get metadata from llama becasue 
            they simply holds the struct pointers and raw data pointers)
    */
    // for(auto & buf_ptr : model->pimpl->bufs){
      
    // }
    return new_metadata;
  }
} // namespace mzCache