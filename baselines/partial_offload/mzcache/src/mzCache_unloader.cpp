#include "mzCache_unloader.h"

namespace mzCache{

  mzCacheUnloader::mzCacheUnloader() {};
  mzCacheUnloader::mzCacheUnloader(llama_model * model) {
    _model = model;
  };

  mzCacheUnloader::~mzCacheUnloader() {};

  void mzCacheUnloader::unload_tensors(){
    // struct impl is only 
  }

  bool mzCacheUnloader::unload_layers_to_file(const std::string &file_path,\
                                              int layer_idx) {
    // Unload specified layers to a file
    try{
        if(_model->unload_layers_to_file(file_path, layer_idx)){
            return true;
    }
    }catch(const std::exception& e){
        std::cerr << e.what() << '\n';
        return false;
    }
    return false;
  }
}
