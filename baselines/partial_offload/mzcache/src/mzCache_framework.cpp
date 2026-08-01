#include "mzCache_framework.h"
// see mzCache_framework.h for detailed descriptions.
namespace mzCache{
  mzCacheFramework::mzCacheFramework(){
  };


  mzCacheFramework::~mzCacheFramework(){

  };

  bool mzCacheFramework::init(){
    //  0. init model load


    //  1. Profile given model (if profile == true)
    if(is_profile_needed){
      profiler = new mzCache::mzProfiler();
    }

    
    //  2. init demons 
    
    // inference engine

    // compressor-unloader
    
    // decompressor

    // loader
    loader = new mzCache::mzCacheLoader();
    // ! Todo: maybe detach() or join() threads.
    // ! Todo: must decide communication mathods between threads.
    //     (socket? pipe? whatever fast, easy and supported in android)
    //     inference engine, loader, decompressor can maybe synchronized with lock & condition_variables.
    loader_thread = std::thread(&mzCache::mzCacheLoader::init, loader);


  }

} // namespace mzCache