/*
  mzCache Note:
  mzCache core framework source code.
  At binary(future impl), code flow should be like below (as simple as possible).
    
    #include mzCache_framework.h
    int main(argc){
      ParameterParser parser(argc);

      mzCache::mzCacheFramework framework(parser.get_inital_params());

      framework.init();
    }
    
  All initalization and starting of demons (load, unload, compress, decompress, inference .. etc)
  must be hided in framework.init(), so that the execution binary is simple.
*/

#include "mzCache_common.h"
#include "mzCache_profiler.h"
#include "mzCache_loader.h"
#include <thread>

namespace mzCache{
  class mzCacheFramework{
    public:
      // mzCacheFramework constructor
      //  parses inital parameters.
      mzCacheFramework();

      // mzCacheFramework destructor
      //  signal terminate all components(demons).
      ~mzCacheFramework();

      // initalize framework (maybe with this flow?)
      //  0. init model load
      //  1. Profile given model (if profile == true)
      //  2. init demons [inference engine / compressor-unloader / decompressor-loader] in seperate threads.
      bool init();

    private:
      // model metadata
        std::vector<mzCache::mz_model_metadata*> model_metadata;

      // mzCache core components (all initalized at init() and pointers are owned here at mzCacheFramework class)
        // Profiler
        mzCache::mzProfiler* profiler;

        // inference engine

        // compressor-unloader
        
        // decompressor-loader
        mzCache::mzCacheLoader * loader;
        std::thread loader_thread;
        // scheduler?

        // memory monitor

      // Inital parameters
      bool is_profile_needed = false;

  };

} // namespace mzCache

