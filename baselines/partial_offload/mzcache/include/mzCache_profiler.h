#include "mzCache_common.h"

/*
  mzCache Minsung Note:
  This file contains the profiler functions for mzCache.
*/



namespace mzCache{
    struct mzCache_profile
    {
      /* data */
    };


    class mzProfiler{
      public:
        // Default constructor
        mzProfiler();

        // Constructor with llama_model (use this)
        mzProfiler(llama_model * model);

        // Defualt destructor
        ~mzProfiler();

        // Profile load function 
        void load_profile();

        // Profile deconpress function
        void decompress_profile();
      
        struct mz_model_metadata get_model_metadata();
        
        // Utility functions
        void print_model_info();

        std::vector<int> get_weights();
        
        std::vector<int> get_weights_from_layers(std::vector<int>& layers);

      private:
        // Model for profile¡
        llama_model * _model;
    };


} // namespace mzCache