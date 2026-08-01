#include "llama-model.h"
#include "mzcache_types.h"
#include "mzcache_threadpool.h"

class mzcache_core;

class mzcache_weight{
public:
    int num_layers;
    std::string model_name;
    std::vector<int> layer_fds;

    int cur_layer_idx_to_unload;

    int n_cur_unload_layers = 0;

    mzcache_weight();
    mzcache_weight(mzcache_core* _core, llama_model * _model);
    ~mzcache_weight();

    bool mzcache_unload_layers(int layer_num);

    bool weight_alloc_worker(int layer_idx, std::shared_ptr<std::promise<void>> alloc_prom);
    bool weight_read_worker(int layer_idx);

    void mzcache_reload_w_layers();
    void mzcache_reload_w_layers_alloc_only();
    void mzcache_reload_w_layers_reload_only();

    // set threadpool when reconfiguring
    void set_thread_pool(ThreadPool* tp) {
        thread_pool = tp;
    }

private:
    llama_model * model;
    mzcache_core* core;
    ThreadPool* thread_pool;
};
