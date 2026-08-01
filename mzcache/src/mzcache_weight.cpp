#include "mzcache_weight.h"
#include "mzcache_core.h"

#include <fcntl.h>      // open, O_RDONLY, O_DIRECT
#include <unistd.h>     // pread, close


mzcache_weight::mzcache_weight(mzcache_core* _core, llama_model * _model)
    : core(_core), model(_model)
{
    num_layers = model->hparams.n_layer;
    model_name = model->name;

    layer_fds.assign(num_layers, -1);

    thread_pool = _core->get_thread_pool();
    cur_layer_idx_to_unload = num_layers - 1;
}

mzcache_weight::~mzcache_weight() {
    for (int i = 0; i < num_layers; ++i) {
        if (layer_fds[i] >= 0) {
            ::close(layer_fds[i]);
            layer_fds[i] = -1; // reset to -1 after closing
        }

        // std::string fn = "./layers/" + model_name + "_layer_" + std::to_string(i) + ".bin";
        // if (::unlink(fn.c_str()) != 0) {
        //     // Ignore a missing file; report only other errors
        //     if (errno != ENOENT) {
        //         perror(("Failed to delete " + fn).c_str());
        //     }
        // }
        // else {
        //     printf("Deleted file: %s\n", fn.c_str());
        // }
    }
}

bool mzcache_weight::mzcache_unload_layers(int layer_num) {
    // Unload the weight layer
    for(int i = 0; i < layer_num; ++i) {
        // Never free weights that cannot be restored: reload reads
        // ./layers/<model>_layer_<n>.bin relative to the cwd, so unloading
        // without a readable dump would strand the layer. Refuse instead —
        // swapout handles this as an ordinary unload failure and stops at
        // the ratio achieved so far.
        std::string fn = "./layers/" + model_name + "_layer_" +
                         std::to_string(cur_layer_idx_to_unload) + ".bin";
        if (::access(fn.c_str(), R_OK) != 0) {
            MZ_LOG_ERROR("weight layer dump %s is not readable (wrong cwd?) — refusing to unload layer %d",
                         fn.c_str(), cur_layer_idx_to_unload);
            return false;
        }

        if(model->unload_buffers(cur_layer_idx_to_unload)) {
            cur_layer_idx_to_unload--;
        }
        else {
            MZ_LOG_ERROR("Failed to unload weight layer %d", cur_layer_idx_to_unload);
            return false;
        }

        n_cur_unload_layers++;
    }
    return true;
}


bool mzcache_weight::weight_alloc_worker(int layer_idx, std::shared_ptr<std::promise<void>> alloc_prom) {
    bool success = model->alloc_weight_layer(layer_idx);

    if (success) {
        alloc_prom->set_value();
    }
    else {
        MZ_LOG_ERROR("Failed to allocate weight layer %d", layer_idx);
        alloc_prom->set_exception(std::make_exception_ptr(std::runtime_error("Failed to allocate weight layer")));
    }

    return success;
}

bool mzcache_weight::weight_read_worker(int layer_idx) {
    std::string fn = "./layers/" + model_name + "_layer_" + std::to_string(layer_idx) + ".bin";
    // std::cout << "Reading weight layer " << layer_idx
    //           << " from file: " << fn << std::endl;

    int fd = ::open(fn.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) throw std::runtime_error("open failed for " + fn);
    layer_fds[layer_idx] = fd;

    // Wait for the allocation to complete
    // std::cout << "Waiting for weight allocation for layer " << layer_idx << std::endl;

    core->wait_for_weight_allocation(layer_idx);

    // std::cout << "Reading weight layer " << layer_idx
    //           << " from file: " << fn << std::endl;

    // Read the weight layer from the specified file
    bool success = model->read_weight_layer(layer_fds[layer_idx], layer_idx);

    if(core->layer_sync.size() == num_layers) {
        core->layer_sync[layer_idx]->taskDone();
    }

    if(layer_fds[layer_idx] >= 0) {
        std::cout << "Why is this fd not closed? " << layer_fds[layer_idx] << std::endl;
        ::close(layer_fds[layer_idx]);
        layer_fds[layer_idx] = -1; // Reset the file descriptor after reading
    }

    return success;
}


void mzcache_weight::mzcache_reload_w_layers() {

    int prev_layer_idx = cur_layer_idx_to_unload;

    std::cout << "Enqueuing read weight layers from " << prev_layer_idx + 1
              << " to " << num_layers - 1 << std::endl;

    for(int L = prev_layer_idx + 1; L < num_layers; L++) {
        auto alloc_prom = std::make_shared<std::promise<void>>();
        core->register_weight_allocation(L, alloc_prom);

        // std::cout << "Enqueuing allocation for layer " << L << std::endl;
        // Enqueue allocation and read tasks for the weight layer

        thread_pool->enqueue(
            CoreType::ALLOC,
            std::bind(&mzcache_weight::weight_alloc_worker, this, L, alloc_prom)
        );
        thread_pool->enqueue(
            CoreType::READ,
            std::bind(&mzcache_weight::weight_read_worker, this, L)
        );
    }

    // auto cur_time = std::chrono::high_resolution_clock::now();
    // //print current absolute time % 1000
    // std::cout << "Enqueued read weight layers at: "
    //           << std::chrono::duration_cast<std::chrono::milliseconds>(cur_time.time_since_epoch()).count() % 1000
    //           << " ms" << std::endl;
    cur_layer_idx_to_unload = num_layers - 1; // Reset to the last layer after enqueueing
}


void mzcache_weight::mzcache_reload_w_layers_alloc_only() {

    int prev_layer_idx = cur_layer_idx_to_unload;

    for(int L = prev_layer_idx + 1; L < num_layers; L++) {
        auto alloc_prom = std::make_shared<std::promise<void>>();
        core->register_weight_allocation(L, alloc_prom);

        // std::cout << "Enqueuing allocation for layer " << L << std::endl;
        // Enqueue allocation and read tasks for the weight layer

        thread_pool->enqueue(
            CoreType::ALLOC,
            std::bind(&mzcache_weight::weight_alloc_worker, this, L, alloc_prom)
        );
    }

    for(int L = prev_layer_idx + 1; L < num_layers; L++) {
        core->wait_for_weight_allocation(L);
    }
}


void mzcache_weight::mzcache_reload_w_layers_reload_only() {

    int prev_layer_idx = cur_layer_idx_to_unload;

    for(int L = prev_layer_idx + 1; L < num_layers; L++) {
        thread_pool->enqueue(
            CoreType::READ,
            std::bind(&mzcache_weight::weight_read_worker, this, L)
        );
    }
    thread_pool->wait_idle(CoreType::READ);
    cur_layer_idx_to_unload = num_layers - 1; // Reset to the last layer after enqueueing
}
