#include "mzCache_loader.h"

namespace mzCache{
  // Todo: fill the functions
  mzCacheLoader::mzCacheLoader() {};
  mzCacheLoader::mzCacheLoader(llama_model * model) {
    _model = model;
  };
  mzCacheLoader::~mzCacheLoader() {};
  void mzCacheLoader::init() {};


bool mzCacheLoader::load_weights_from_files_pipelined(
    llama_model_loader*& ml,
    std::vector<int>& tensors_to_reload,
    double& alloc_time, double& read_time, double& copy_time) {

    // std::map<CoreType, std::vector<int>> thread_pool_config = {
    //   // { CoreType::ALLOC,     {1} },      // parallelize ALLOC
    //   // { CoreType::READ,      {2} },
    //   // { CoreType::COPY,      {3} }
    //   { CoreType::ALLOC,     {4} },      // best setting
    //   { CoreType::READ,      {6} },
    //   { CoreType::COPY,      {7} }
    // };
    // // std::map<CoreType,std::vector<int>> core_configs = {
    // //     { CoreType::BIG,    {6, 7} },        // big cores (6,7)
    // //     { CoreType::MIDDLE, {0,1,2,3,4,5} }  // middle cores (0~5)
    // // };
    // ThreadPool thread_pool(thread_pool_config);

    // // Step 1: Group tensors by layer index
    // std::map<int, std::vector<int>> layer_to_tensors;
    // for (int tidx : tensors_to_reload) {
    //     if (tidx < 0 || tidx >= _model->tensor_name_and_idx.size()) continue;
    //     int layer_idx = _model->tensor_name_and_idx[tidx].second;
    //     layer_to_tensors[layer_idx].push_back(tidx);
    // }

    // std::atomic<bool> failed{false};
    // std::mutex latency_mutex;
    // std::map<int, double> alloc_latencies, read_latencies, copy_latencies;
    // double total_alloc = 0, total_read = 0, total_copy = 0;

    // std::mutex alloc_mutex, read_mutex, copy_mutex;

    // std::map<int, std::shared_future<void>> alloc_done;
    // std::map<int, std::shared_future<void>> read_done;
    // std::map<int, std::shared_future<void>> copy_done;
    // std::map<int, std::vector<std::pair<std::unique_ptr<uint8_t[]>, size_t>>> read_buffers;

    // std::vector<std::future<void>> futures;

    // for (const auto& [layer_idx, layer_tensors] : layer_to_tensors) {
    //     // ALLOC
    //     auto alloc_promise = std::make_shared<std::promise<void>>();
    //     alloc_done[layer_idx] = alloc_promise->get_future().share();
    //     futures.push_back(thread_pool.enqueue(CoreType::ALLOC, [&, layer_idx, alloc_promise]() {
    //         if (failed) return;
    //         double atime = 0;
    //         struct timespec start, end;
    //         // std::cout << "[ALLOC] layer " << layer_idx << " on CPU " << sched_getcpu() << "\n";
    //         // std::cout << "[ALLOC] layer " << layer_idx  << "\n";
    //         clock_gettime(CLOCK_MONOTONIC, &start);
    //         {
    //             std::lock_guard<std::mutex> lock(alloc_mutex);
    //             if (!_model->alloc_buffers_layerwise_sync(ml, layer_to_tensors[layer_idx])) {
    //                 failed = true;
    //                 return;
    //             }
    //         }
    //         clock_gettime(CLOCK_MONOTONIC, &end);
    //         atime = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3;
    //         {
    //             std::lock_guard<std::mutex> lock(latency_mutex);
    //             alloc_latencies[layer_idx] = atime;
    //             total_alloc += atime;
    //         }
    //         alloc_promise->set_value();
    //     }));

    //     // READ
    //     auto read_promise = std::make_shared<std::promise<void>>();
    //     read_done[layer_idx] = read_promise->get_future().share();
    //     futures.push_back(thread_pool.enqueue(CoreType::READ, [&, layer_idx, read_promise]() {
    //         if (failed) return;
    //         alloc_done[layer_idx].wait();
    //         double rtime = 0;
    //         struct timespec start, end;
    //         // std::cout << "[READ] layer " << layer_idx << " on CPU " << sched_getcpu() << "\n";
    //         // std::cout << "[READ] layer " << layer_idx  << "\n";
            
    //         std::vector<std::pair<std::unique_ptr<uint8_t[]>, size_t>> buf;
    //         clock_gettime(CLOCK_MONOTONIC, &start);
    //         {
    //             std::lock_guard<std::mutex> lock(read_mutex);
    //             buf = _model->read_weights_to_buffers_layerwise_sync(ml, layer_to_tensors[layer_idx]);
    //         }
    //         clock_gettime(CLOCK_MONOTONIC, &end);
    //         rtime = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3;
    //         {
    //             std::lock_guard<std::mutex> lock(latency_mutex);
    //             read_latencies[layer_idx] = rtime;
    //             total_read += rtime;
    //         }
    //         read_buffers[layer_idx] = std::move(buf);
    //         read_promise->set_value();
    //     }));

    //     // COPY
    //     auto copy_promise = std::make_shared<std::promise<void>>();
    //     copy_done[layer_idx] = copy_promise->get_future().share();
    //     futures.push_back(thread_pool.enqueue(CoreType::COPY, [&, layer_idx, copy_promise]() {
    //         if (failed) return;
    //         read_done[layer_idx].wait();
    //         // std::cout << "[COPY] layer " << layer_idx << " on CPU " << sched_getcpu() << "\n";
    //         // std::cout << "[COPY] layer " << layer_idx  << "\n";
    //         double ctime = 0;
    //         struct timespec start, end;
    //         clock_gettime(CLOCK_MONOTONIC, &start);
    //         {
    //             std::lock_guard<std::mutex> lock(copy_mutex);
    //             if (!_model->copy_weights_to_backend_buffers_layerwise_sync(ml, layer_to_tensors[layer_idx], std::move(read_buffers[layer_idx]))) {
    //                 failed = true;
    //                 return;
    //             }
    //         }
    //         clock_gettime(CLOCK_MONOTONIC, &end);
    //         ctime = (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3;
    //         {
    //             std::lock_guard<std::mutex> lock(latency_mutex);
    //             copy_latencies[layer_idx] = ctime;
    //             total_copy += ctime;
    //         }
    //         copy_promise->set_value();
    //     }));
    // }

    // for (auto& f : futures) {
    //     f.wait();
    // }

    // if (failed) return false;

    // alloc_time = total_alloc;
    // read_time = total_read;
    // copy_time = total_copy;
    std::cout << "Toal alloc time " << alloc_time << "us\n";
    std::cout << "Toal read time " << read_time << "us\n";
    std::cout << "Toal copy time " << copy_time << "us\n";
    // for (auto& [l, t] : alloc_latencies) std::cout << "Layer " << l << " alloc_time: " << t << " us\n";
    // for (auto& [l, t] : read_latencies) std::cout << "Layer " << l << " read_time: " << t << " us\n";
    // for (auto& [l, t] : copy_latencies) std::cout << "Layer " << l << " copy_time: " << t << " us\n";

    return true;
}





} // namespace mzCache