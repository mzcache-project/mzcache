#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <stdexcept>
#include <map>
#include <cstdio>
#include <cstdlib>

// Core type definitions (for API compatibility)
enum class CoreType {
    ALLOC,
    READ,
    DECOMP,
    DECOMP_KERNEL,
    FREE,
};

class ThreadPool {
public:
  ThreadPool(const std::map<CoreType,std::vector<int>>& cfg) : stop(false) {
    // Create one ThreadGroup per CoreType
    for (auto& [type, cores] : cfg) {
      // Capturing a structured binding in a lambda is C++20; copy to a local
      // so this header also compiles with C++17 toolchains (e.g. NDK r25).
      const CoreType group_type = type;
      auto& grp = groups[group_type];  // groups: map<CoreType,ThreadGroup>
      // Spawn one thread per core_id
      for (int core_id : cores) {
        grp.workers.emplace_back(
          [this,group_type,core_id]{
            set_affinity(core_id);
            this->worker_loop(group_type);
          }
        );
      }
    }
  }

  ~ThreadPool() {
    // Signal stop to all groups and join
    {
      std::unique_lock lk(global_mutex);
      stop = true;
    }
    for (auto& [t,grp] : groups) grp.condition.notify_all();
    for (auto& [t,grp] : groups)
      for (auto& w : grp.workers) if (w.joinable()) w.join();
  }

  template<class F, class... A>
  auto enqueue(CoreType type, F&& f, A&&... a) {
    using R = std::invoke_result_t<F,A...>;
    // A worker task that throws must fail loudly: packaged_task would stash
    // the exception in a future nobody reads, the task's layer_sync
    // taskDone() never runs, and the next swapin deadlocks with no output.
    auto task = std::make_shared<std::packaged_task<R()>>(
      [bound = std::bind(std::forward<F>(f), std::forward<A>(a)...)]() mutable -> R {
        try {
          return bound();
        } catch (const std::exception & e) {
          fprintf(stderr, "[MZCACHE][FATAL] worker task threw: %s\n", e.what());
          fflush(stderr);
          abort();
        } catch (...) {
          fprintf(stderr, "[MZCACHE][FATAL] worker task threw a non-std exception\n");
          fflush(stderr);
          abort();
        }
      }
    );
    auto fut = task->get_future();
    {
      auto& grp = groups.at(type);
      std::unique_lock lk(grp.mtx);
      if (stop) throw std::runtime_error("stopped");
      grp.tasks.emplace([task]{ (*task)(); });
    }
    groups.at(type).condition.notify_one();
    return fut;
  }

  void wait_idle(CoreType type) {
      auto& grp = groups.at(type);
      std::unique_lock lk(grp.mtx);
      grp.idle_cv.wait(lk, [&]{ return grp.tasks.empty() && grp.active == 0; });
  }

  void wait_idle_all() {
      for (auto& [t, grp] : groups) {
          auto& g = grp;  // C++17: structured bindings cannot be captured
          std::unique_lock lk(g.mtx);
          g.idle_cv.wait(lk, [&]{ return g.tasks.empty() && g.active == 0; });
      }
  }

private:
  struct ThreadGroup {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable condition;

    std::atomic<int>                    active{0};
    std::condition_variable             idle_cv;
  };

  std::map<CoreType,ThreadGroup> groups;
  std::mutex global_mutex;
  bool stop;

  void worker_loop(CoreType type) {
    auto& grp = groups.at(type);
    for (;;) {
      std::function<void()> t;
      {
        std::unique_lock lk(grp.mtx);
        grp.condition.wait(lk, [&]{ return stop || !grp.tasks.empty(); });
        if (stop && grp.tasks.empty()) return;
        t = std::move(grp.tasks.front());
        grp.tasks.pop();
        grp.active++;  
      }

      t();

      {
        std::lock_guard lk(grp.mtx);
        if(--grp.active == 0 && grp.tasks.empty()) {
          grp.idle_cv.notify_all();
        }
      }
    }
  }

  void set_affinity(int core_id) {
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core_id,&cs);
    if (sched_setaffinity(0,sizeof(cs),&cs)!=0) perror("affinity");
  }
};

