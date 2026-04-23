#include "hfm/runtime/thread_pool.hpp"

namespace hfm {

ThreadPool::ThreadPool(std::size_t n_threads) {
    if (n_threads == 0) {
        n_threads = std::thread::hardware_concurrency();
    }
    for (std::size_t i = 0; i < n_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    stop_ = true;
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

ThreadPool& global_thread_pool() {
    static ThreadPool pool;
    return pool;
}

} // namespace hfm
