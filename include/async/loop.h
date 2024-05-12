#pragma once 

#include <cstddef>
#include <thread>
#include <functional>

#include "util/timer.h"
#include "async/task.h"
#include "async/future.h"
#include "async/async_impl/basic_loop.h"
#include "async/async_impl/uring_loop.h"

namespace spy {

    class ThreadLocalLoop {
    private:
        BasicLoop *basic_loop_ptr_;
        UringLoop *uring_loop_ptr_;

    public:
        ThreadLocalLoop(): basic_loop_ptr_(nullptr), uring_loop_ptr_(nullptr) {}

    public:
        bool has_basic_loop() const { return basic_loop_ptr_ != nullptr; }
        bool has_uring_loop() const { return uring_loop_ptr_ != nullptr; }

        BasicLoop &get_basic_loop() { return *basic_loop_ptr_; }
        UringLoop &get_uring_loop() { return *uring_loop_ptr_; }

        void set_basic_loop(BasicLoop *loop_ptr) { basic_loop_ptr_ = loop_ptr; }
        void set_uring_loop(UringLoop *loop_ptr) { uring_loop_ptr_ = loop_ptr; }
    };

    inline static thread_local ThreadLocalLoop global_loop;

    inline BasicLoop &get_thread_local_basic_loop() { return global_loop.get_basic_loop(); }

    inline UringLoop &get_thread_local_uring_loop() { return global_loop.get_uring_loop(); }

    struct SystemLoop {
    public:
        using Duration = std::chrono::system_clock::duration;

    private:
        std::unique_ptr<BasicLoop[]>    basic_loop_array_;

        std::unique_ptr<UringLoop[]>    uring_loop_array_;

        std::unique_ptr<std::thread[]>  thread_array_;

        size_t                     num_worker_;

        std::stop_source                stop_source_;

    public:
        SystemLoop() = default;

        SystemLoop(SystemLoop &&) = delete;

        ~SystemLoop() { stop(); }

    public:
        BasicLoop &get_any_worker_loop()              { return basic_loop_array_[0];          }

        BasicLoop &current_worker_loop()              { return global_loop.get_basic_loop();  }

        BasicLoop &get_worker_loop(size_t index) { return basic_loop_array_[index];      }

    public:
        bool is_started() const noexcept                    { return thread_array_ != nullptr; }

        static bool is_this_thread_worker() noexcept        { return global_loop.has_basic_loop(); }

        size_t this_thread_worker_id() const noexcept  { return &global_loop.get_basic_loop() - basic_loop_array_.get(); }

        size_t num_worker() const noexcept             { return num_worker_; }

    public:
        /*!
         * @brief Start worker threads
         * @param num_worker The number of worker threads
         * @param num_batch_wait The number of tasks batched to execute at once
         * @param batch_time_out
         * @param batch_time_out_delta
         */
        void start(size_t num_worker = 0, size_t num_batch_wait = 1,
                Duration batch_time_out = std::chrono::milliseconds(15),
                Duration batch_time_out_delta = std::chrono::milliseconds(12)) {
                
            if (thread_array_) [[unlikely]] { throw std::runtime_error("loop already started"); }

            bool set_affinity = false;
            if (num_worker < 1) [[unlikely]] {
                num_worker   = std::thread::hardware_concurrency();
                set_affinity = true;
            }

            thread_array_       = std::make_unique<std::thread[]>(num_worker);
            basic_loop_array_   = std::make_unique<BasicLoop[]>(num_worker);
            uring_loop_array_   = std::make_unique<UringLoop[]>(num_worker);
            num_worker_         = num_worker;

            for (size_t i = 0; i < num_worker; ++i) {
                thread_array_[i] = std::thread(&SystemLoop::thread_entry, this, i,
                                        num_worker, num_batch_wait,
                                        batch_time_out, batch_time_out_delta);
    #if defined(__linux__) && defined(_GLIBCXX_HAS_GTHREADS)
                if (set_affinity) {
                    pthread_t handle = thread_array_[i].native_handle();
                    cpu_set_t cpuset;
                    CPU_ZERO(&cpuset);
                    CPU_SET(i, &cpuset);
                    pthread_setaffinity_np(handle, sizeof(cpuset), &cpuset);
                }
    #elif defined(_WIN32)
                if (set_affinity && num_worker <= 64) {
                    SetThreadAffinityMask(thread_array_[i].native_handle(), 1ull << i);
                }
    #endif
            }
        }

        /*!
         * @brief Blockingly wait for all worker threads stop
         */
        void stop() {
            if (thread_array_) {
                // Set stop flag
                stop_source_.request_stop();
                // Wait for all threads
                for (size_t i = 0; i < num_worker_; ++i) { thread_array_[i].join(); }
                // Release all resources
                thread_array_.reset();
                basic_loop_array_.reset();
                uring_loop_array_.reset();
            }
        }

    public:
        /*!
         * @brief Distribute a task to coroutine pool without trace
         * @param task The coroutine task 
         */
        template <class T, class T_Promise>
        inline void co_spawn(Task<T, T_Promise> &&task) {
            return loop_enqueue_detach(current_worker_loop(), std::move(task));
        }

        /*!
         * @brief Distribute a task to coroutine pool and get a future value
         * @param task The coroutine task
         * @return The future return value
         */
        template <class T, class T_Promise>
        inline Future<T> co_future(Task<T, T_Promise> task) {
            return loop_enqueue_future(current_worker_loop(), std::move(task));
        }

        /*!
         * @brief Distribute a task to a specific worker in the coroutine pool without trace
         * @param worker_id The id of coroutine worker
         * @param task The coroutine task 
         */
        template <class T, class T_Promise>
        inline void co_spawn(size_t worker_id, Task<T, T_Promise> task) {
            return loop_enqueue_detach(get_worker_loop(worker_id), std::move(task));
        }

        /*!
         * @brief Distribute a task to a specific worker in the coroutine pool and get a future value
         * @param worker_id The id of coroutine worker
         * @param task The coroutine task
         * @return The future return value
         */
        template <class T, class T_Promise>
        inline Future<T> co_future(size_t worker_id, Task<T, T_Promise> task) {
            return loop_enqueue_future(get_worker_loop(worker_id), std::move(task));
        }

        /*!
         * @brief Execute a coroutine synchronously
         * @param task The coroutine task
         * @return The return value of the coroutine task
         */
        template <class T, class T_Promise>
        inline auto co_synchronize(Task<T, T_Promise> task) {
            if (!is_started()) { start(); }
            return loop_enqueue_synchronized(get_any_worker_loop(), std::move(task));
        }

        /*!
         * @brief Generate a coroutine task with invocable object
         * @param func The invocable object for coroutine task
         * @param args The arguments of the invocable object
         * @return An coroutine task with function bound with several arguments
         */
        template <class F, class... Args>
            requires std::is_invocable_r_v<Task<>, F, Args...>
        inline Task<> co_bind(F &&func, Args &&...args) {
            Task<> task = [](auto func) mutable -> Task<> {
                co_await std::move(func)();
            }(std::bind(std::forward<F>(func), std::forward<Args>(args)...));
            return task;
        }

    protected:
        void thread_entry(int i, int num_worker, int num_batch_wait,
                        Duration batch_time_out,
                        Duration batch_time_out_delta) {
            // Get loop structure
            auto &cur_basic_loop = basic_loop_array_[i];
            auto &cur_uring_loop = uring_loop_array_[i];
            global_loop.set_basic_loop(&cur_basic_loop);
            global_loop.set_uring_loop(&cur_uring_loop);

            // steal work from other threads
            const auto steal_work = [&] {
                for (int j = 1; j < num_worker; ++j) {
                    auto other_idx = (i + j) % num_worker;
                    if (basic_loop_array_[other_idx].run()) { return true; }
                }
                return false;
            };

        compute: /* Execute */
            cur_basic_loop.run();

        event: /* Get new event */

            // If there is any left uring task, execute it
            if (const size_t num_event = cur_uring_loop.has_event()) {
                cur_uring_loop.run_batched_and_nowait(num_event);
                goto compute;
            }

            // If successfully steal one task from other threads, execute it.
            if (steal_work()) { goto compute; }

            auto timeout = batch_time_out;
            while (!stop_source_.stop_requested()) [[likely]] {
                auto ts = duration_to_kernel_timespec(timeout);
                // execute uring task directly 
                if (cur_uring_loop.run_batched_and_wait(num_batch_wait, &ts)) { goto compute; }
                // If there is any left basic task, execute it.
                if (cur_basic_loop.run()) { goto event; }
                // If successfully steal one task from other threads, execute it.
                if (steal_work()) { goto compute; }

                timeout += batch_time_out_delta;
            }
        }

    };

} // namespace spy