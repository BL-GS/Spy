/*
 * @author: BL-GS 
 * @date:   24-4-13
 */

#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <concurrentqueue/concurrentqueue.h>

#include "util/shell/logger.h"
#include "backend/cpu/type.h"
#include "backend/cpu/operator_impl.h"

namespace spy {

	class DefaultCPUBackend;

	class DefaultThreadPool {
		friend class DefaultCPUBackend;
	public:
		using TaskFunc  = std::function<void()>;
		using TaskQueue = moodycamel::ConcurrentQueue<TaskFunc>;

	private:
		/// Worker threads
		std::vector<std::thread>    	workers_;
		/// The task queue buffering all tasks
		TaskQueue                       task_queue_;
		/// Stop all threads if true
		std::atomic_flag        		stop_flag_;

	public:
		DefaultThreadPool() = default;

		~DefaultThreadPool() noexcept {
			stop_flag_.test_and_set();
			// Push nullptr to notifying all threads
			for (size_t i = 0; i < workers_.size(); ++i) { task_queue_.enqueue(nullptr); }
			// Wait for all threads
			for (auto &worker: workers_) { worker.join(); }
		}

	protected:
		void reserve(uint32_t num_thread) {
			workers_.reserve(num_thread);
			for (size_t i = 0; i < num_thread; ++i) {
				workers_.emplace_back([this, i](){
					while (!stop_flag_.test()) {
						// Sleep if no task
						TaskFunc task;
						// Try to fetch a task
						while (!task_queue_.try_dequeue(task)) { std::this_thread::yield(); }
						// Execute task
						if (task) { task(); }
					}
				});
			}
		}

	public:
		int submit(std::function<void(int)> &&task_func, int concurrency) {
			if (concurrency == 1) {
				task_queue_.enqueue([func=std::move(task_func)](){ func(0); });
				return 0;
			}
			std::vector<TaskFunc> func_bulk;
			func_bulk.reserve(concurrency);
			for (int i = 0; i < concurrency; ++i) { func_bulk.emplace_back([task_func, i] { task_func(i); }); }
			task_queue_.enqueue_bulk(func_bulk.begin(), concurrency);
			return 0;
		}
	public:
		bool poll() { return task_queue_.size_approx() == 0; }

		void sync()  { while (task_queue_.size_approx() != 0) { std::this_thread::yield(); } }

		size_t max_concurrency() const { return workers_.size(); }
	};

	class DefaultCPUBackend final: public CPUBackend {
	private:
		DefaultThreadPool thread_pool_;

	public:
		DefaultCPUBackend(int num_thread, int64_t max_mem) {
			const size_t max_num_thread  = CPUBackend::get_max_concurrency();
			const size_t real_num_thread = (num_thread < 0) ? max_num_thread : num_thread;
			if (real_num_thread > max_num_thread) {
				spy_warn("The number of threads in pool ({}) is larger than the maximum concurrency ({})",
					real_num_thread, max_num_thread);
			}
			thread_pool_.reserve(num_thread);

			const size_t max_mem_size  =  CPUBackend::get_max_memory_capacity();
			const size_t real_mem_size = (max_mem < 0) ? max_mem_size : max_mem;
			if (real_mem_size > max_mem_size) {
				spy_warn("The size of memory in pool ({}) is larger than the maximum memory capacity ({})",
					real_mem_size, max_mem_size);
			}
		}

		~DefaultCPUBackend() noexcept override = default;

	public:
		void * alloc_memory(size_t size) override {
			return new uint8_t[size];
		}

		void dealloc_memory(void *ptr, size_t size) override {
			delete[] static_cast<uint8_t *>(ptr);
		}

	public:
		/*!
		 * @brief Submit `concurrency` tasks into task queue
		 * @param func task function
		 * @param concurrency The number of task in batch for parallel execution
		 * @return non-sense value
		 */
		int submit(std::function<void (int)> &&func, int concurrency) override {
			return thread_pool_.submit(std::forward<decltype(func)>(func), concurrency);
		}

		/*!
		 * @brief Return whether all worker threads have finished tasks in the task queue.
		 * @note This backend DO NOT support synchronization in fine granularity
		 */
		bool poll([[maybe_unused]] int task_token) override { return thread_pool_.poll(); }

		/*!
		 * @brief Waiting for all worker threads finishing tasks in the task queue
		 * @note This backend DO NOT support synchronization in fine granularity
		 */
		void sync([[maybe_unused]] int task_token) override { thread_pool_.sync(); }

	public:
		/*!
		 * @brief Return the maximum concurrency allowed for tasks
		 * @note This backend is integrated with thread pool with task queue, which allows any number of concurrent tasks .
		 * Therefore, the max concurrency is equal to the avail concurrency
		 * @return The maximum concurrency
		 */
		size_t get_max_concurrency() const override { return thread_pool_.max_concurrency(); }

		/*!
		 * @brief Return the maximum concurrency allowed for tasks
		 * @note This backend is integrated with thread pool with task queue, which allows any number of concurrent tasks .
		 * Therefore, the max concurrency is equal to the avail concurrency
		 * @return The available concurrency
		 */
		size_t get_avail_concurrency() const override { return thread_pool_.max_concurrency(); }
	};

} // namespace spy