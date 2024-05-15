/*
 * @author: BL-GS 
 * @date:   24-4-13
 */

#pragma once

#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <functional>

#include "backend/gpu/type.h"
#include "backend/gpu/operator_impl.h"

namespace spy {

	class DefaultGPUBackend final: public GPUBackend {
	private:
		std::mutex 									 task_lock_;

		std::condition_variable						 task_cond_;

		std::queue<std::function<void (int)>> task_list_;

		std::thread 	 worker_thread_;

		std::stop_source stop_source_;

	public:
		DefaultGPUBackend(int device_id): GPUBackend(device_id), worker_thread_([this](){ thread_entry(); }) { }

		~DefaultGPUBackend() noexcept {
			stop_source_.request_stop();
			task_cond_.notify_all();
			worker_thread_.join();
		}

	public:
		void * alloc_memory(size_t size) override;

		void dealloc_memory(void *ptr, size_t size) override;

	public:
		/*!
		 * @brief Submit `concurrency` tasks into task queue
		 * @param func task function
		 * @param concurrency The number of task in batch for parallel execution
		 * @return non-sense value
		 */
		int submit(std::function<void (int)> &&func, int concurrency) override {
			std::unique_lock<std::mutex> lock(task_lock_);
			task_list_.push(std::forward<std::function<void (int)>>(func));
			lock.unlock();
			task_cond_.notify_one();
			return 0;
		}

		/*!
		 * @brief Return whether all worker threads have finished tasks in the task queue.
		 * @note This backend DO NOT support synchronization in fine granularity
		 */
		bool poll(int task_token) override {
			sync(task_token);
			return true;
		}

		/*!
		 * @brief Waiting for all worker threads finishing tasks in the task queue
		 * @note This backend DO NOT support synchronization in fine granularity
		 */
		void sync(int task_token) override {
			while (true) {
				std::lock_guard<std::mutex> lock(task_lock_);
				if (!task_list_.empty()) { 
					GPUBackend::sync(task_token);
					return; 
				}
				std::this_thread::yield();
			}
		}

	private:
		void thread_entry() {
			while (!stop_source_.stop_requested()) [[likely]] {
				std::unique_lock<std::mutex> lock(task_lock_);
				task_cond_.wait(lock);

				if (!task_list_.empty()) {
					auto new_task = std::move(task_list_.front());
					task_list_.pop();
					
					new_task(0);
				}
			}
		}

	};

} // namespace spy