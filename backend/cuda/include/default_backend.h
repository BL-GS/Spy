/*
 * @author: BL-GS 
 * @date:   24-4-13
 */

#pragma once

#include <charconv>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <stop_token>
#include <thread>
#include <functional>

#include "cuda_backend.h"
#include "gpu_device.h"
#include "task.h"

namespace spy::gpu {

	class DefaultGPUBackend final: public GPUBackend {
	public:
		static constexpr int DEFAULT_DEVICE_ID = 0;

	private:
		std::mutex 									 task_lock_;

		std::condition_variable						 task_cond_;

		std::queue<OperatorEnvParam> 				 task_list_;

		std::thread 	 worker_thread_;

		std::stop_source stop_source_;

	public:
		DefaultGPUBackend(const BackendFactory::BackendConfiguration &config): 
			GPUBackend(config.parse_or("device_id", DEFAULT_DEVICE_ID)), 
			worker_thread_([this](){ thread_entry(); }) { }

		DefaultGPUBackend(int device_id): 
			GPUBackend(device_id), worker_thread_([this](){ thread_entry(); }) { }

		~DefaultGPUBackend() noexcept {
			stop_source_.request_stop();
			task_cond_.notify_all();
			worker_thread_.join();
		}

	public:
		void *alloc_memory(size_t size) override {
			auto *pool_ptr = metadata_.get_memory_pool();
			return pool_ptr->allocate(size);
		}

		void dealloc_memory(void *ptr, size_t size) override {
			auto *pool_ptr = metadata_.get_memory_pool();
			pool_ptr->deallocate(ptr, size);
		}

	public:
		/*!
		 * @brief Submit `concurrent` tasks into task queue
		 * @param op_node_ptr The pointer of the submitted OperatorNode
		 * @param callback The callback function 
		 */
		void submit(OperatorNode *op_node_ptr, std::function<void()> &&callback = nullptr) override {
			std::unique_lock<std::mutex> lock(task_lock_);

			OperatorEnvParam param{
				.stream   = metadata_.get_stream(0),  
				.node_ptr = op_node_ptr,
				.func	  = get_execute_func(op_node_ptr->op_type),
				.callback = callback
			};

			task_list_.push(param);
			lock.unlock();
			task_cond_.notify_one();
		}

	private:
		void thread_entry() {
			while (!stop_source_.stop_requested()) [[likely]] {
				std::unique_lock<std::mutex> lock(task_lock_);
				task_cond_.wait(lock);

				if (!task_list_.empty()) {
					const auto new_task = std::move(task_list_.front());
					task_list_.pop();

					new_task.func(this, new_task);
					new_task.callback();
				}
			}
		}

	};

} // namespace spy