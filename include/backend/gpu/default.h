/*
 * @author: BL-GS 
 * @date:   24-4-13
 */

#pragma once

#include <functional>

#include "backend/gpu/type.h"
#include "backend/gpu/operator_impl.h"

namespace spy {

	class DefaultGPUBackend;

	class DefaultGPUBackend final: public GPUBackend {
	private:
		void *metadata_ptr_;

	public:
		DefaultGPUBackend(int num_thread, int64_t max_mem);

		~DefaultGPUBackend() noexcept override;

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
		int submit(std::function<void (int)> &&func, int concurrency) override;

		/*!
		 * @brief Return whether all worker threads have finished tasks in the task queue.
		 * @note This backend DO NOT support synchronization in fine granularity
		 */
		bool poll([[maybe_unused]] int task_token) override;

		/*!
		 * @brief Waiting for all worker threads finishing tasks in the task queue
		 * @note This backend DO NOT support synchronization in fine granularity
		 */
		void sync([[maybe_unused]] int task_token) override;

	public:
		/*!
		 * @brief Return the maximum concurrency allowed for tasks
		 * @note This backend is integrated with thread pool with task queue, which allows any number of concurrent tasks .
		 * Therefore, the max concurrency is equal to the avail concurrency
		 * @return The maximum concurrency
		 */
		size_t get_max_concurrency() const override;

		/*!
		 * @brief Return the maximum concurrency allowed for tasks
		 * @note This backend is integrated with thread pool with task queue, which allows any number of concurrent tasks .
		 * Therefore, the max concurrency is equal to the avail concurrency
		 * @return The available concurrency
		 */
		size_t get_avail_concurrency() const override;
	};

} // namespace spy