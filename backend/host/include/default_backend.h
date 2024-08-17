/*
 * @author: BL-GS 
 * @date:   24-4-13
 */

#pragma once

#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#ifdef UNOFFICIAL_CONCURRENTQUEUE
    #include <concurrentqueue/concurrentqueue.h>
#else 
    #include <concurrentqueue.h>
#endif

#include "util/log/logger.h"
#include "task.h"
#include "cpu_backend.h"
#include "graph/op_node.h"
#include "operator_impl.h"
#include "perf/event.h"

namespace spy::cpu {

	class DefaultCPUBackend;

	class DefaultThreadPool {
		friend class DefaultCPUBackend;
	public:
		using TaskFunc  = OperatorResult (*)(CPUBackend *, const OperatorEnvParam &, OperatorNode *);
		using TaskInfo  = OperatorEnvParam;
		using TaskQueue = moodycamel::ConcurrentQueue<TaskInfo>;

	private:
		/// Worker threads
		std::vector<std::thread>    	workers_;
		/// The task queue buffering all tasks
		TaskQueue                       task_queue_;
		/// Stop all threads if true
		std::atomic_flag        		stop_flag_;
		/// Pointer to the backend
		CPUBackend *					backend_ptr_;

	public:
		DefaultThreadPool(CPUBackend * backend_ptr): backend_ptr_(backend_ptr) {}

		~DefaultThreadPool() noexcept {
			stop_flag_.test_and_set();
			// Push nullptr to notifying all threads
			for (size_t i = 0; i < workers_.size(); ++i) { 
				task_queue_.enqueue({}); 
			}
			// Wait for all threads
			for (auto &worker: workers_) { worker.join(); }
		}

	protected:
		void reserve(uint32_t num_thread) {
			workers_.reserve(num_thread);
			for (size_t i = 0; i < num_thread; ++i) {
				workers_.emplace_back([this](){
					while (!stop_flag_.test()) {
						// Sleep if no task
						TaskInfo task_info;
						// Try to fetch a task
						while (!task_queue_.try_dequeue(task_info)) { std::this_thread::yield(); }
						if (!task_info.func) { continue; }

						// Execute task
						const OperatorResult result = task_info.func(backend_ptr_, task_info, task_info.node_ptr);

						using magic_enum::enum_name;
						spy_assert(result.phase == OperatorPhaseType::End,
							"Cannot finish executing operator {} at phase {}", task_info.node_ptr->op_type, result.phase);
						spy_assert(result.status == OperatorStatus::Success,
						    "Failed execute operator: {}", result.status);
					}
				});
			}
		}

	public:
		void submit(TaskInfo &&task_info) {
			const auto &header_ptr = task_info.header_ptr;
			const int   num_task   = header_ptr->num_task;
			if (num_task == 1) {
				task_info.concurrency = 1;
				task_info.tid 		  = 0;
				header_ptr->init(task_info);
				task_queue_.enqueue(std::forward<TaskInfo>(task_info));
			} else {
				const int concurrency = task_info.concurrency = std::min<int>(max_concurrency(), num_task);
				header_ptr->init(task_info);
				std::vector<TaskInfo> func_bulk;
				func_bulk.reserve(concurrency);
				for (int i = 0; i < concurrency; ++i) { 
					func_bulk.emplace_back(task_info.fork(i)); 
				}
				task_queue_.enqueue_bulk(func_bulk.begin(), concurrency);
			}
		}

	public:
		bool poll() { return task_queue_.size_approx() == 0; }

		void sync()  { while (task_queue_.size_approx() != 0) { std::this_thread::yield(); } }

		size_t max_concurrency() const { return workers_.size(); }
	};

	class DefaultCPUBackend final: public CPUBackend {
	public:
		static constexpr int DEFAULT_NUM_THREAD = -1;
		static constexpr int DEFAULT_MAX_MEM    = -1;
	private:
		DefaultThreadPool thread_pool_;

	public:
		DefaultCPUBackend(const BackendFactory::BackendConfiguration &config): thread_pool_(this) {
			int num_thread = config.parse_or("num_thread", DEFAULT_NUM_THREAD);
			int max_mem	   = config.parse_or("max_mem", DEFAULT_MAX_MEM);
			init(num_thread, max_mem);
		}

		DefaultCPUBackend(int num_thread, int64_t max_mem): thread_pool_(this) { init(num_thread, max_mem); }

		~DefaultCPUBackend() noexcept override = default;

	private:
		void init(int num_thread, int64_t max_mem) {
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

	public:
		void * alloc_memory(size_t size) override {
			return new uint8_t[size];
		}

		void dealloc_memory(void *ptr, [[maybe_unused]]size_t size) override {
			delete[] static_cast<uint8_t *>(ptr);
		}

	public:
		/*!
		 * @brief Submit `concurrency` tasks into task queue
		 * @param op_node The operator node to be executed
		 * @param callback The callback function after executing the `op_node`
		 */
		void submit(OperatorNode *op_node_ptr, std::function<void()> &&callback) override {
			using TaskInfo = DefaultThreadPool::TaskInfo;

			const OperatorType op_type = op_node_ptr->op_type;
			TaskInfo info {
				.func 		 = get_execute_func(op_type),
				.node_ptr	 = op_node_ptr,
				.header_ptr  = get_control_header(op_type, op_node_ptr)
			};

			if (info.header_ptr == nullptr) {
				const OperatorResult result = info.func(this, info, op_node_ptr);
				spy_assert(result.status == OperatorStatus::Success);
				callback();
			} else {
				info.header_ptr->callback = std::forward<std::function<void()>>(callback);
				thread_pool_.submit(std::move(info));
			}
		}

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