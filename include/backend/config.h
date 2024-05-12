/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstddef>
#include <memory>
#include <functional>

#include "util/shell/logger.h"
#include "backend/type.h"

namespace spy {

	struct OperatorNode;
	struct OperatorEnvParam;

	class AbstractBackend {
	public:
		AbstractBackend() = default;

		virtual ~AbstractBackend() = default;

	public:
		/*!
		 * @brief Get the maximum size of memory that can be allocated from this backend
		 */
		virtual size_t get_max_memory_capacity() 	const = 0;

		/*!
		 * @brief Get the size of available memory
		 */
		virtual size_t get_avail_memory_capacity() 	const = 0;

	public: /* Data Management */
		/*!
		 * @brief Allocate memory from this backend.
		 * @param size The expect size of memory
		 * @note The real size may be larger than size but never smaller than `size`
		 */
		virtual void *alloc_memory(size_t size) 			 = 0;

		/*!
		 * @brief Deallocate memory that allocated from this backend.
		 * @param ptr The start pointer of allocated memory
		 * @param size The size of allocated memory
		 * @note The behaviour of deallocating memory from other backend is undefined.
		 */
		virtual void  dealloc_memory(void *ptr, size_t size) = 0;

	public: /* Processor Operation */
		/*!
		 * @brief Get the maximum concurrency of the backend.
		 * @note If the backend cannot receive concurrent requests, it should return 1;
		 */
		virtual size_t get_max_concurrency() 	const = 0;

		/*!
		 * @brief Get the available concurrency of the backend.
		 * @note The number of the given task SHOULD NOT be larger than the available concurrency.
		 * @note If the backend cannot buffer requests, it may incurs failure or blocking.
		 */
		virtual size_t get_avail_concurrency() 	const = 0;

	public: /* Schedule */
		/*!
		 * @brief Submit `concurrency` tasks to the backend.
		 * @param task The function of task
		 * @param concurrency The number of task to be executed concurrently
		 * @note The task function SHOULD NOT use blocking algorithm if the backend support concurrency overcommitment..
		 * @return The credit of the task, which can be used to execute `poll` and `sync`
		 */
		virtual int    submit(std::function<void(int)> &&task, int concurrency)	= 0;

		/*!
		 * @brief Query whether the backend supports `poll` call
		 */
		virtual bool   support_poll() const { return false; }

		/*!
		 * @brief Query whether the backend supports `sync` call
		 */
		virtual bool   support_sync() const { return false; }

		/*!
		 * @brief Query whether the task is finished if supported
		 * @param task_credit The credit of the specific task
		 */
		virtual bool   poll(int task_credit)	{
			if (support_poll()) {
				throw SpyUnimplementedException("Unimplemented `poll` call");
			} else {
				throw SpyUnimplementedException("This backend do not support `poll` function");
			}
		}

		/*!
		 * @brief Synchronize with a specific task
		 * @param task_credit The credit of the specific task
		 */
		virtual void   sync(int task_credit) {
			if (support_sync()) {
				throw SpyUnimplementedException("Unimplemented `sync` call");
			} else {
				throw SpyUnimplementedException("This backend do not support explicit synchronization");
			}
		}

	public:
		/*!
		 * @brief Get the number of task that can be executed in parallelism.
		 * Most of the time, the number will be 1 if not to be executed on accelerator
		 */
		virtual size_t get_task_num(const OperatorNode *op_node) const = 0;

		/*!
		 * @brief Get the number of task that can be executed in parallelism.
		 * Most of the time, the number will be 1 if not to be executed on accelerator
		 */
		virtual size_t get_buffer_size(const OperatorNode *op_node) const = 0;

		/*!
		 * @brief Execute the operator
		 * @param param The parameter of environment. Specifically, it denote the concurrency and thread id of CPU backend.
		 * @param op_node The OperatorNode deriving from OperatorDefinition<T_op_type>, which store necessary operands and hyper parameters.
		 * @return true if supported, otherwise false.
		 */
		virtual bool execute(const OperatorEnvParam &param, OperatorNode *op_node) = 0;
	};

	template<BackendType T_backend>
	class Backend { };


}  // namespace spy