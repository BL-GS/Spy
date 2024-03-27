/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstddef>
#include <memory>
#include <functional>

#include "util/logger.h"
#include "backend/type.h"

namespace spy {

	struct OperatorNode;
	struct OperatorEnvParam;

	class AbstractBackend {
	public:
		virtual ~AbstractBackend() = default;

	public:
		virtual size_t get_max_memory_capacity() 	const = 0;
		virtual size_t get_avail_memory_capacity() 	const = 0;

	public: /* Data Management */
		virtual void *alloc_memory(size_t size) 			 = 0;
		virtual void  dealloc_memory(void *ptr, size_t size) = 0;

	public: /* Processor Operation */
		virtual size_t get_max_concurrency() 	const = 0;
		virtual size_t get_avail_concurrency() 	const = 0;

	public: /* Schedule */
		virtual int    submit(std::function<void(int)> &&task, int concurrency)	= 0;
		virtual bool   poll(int task_token)	{
			throw SpyUnimplementedException("This backend do not support `poll` function");
		}
		virtual void   sync(int task_token) {
			throw SpyUnimplementedException("This backend do not support explicit synchronization");
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
		 * @param op_node The OperatorNode deriving from OperatorDinition<T_op_type>, which store necessary operands and hyper parameters.
		 * @return true if supported, otherwise false.
		 */
		virtual bool execute(const OperatorEnvParam &param, OperatorNode *op_node) = 0;
	};

	template<BackendType T_backend>
	class Backend { };


}  // namespace spy