/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstddef>
#include <memory>
#include <functional>
#include <unordered_map>

#include "util/shell/logger.h"
#include "util/wrapper/config_table.h"
#include "backend/type.h"

namespace spy {

	struct OperatorNode;
	struct OperatorEnvParam;

	enum class OperatorStatus {
		/// This operator is finished successfully
		Success,
		/// This operator failed to be finished
		Fail,
		/// This operator is unsupported on this backend
		Unsupport
	};

	using BackendTaskCredit = int;

	class AbstractBackend {
	private:
		BackendType backend_type_;

	public:
		AbstractBackend(BackendType backend_type): backend_type_(backend_type) {}

		virtual ~AbstractBackend() = default;

	public:
		BackendType type() const { return backend_type_; }

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
		virtual BackendTaskCredit submit(std::function<void(int)> &&task, int concurrency)	= 0;

		/*!
		 * @brief Query whether the task is finished if supported
		 * @param task_credit The credit of the specific task
		 */
		virtual bool poll(BackendTaskCredit task_credit)	{
			throw SpyUnimplementedException("This backend do not support `poll` function");
		}

		/*!
		 * @brief Synchronize with a specific task
		 * @param task_credit The credit of the specific task
		 */
		virtual void sync(BackendTaskCredit task_credit) {
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
		 * @param op_node The OperatorNode deriving from OperatorDefinition<T_op_type>, which store necessary operands and hyper parameters.
		 * @return true if supported, otherwise false.
		 */
		virtual OperatorStatus execute(const OperatorEnvParam &param, OperatorNode *op_node) = 0;
	};

	class BackendFactory {
	public:
		using BackendConfiguration = ConfigTable;
		using BackendGeneratorFunc = std::unique_ptr<AbstractBackend>(*)(const BackendConfiguration &);

	private:
		std::unordered_map<std::string, BackendGeneratorFunc> generator_map_;
		
	public:
		BackendFactory();

	public:
		std::unique_ptr<AbstractBackend> init_backend(const std::string &backend_name, const BackendConfiguration &config) {
			const auto iter = generator_map_.find(backend_name);
			if (iter == generator_map_.end()) {
				spy_fatal("Failed to find backend with name: {}", backend_name);
			}

			const BackendGeneratorFunc generator_func = iter->second;
			return generator_func(config);
		}

		void add_backend_map(const std::string &backend_name, const BackendGeneratorFunc func) {
			const auto [iter, no_overwrite] = generator_map_.insert_or_assign(backend_name, func);
			if (!no_overwrite) {
				spy_warn("Overwrite backend: {}", backend_name);
			}
		}

	public: /* Rule */
		static std::string make_backend_name(const std::string_view device_type, const std::string_view policy_name) {
			return std::string(device_type) + ":" + std::string(policy_name);
		}
	};

#ifdef SPY_BACKEND_CPU
	namespace cpu { void init_backend(BackendFactory &); }
#endif

#ifdef SPY_BACKEND_GPU
	namespace gpu { void init_backend(BackendFactory &); }
#endif

	inline BackendFactory::BackendFactory() {
#ifdef SPY_BACKEND_CPU
		cpu::init_backend(*this);
#endif

#ifdef SPY_BACKEND_GPU
		gpu::init_backend(*this);
#endif
	}

}  // namespace spy