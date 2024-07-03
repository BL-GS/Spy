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
#include "operator/type.h"

namespace spy {

	struct OperatorNode;

	enum class OperatorStatus {
		/// This operator is finished successfully
		Success,
		/// This operator failed to be finished
		Fail,
		/// This operator is unsupported on this backend
		Unsupport
	};

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

		virtual bool is_support(OperatorType op_type) const = 0;

		/*!
		 * @brief Submit `concurrency` tasks to the backend.
		 * @param op_node The operator node to be executed
		 * @param callback The callback function after executing the `op_node`
		 * @note The task function SHOULD NOT use blocking algorithm if the backend support concurrency overcommitment..
		 */
		virtual void submit(OperatorNode *op_node_ptr, std::function<void()> &&callback = nullptr)	= 0;
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

	namespace cpu { void init_backend(BackendFactory &); }

#ifdef SPY_BACKEND_CUDA
	namespace gpu { void init_backend(BackendFactory &); }
#endif

	inline BackendFactory::BackendFactory() {
		cpu::init_backend(*this);
#ifdef SPY_BACKEND_CUDA
		gpu::init_backend(*this);
#endif
	}

}  // namespace spy

SPY_ENUM_FORMATTER(spy::OperatorStatus);