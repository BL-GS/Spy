/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "backend/config.h"
#include "backend/gpu/operator_impl.h"

namespace spy {

	class GPUBackend: public AbstractBackend {
	protected:
		/// Pointer to the metadata of the GPU device
		/// which is allocated and deallocated in CUDA file
		void *metadata_ptr_;

	public:
		GPUBackend(int device_id);

		~GPUBackend() noexcept override = default;

	public: /* Memory Operation */
		size_t get_max_memory_capacity() 	const override;
		size_t get_avail_memory_capacity() 	const override;

	public: /* Data Management */
		void *alloc_memory(size_t size) override;
		void  dealloc_memory(void *ptr, [[maybe_unused]]size_t size) 	override;

	public: /* Processor Operation */
		/*!
		 * @brief Return the max concurrency of GPU.
		 * For backend supporting stream-level schedule, it can be the maximum number of streams.
		 */
		size_t get_max_concurrency() 	const override { return 1; }

		/*!
		 * @brief Return the available concurrency of GPU.
		 * For backend supporting stream-level schedule, it can be the number of idle streams.
		 */
		size_t get_avail_concurrency() 	const override { return 1; }

	public:
		template<OperatorType T_op_type>
		struct TaskNumExtractor {
			constexpr auto operator()() const { return gpu::OperatorImpl<T_op_type>::get_task_num; }
		};

		size_t get_task_num(const OperatorNode *node_ptr) const override {
			const OperatorType op_type = node_ptr->op_type;
			const auto func = operator_type_switch<TaskNumExtractor>(op_type);
			return func(this, node_ptr);
		}

		template<OperatorType T_op_type>
		struct GetBufferSizeExtractor {
			constexpr auto operator()() const { return gpu::OperatorImpl<T_op_type>::get_buffer_size; }
		};

		size_t get_buffer_size(const OperatorNode *node_ptr) const override {
			const OperatorType op_type = node_ptr->op_type;
			const auto func = operator_type_switch<GetBufferSizeExtractor>(op_type);
			return func(this, node_ptr);
		}

		template<OperatorType T_op_type>
		struct ExecuteExtractor {
			constexpr auto operator()() const { return gpu::OperatorImpl<T_op_type>::execute; }
		};

		bool execute(const OperatorEnvParam &param, OperatorNode *node_ptr) override {
			const OperatorType op_type = node_ptr->op_type;
			const auto func = operator_type_switch<ExecuteExtractor>(op_type);
			return func(this, param, node_ptr);
		}
	};
}  // namespace spy
