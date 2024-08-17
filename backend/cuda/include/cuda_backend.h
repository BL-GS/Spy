#pragma once

#include "backend/backend.h"
#include "gpu_device.h"
#include "operator/type.h"
#include "task.h"

namespace spy::gpu {

	void print_cuda_devices();

	class GPUBackend: public Backend {
	public:
		using TaskFunc = OperatorStatus (*)(GPUBackend *, const OperatorEnvParam &);

	public:
		/// Pointer to the metadata of the GPU device
		/// which is allocated and deallocated in CUDA file
		DeviceContext metadata_;

	public:
		GPUBackend(int device_id);

		~GPUBackend() noexcept override = default;

	public: /* Memory Operation */
		size_t get_max_memory_capacity() 	const override;

		size_t get_avail_memory_capacity() 	const override;

	public: /* Data Management */
		void *alloc_memory(size_t size) 				override { return nullptr; }

		void  dealloc_memory(void *ptr, size_t size) 	override { };

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
		bool is_support(OperatorType op_type) const override;

		/*!
		 * @brief Execute the operator
		 * @param param The parameter of environment. Specifically, it denote the concurrency and thread id of CPU backend.
		 * @return true if supported, otherwise false.
		 */
		virtual OperatorStatus execute(const OperatorEnvParam &param);

	protected:
		TaskFunc get_execute_func(OperatorType op_type) const;
	};

}  // namespace spy