#pragma once

#include "backend/config.h"
#include "gpu_device.h"

namespace spy::gpu {

	void print_cuda_devices();

	class GPUBackend: public AbstractBackend {
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

	public:
		void sync(int task_token) override;

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

		size_t get_task_num(const OperatorNode *node_ptr) const override;

		size_t get_buffer_size(const OperatorNode *node_ptr) const override;

		OperatorStatus execute(const OperatorEnvParam &param, OperatorNode *node_ptr) override;
	};

}  // namespace spy