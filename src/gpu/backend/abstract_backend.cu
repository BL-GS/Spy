/*
 * @author: BL-GS 
 * @date:   24-4-25
 */


#include <cuda.h>
#include <cuda_runtime.h>

#include "operator/type.h"
#include "gpu_device.h"
#include "operator_impl.h"
#include "abstract_backend.h"

namespace spy::gpu {

	GPUBackend::GPUBackend(int device_id): metadata_(device_id) {}

	size_t GPUBackend::get_max_memory_capacity() const {
		return metadata_.get_max_memory_capacity();
	}

	size_t GPUBackend::get_avail_memory_capacity() const {
		return metadata_.get_avail_memory_capacity();
	}

	void GPUBackend::sync(int task_token) {
		if (task_token >= 0 && task_token < DeviceContext::MAX_NUM_STREAM) { 
			gpu::gpu_check(cudaStreamSynchronize(metadata_.get_stream(task_token)));
		} else {
			gpu::gpu_check(cudaDeviceSynchronize());
		}
	}

	template<OperatorType T_op_type>
	struct TaskNumExtractor {
		constexpr auto operator()() const { return gpu::OperatorImpl<T_op_type>::get_task_num; }
	};

	size_t GPUBackend::get_task_num(const OperatorNode *node_ptr) const {
		const OperatorType op_type = node_ptr->op_type;
		const auto func = operator_type_switch<TaskNumExtractor>(op_type);
		return func(this, node_ptr);
	}

	template<OperatorType T_op_type>
	struct GetBufferSizeExtractor {
		constexpr auto operator()() const { return gpu::OperatorImpl<T_op_type>::get_buffer_size; }
	};

	size_t GPUBackend::get_buffer_size(const OperatorNode *node_ptr) const {
		const OperatorType op_type = node_ptr->op_type;
		const auto func = operator_type_switch<GetBufferSizeExtractor>(op_type);
		return func(this, node_ptr);
	}

	template<OperatorType T_op_type>
	struct ExecuteExtractor {
		constexpr auto operator()() const { return gpu::OperatorImpl<T_op_type>::execute; }
	};

	OperatorStatus GPUBackend::execute(const OperatorEnvParam &param, OperatorNode *node_ptr) {
		const OperatorType op_type = node_ptr->op_type;
		const auto func = operator_type_switch<ExecuteExtractor>(op_type);
		return func(this, param, node_ptr);
	}

} // namespace spy
