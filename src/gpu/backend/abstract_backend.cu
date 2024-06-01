/*
 * @author: BL-GS 
 * @date:   24-4-25
 */


#include <cuda.h>
#include <cuda_runtime.h>

#include "operator/type.h"
#include "graph/graph.h"
#include "gpu_device.h"
#include "operator_impl.h"
#include "abstract_backend.h"

namespace spy::gpu {

	GPUBackend::GPUBackend(int device_id): AbstractBackend(BackendType::Device), metadata_(device_id) {}

	size_t GPUBackend::get_max_memory_capacity() const {
		return metadata_.get_max_memory_capacity();
	}

	size_t GPUBackend::get_avail_memory_capacity() const {
		return metadata_.get_avail_memory_capacity();
	}

	template<OperatorType T_op_type>
	struct ExecuteExtractor {
		constexpr auto operator()() const { return gpu::OperatorImpl<T_op_type>::execute; }
	};

	OperatorStatus GPUBackend::execute(const OperatorEnvParam &param) {
		const OperatorNode *node_ptr = param.node_ptr;
		const OperatorType op_type = node_ptr->op_type;
		const auto func = get_execute_func(op_type);
		return func(this, param);
	}

	GPUBackend::TaskFunc GPUBackend::get_execute_func(OperatorType op_type) const {
		return operator_type_switch<ExecuteExtractor>(op_type);
	}

} // namespace spy
