#include "operator/detail/matmul.h"
#include "backend/gpu/gpu.h"
#include "gpu_device.h"
#include "backend/gpu/operator_impl.h"

namespace spy::gpu {

    size_t OperatorMatMulImpl::get_task_num(const GPUBackend *backend_ptr, const OperatorNode *op_node) { return 1; }

    size_t OperatorMatMulImpl::get_buffer_size(const GPUBackend *backend_ptr, const OperatorNode *op_node) { return 1; }

    bool OperatorMatMulImpl::execute(GPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
        DeviceContext &context = *static_cast<DeviceContext *>(backend_ptr->metadata_ptr_);
        const auto &result    = op_node->get_output<DataNode>(0).tensor;
		const auto &operand_0 = op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1 = op_node->get_input<DataNode>(1).tensor;

        cuda_op_matmul(context, result, operand_0, operand_1);
        return true;
    }

} // namespace spy::gpu