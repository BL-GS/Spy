#include "operator/detail/matmul.h"
#include "abstract_backend.h"
#include "operator_impl.h"

namespace spy::gpu {

    OperatorStatus OperatorMatMulImpl::execute(GPUBackend *backend_ptr, const OperatorEnvParam &param) {
        const OperatorNode *op_node = param.node_ptr;
        const auto &result    = op_node->get_output<DataNode>(0).tensor;
		const auto &operand_0 = op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1 = op_node->get_input<DataNode>(1).tensor;

        cuda_op_matmul(backend_ptr->metadata_, result, operand_0, operand_1);
        return OperatorStatus::Success;
    }

} // namespace spy::gpu