#include "operator/detail/binary_op.h"
#include "abstract_backend.h"
#include "operator_impl.h"

namespace spy::gpu {

    OperatorStatus OperatorAddImpl::execute(GPUBackend *backend_ptr, const OperatorEnvParam &param) {
        const OperatorNode *op_node = param.node_ptr;
        const auto &result    = op_node->output(0).tensor;
		const auto &operand_0 = op_node->input(0).tensor;
		const auto &operand_1 = op_node->input(1).tensor;

        cuda_op_add(backend_ptr->metadata_, result, operand_0, operand_1);
        return OperatorStatus::Success;
    }

    OperatorStatus OperatorSubImpl::execute(GPUBackend *backend_ptr, const OperatorEnvParam &param) {
        const OperatorNode *op_node = param.node_ptr;
        const auto &result    = op_node->output(0).tensor;
		const auto &operand_0 = op_node->input(0).tensor;
		const auto &operand_1 = op_node->input(1).tensor;

        cuda_op_sub(backend_ptr->metadata_, result, operand_0, operand_1);
        return OperatorStatus::Success;
    }

    OperatorStatus OperatorMulImpl::execute(GPUBackend *backend_ptr, const OperatorEnvParam &param) {
        const OperatorNode *op_node = param.node_ptr;
        const auto &result    = op_node->output(0).tensor;
		const auto &operand_0 = op_node->input(0).tensor;
		const auto &operand_1 = op_node->input(1).tensor;

        cuda_op_add(backend_ptr->metadata_, result, operand_0, operand_1);
        return OperatorStatus::Success;
    }


    OperatorStatus OperatorDivImpl::execute(GPUBackend *backend_ptr, const OperatorEnvParam &param) {
        const OperatorNode *op_node = param.node_ptr;
        const auto &result    = op_node->output(0).tensor;
		const auto &operand_0 = op_node->input(0).tensor;
		const auto &operand_1 = op_node->input(1).tensor;

        cuda_op_add(backend_ptr->metadata_, result, operand_0, operand_1);
        return OperatorStatus::Success;
    }

} // namespace spy::gpu