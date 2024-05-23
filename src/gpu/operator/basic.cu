#include "operator/detail/binary_op.h"
#include "abstract_backend.h"
#include "operator_impl.h"

namespace spy::gpu {

    size_t OperatorAddImpl::get_task_num(const GPUBackend *backend_ptr, const OperatorNode *op_node) { return 1; }

    size_t OperatorAddImpl::get_buffer_size(const GPUBackend *backend_ptr, const OperatorNode *op_node) { return 0; }

    bool OperatorAddImpl::execute(GPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
        const auto &result    = op_node->get_output<DataNode>(0).tensor;
		const auto &operand_0 = op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1 = op_node->get_input<DataNode>(1).tensor;

        cuda_op_add(backend_ptr->metadata_, result, operand_0, operand_1);
        return true;
    }

    size_t OperatorSubImpl::get_task_num(const GPUBackend *backend_ptr, const OperatorNode *op_node) { return 1; }

    size_t OperatorSubImpl::get_buffer_size(const GPUBackend *backend_ptr, const OperatorNode *op_node) { return 0; }

    bool OperatorSubImpl::execute(GPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
        const auto &result    = op_node->get_output<DataNode>(0).tensor;
		const auto &operand_0 = op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1 = op_node->get_input<DataNode>(1).tensor;

        cuda_op_sub(backend_ptr->metadata_, result, operand_0, operand_1);
        return true;
    }

    size_t OperatorMulImpl::get_task_num(const GPUBackend *backend_ptr, const OperatorNode *op_node) { return 1; }

    size_t OperatorMulImpl::get_buffer_size(const GPUBackend *backend_ptr, const OperatorNode *op_node) { return 0; }

    bool OperatorMulImpl::execute(GPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
        const auto &result    = op_node->get_output<DataNode>(0).tensor;
		const auto &operand_0 = op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1 = op_node->get_input<DataNode>(1).tensor;

        cuda_op_add(backend_ptr->metadata_, result, operand_0, operand_1);
        return true;
    }

    size_t OperatorDivImpl::get_task_num(const GPUBackend *backend_ptr, const OperatorNode *op_node) { return 1; }

    size_t OperatorDivImpl::get_buffer_size(const GPUBackend *backend_ptr, const OperatorNode *op_node) { return 0; }

    bool OperatorDivImpl::execute(GPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
        const auto &result    = op_node->get_output<DataNode>(0).tensor;
		const auto &operand_0 = op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1 = op_node->get_input<DataNode>(1).tensor;

        cuda_op_add(backend_ptr->metadata_, result, operand_0, operand_1);
        return true;
    }

} // namespace spy::gpu