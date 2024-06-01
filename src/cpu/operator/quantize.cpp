
#include <magic_enum.hpp>

#include "number/tensor.h"
#include "number/quantization.h"
#include "graph/graph.h"
#include "operator/config.h"
#include "operator_impl.h"

namespace spy::cpu {

	std::shared_ptr<ControlHeader> OperatorQuantizeImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        const auto &operand  = op_node->get_input<DataNode>(0).tensor;
        const auto &shape_operand = operand.get_shape();
        const int num_task = shape_operand.num_row();
        return std::make_shared<ControlHeader>(num_task);
    }

	OperatorResult OperatorQuantizeImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand = op_node->get_input<DataNode>(0).tensor;
		const auto &result  = op_node->get_output<DataNode>(0).tensor;

        const NumberType from_type = operand.get_number_type();
        const NumberType to_type   = result.get_number_type();

        const Shape &shape_operand = operand.get_shape();
        auto [ne00, ne01, ne02, ne03] = shape_operand.elements;
        const size_t num_row  = shape_operand.num_row();

        for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
            const size_t i03 = row_idx / (ne01 * ne02);
            const size_t i02 = row_idx % (ne01 * ne02) / ne01;
            const size_t i01 = row_idx % ne01;

            const void *src_ptr = operand.get({0, i01, i02, i03});
            void *dst_ptr       = result.get({0, i01, i02, i03});
	        auto_quantize_inner(from_type, src_ptr, to_type, dst_ptr, ne00 / get_block_size(from_type));
        }

        return { 0_op_end };
    }

}  // namespace spy::cpu