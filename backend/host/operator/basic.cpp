#include "number/tensor.h"
#include "graph/graph.h"
#include "operator_impl.h"

namespace spy::cpu {

	inline static std::shared_ptr<ControlHeader> binary_control_header(const OperatorNode *op_node) {
		const auto &operand_0	= op_node->input_data(0)->tensor;
		const auto &shape_0  = operand_0.get_shape();

		const int num_task = shape_0.num_row();
		return std::make_shared<ControlHeader>(num_task);
	}

	template<class T_OpFunc>
	inline static OperatorResult binary_execution(const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand_0 = op_node->input_data(0)->tensor;
		const auto &operand_1 = op_node->input_data(1)->tensor;
		const auto &result    = op_node->output_data(0)->tensor;

		const auto &shape_0     = operand_0.get_shape();
		const auto &shape_1     = operand_1.get_shape();

		const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
		const auto [ne10, ne11, ne12, ne13] = shape_1.elements;

		const int64_t num_row = ne03 * ne02 * ne01;

		for (int64_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
			const int64_t i03 = row_idx / (ne01 * ne02);
			const int64_t i02 = (row_idx - i03 * ne03) / ne01;
			const int64_t i01 = row_idx - i03 * ne03 - i02 * ne02;

			const int64_t i13 = i03 % ne13;
			const int64_t i12 = i02 % ne12;
			const int64_t i11 = i01 % ne11;

			for (int64_t row_repeat_idx = 0; row_repeat_idx < ne00 / ne10; ++row_repeat_idx) {
				const int64_t i00 = row_repeat_idx * ne00;
				const float *src0_row_ptr = operand_0.get<float>({i00, i01, i02, i03});
				const float *src1_row_ptr = operand_1.get<float>({0, i11, i12, i13});
				float *		 dst_row_ptr  = result.get<float>({i00, i01, i02, i03});
				std::transform(src0_row_ptr, src0_row_ptr + ne10, src1_row_ptr, 
					dst_row_ptr, T_OpFunc()
				);
			}
		}

		return { 0_op_end };
	}

	std::shared_ptr<ControlHeader> OperatorAddImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const DerivedOperatorNode *op_node) {
		return binary_control_header(op_node);
	}

	std::shared_ptr<ControlHeader> OperatorSubImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const DerivedOperatorNode *op_node) {
		return binary_control_header(op_node);
	}

	std::shared_ptr<ControlHeader> OperatorMulImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const DerivedOperatorNode *op_node) {
		return binary_control_header(op_node);
	}

	std::shared_ptr<ControlHeader> OperatorDivImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const DerivedOperatorNode *op_node) {
		return binary_control_header(op_node);
	}

	OperatorResult OperatorAddImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, DerivedOperatorNode *op_node) {
		return binary_execution<std::plus<float>>(param, op_node);
	}

	OperatorResult OperatorSubImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, DerivedOperatorNode *op_node) {
		return binary_execution<std::minus<float>>(param, op_node);
	}

	OperatorResult OperatorMulImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, DerivedOperatorNode *op_node) {
		return binary_execution<std::multiplies<float>>(param, op_node);
	}

	OperatorResult OperatorDivImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, DerivedOperatorNode *op_node) {
		return binary_execution<std::divides<float>>(param, op_node);
	}

}  // namespace spy::cpu