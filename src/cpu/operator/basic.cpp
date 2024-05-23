#include "number/tensor.h"
#include "graph/graph.h"
#include "operator/config.h"
#include "operator_impl.h"

namespace spy::cpu {

	size_t OperatorAddImpl::get_task_num([[maybe_unused]] const CPUBackend *backend_ptr, const OperatorNode *op_node) {
		const auto &operand_0	= op_node->get_input<DataNode>(0).tensor;
		const auto &shape_0  = operand_0.get_shape();
		return shape_0.num_row();
	}

	size_t OperatorAddImpl::get_buffer_size([[maybe_unused]] const CPUBackend *backend_ptr, [[maybe_unused]] const OperatorNode *op_node) {
		return 0;
	}

	bool OperatorAddImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand_0 = op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1 = op_node->get_input<DataNode>(1).tensor;
		const auto &result    = op_node->get_output<DataNode>(0).tensor;

		const auto &shape_0     = operand_0.get_shape();
		const auto &shape_1     = operand_1.get_shape();

		const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
		const auto [ne10, ne11, ne12, ne13] = shape_1.elements;

		const size_t num_row = ne03 * ne02 * ne01;

		for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
			const size_t i03 = row_idx / (ne01 * ne02);
			const size_t i02 = (row_idx - i03 * ne03) / ne01;
			const size_t i01 = row_idx - i03 * ne03 - i02 * ne02;

			const size_t i13 = i03 % ne13;
			const size_t i12 = i02 % ne12;
			const size_t i11 = i01 % ne11;

			for (size_t row_repeat_idx = 0; row_repeat_idx < ne00 / ne10; ++row_repeat_idx) {
				const size_t i00 = row_repeat_idx * ne00;
				const float *src0_row_ptr = operand_0.get<float>({i00, i01, i02, i03});
				const float *src1_row_ptr = operand_1.get<float>({0, i11, i12, i13});
				float *		 dst_row_ptr  = result.get<float>({i00, i01, i02, i03});
				std::transform(src0_row_ptr, src0_row_ptr + ne10, src1_row_ptr, 
					dst_row_ptr, std::plus<float>()
				);
			}
		}

		return true;
	}


	size_t OperatorSubImpl::get_task_num([[maybe_unused]] const CPUBackend *backend_ptr, const OperatorNode *op_node) {
		const auto &operand_0	= op_node->get_input<DataNode>(0).tensor;
		const auto &shape_0     = operand_0.get_shape();
		return shape_0.num_row();
	}

	size_t OperatorSubImpl::get_buffer_size([[maybe_unused]] const CPUBackend *backend_ptr, [[maybe_unused]] const OperatorNode *op_node) {
		return 0;
	}

	bool OperatorSubImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand_0 = op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1 = op_node->get_input<DataNode>(1).tensor;
		const auto &result    = op_node->get_output<DataNode>(0).tensor;

		const auto &shape_0     = operand_0.get_shape();
		const auto &shape_1     = operand_1.get_shape();

		const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
		const auto [ne10, ne11, ne12, ne13] = shape_1.elements;

		const size_t num_row = ne03 * ne02 * ne01;

		for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
			const size_t i03 = row_idx / (ne01 * ne02);
			const size_t i02 = (row_idx - i03 * ne03) / ne01;
			const size_t i01 = row_idx - i03 * ne03 - i02 * ne02;

			const size_t i13 = i03 % ne13;
			const size_t i12 = i02 % ne12;
			const size_t i11 = i01 % ne11;

			for (size_t row_repeat_idx = 0; row_repeat_idx < ne00 / ne10; ++row_repeat_idx) {
				const size_t i00 = row_repeat_idx * ne00;
				const float *src0_row_ptr = operand_0.get<float>({i00, i01, i02, i03});
				const float *src1_row_ptr = operand_1.get<float>({0, i11, i12, i13});
				float *		 dst_row_ptr  = result.get<float>({i00, i01, i02, i03});
				std::transform(src0_row_ptr, src0_row_ptr + ne10, src1_row_ptr, 
					dst_row_ptr, std::minus<float>()
				);
			}
		}

		return true;
	}

	size_t OperatorMulImpl::get_task_num([[maybe_unused]] const CPUBackend *backend_ptr, const OperatorNode *op_node) {
		const auto &operand_0	= op_node->get_input<DataNode>(0).tensor;
		const auto &shape_0     = operand_0.get_shape();
		return shape_0.num_row();
	}

	size_t OperatorMulImpl::get_buffer_size([[maybe_unused]] const CPUBackend *backend_ptr, [[maybe_unused]] const OperatorNode *op_node) {
		return 0;
	}

	bool OperatorMulImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand_0 = op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1 = op_node->get_input<DataNode>(1).tensor;
		const auto &result    = op_node->get_output<DataNode>(0).tensor;

		const auto &shape_0     = operand_0.get_shape();
		const auto &shape_1     = operand_1.get_shape();

		const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
		const auto [ne10, ne11, ne12, ne13] = shape_1.elements;

		const size_t num_row = ne03 * ne02 * ne01;

		for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
			const size_t i03 = row_idx / (ne01 * ne02);
			const size_t i02 = (row_idx - i03 * ne03) / ne01;
			const size_t i01 = row_idx - i03 * ne03 - i02 * ne02;

			const size_t i13 = i03 % ne13;
			const size_t i12 = i02 % ne12;
			const size_t i11 = i01 % ne11;

			for (size_t row_repeat_idx = 0; row_repeat_idx < ne00 / ne10; ++row_repeat_idx) {
				const size_t i00 = row_repeat_idx * ne00;
				const float *src0_row_ptr = operand_0.get<float>({i00, i01, i02, i03});
				const float *src1_row_ptr = operand_1.get<float>({0, i11, i12, i13});
				float *		 dst_row_ptr  = result.get<float>({i00, i01, i02, i03});
				std::transform(src0_row_ptr, src0_row_ptr + ne10, src1_row_ptr, 
					dst_row_ptr, std::multiplies<float>()
				);
			}
		}

		return true;
	}

	size_t OperatorDivImpl::get_task_num([[maybe_unused]] const CPUBackend *backend_ptr, const OperatorNode *op_node) {
		const auto &operand_0	= op_node->get_input<DataNode>(0).tensor;
		const auto &shape_0     = operand_0.get_shape();
		return shape_0.num_row();
	}

	size_t OperatorDivImpl::get_buffer_size([[maybe_unused]] const CPUBackend *backend_ptr, [[maybe_unused]] const OperatorNode *op_node) {
		return 0;
	}

	bool OperatorDivImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand_0 = op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1 = op_node->get_input<DataNode>(1).tensor;
		const auto &result    = op_node->get_output<DataNode>(0).tensor;

		const auto &shape_0     = operand_0.get_shape();
		const auto &shape_1     = operand_1.get_shape();

		const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
		const auto [ne10, ne11, ne12, ne13] = shape_1.elements;

		const size_t num_row = ne03 * ne02 * ne01;

		for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
			const size_t i03 = row_idx / (ne01 * ne02);
			const size_t i02 = (row_idx - i03 * ne03) / ne01;
			const size_t i01 = row_idx - i03 * ne03 - i02 * ne02;

			const size_t i13 = i03 % ne13;
			const size_t i12 = i02 % ne12;
			const size_t i11 = i01 % ne11;

			for (size_t row_repeat_idx = 0; row_repeat_idx < ne00 / ne10; ++row_repeat_idx) {
				const size_t i00 = row_repeat_idx * ne00;
				const float *src0_row_ptr = operand_0.get<float>({i00, i01, i02, i03});
				const float *src1_row_ptr = operand_1.get<float>({0, i11, i12, i13});
				float *		 dst_row_ptr  = result.get<float>({i00, i01, i02, i03});
				std::transform(src0_row_ptr, src0_row_ptr + ne10, src1_row_ptr, 
					dst_row_ptr, std::divides<float>()
				);
			}
		}

		return true;
	}

}  // namespace spy::cpu