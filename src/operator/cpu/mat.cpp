#include <immintrin.h>
#include <magic_enum.hpp>
#include <magic_enum_switch.hpp>

#include "util/logger.h"
#include "number/tensor.h"
#include "number/compute/dot.h"
#include "number/quantization.h"
#include "graph/graph.h"
#include "operator/operator.h"
#include "backend/cpu/operator_impl.h"

namespace spy::cpu {

    using magic_enum::enum_name;
    using magic_enum::enum_switch;

	size_t OperatorMatMulImpl::get_task_num([[maybe_unused]] const CPUBackend *backend_ptr, const OperatorNode *op_node) {
		const auto &operand_0	= op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1	= op_node->get_input<DataNode>(1).tensor;

        const auto &shape_0     = operand_0.get_shape();
        const auto &shape_1     = operand_1.get_shape();

        const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
        const auto [ne10, ne11, ne12, ne13] = shape_1.elements;

        return ne03 * ne02 * ne11 * ne01;
	}

	size_t OperatorMatMulImpl::get_buffer_size([[maybe_unused]] const CPUBackend *backend_ptr, const OperatorNode *op_node) {
		const auto &operand_0	= op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1	= op_node->get_input<DataNode>(1).tensor;

        const auto &shape_0     = operand_0.get_shape();
        const auto &shape_1     = operand_1.get_shape();

		const auto type_0 = shape_0.number_type;
		const auto type_1 = shape_1.number_type;

        if (is_trivial(type_0) && is_trivial(type_1)) { return 0; }

        switch (type_0) {
        case NumberType::FP16:
            return get_row_size(NumberType::FP16, shape_1.total_element());
        case NumberType::Q8_0:
            return get_row_size(NumberType::Q8_0, shape_1.total_element());
        default:
	        SPY_ASSERT_FMT(false, "Unimplemented operator of type: {}", enum_name(type_0));
			return 0;
        }
        return 0;
	}

    bool OperatorMatMulImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand_0 = op_node->get_input<DataNode>(0).tensor;
		const auto &operand_1 = op_node->get_input<DataNode>(1).tensor;
		const auto &result    = op_node->get_output<DataNode>(0).tensor;

        const auto &shape_0     = operand_0.get_shape();
        const auto &shape_1     = operand_1.get_shape();
        const auto &shape_res   = result.get_shape();

		const auto type_0 = shape_0.number_type;
		const auto type_1 = shape_1.number_type;

        const auto [ne0, ne1, ne2, ne3]     = shape_res.elements;
        const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
        const auto [ne10, ne11, ne12, ne13] = shape_1.elements;

        const bool   is_contiguous_1 = shape_1.is_continugous();
		const size_t num_dst         = shape_res.total_element();
        const size_t row_size 		 = get_row_size(shape_1.number_type, ne10);

        const bool has_buffer = !param.buffer_span.empty();

        const auto matmul_without_buffer = [&](){
            SPY_ASSERT_FMT(type_0 == type_1, "Expect the operands to be of the same type: {}, {}", enum_name(type_0), enum_name(type_1));

            const auto dot_func = enum_switch([type_1](auto T_type_0) {
                return enum_switch([T_type_0](auto T_type_1) {
                    return Dot<T_type_0, T_type_1>::exec_raw;
                }, type_1);
            }, type_0);

            for (size_t col_idx = param.tid; col_idx < num_dst; col_idx += param.concurrency) {
                const size_t i03 = col_idx / (ne02 * ne11 * ne01);
                const size_t i02 = col_idx % (ne02 * ne11 * ne01) / (ne11 * ne01);
                const size_t i11 = col_idx % (ne11 * ne01) / ne01;
                const size_t i01 = col_idx % (ne11 * ne01) % ne01;

                const void *src1_col = operand_1.get<const void>({0, i11, i02, i03});
                const void *src0_row = operand_0.get<const void>({0, i01, i02, i03});
                float *dst_element   = result.get<float>({i01, i11, i02, i03});
                      *dst_element   = dot_func(src0_row, src1_col, ne00);
            }
        };

        const auto matmul_with_buffer = [&](NumberType type_mid){
            const auto dot_func = enum_switch([type_mid](auto T_type_0) {
                return enum_switch([T_type_0](auto T_type_1) {
                    return Dot<T_type_0, T_type_1>::exec_raw;
                }, type_mid);
            }, type_0);

            // Init buffer
            auto *mat_node_ptr = static_cast<OperatorDefinition<OperatorType::MatMul> *>(op_node);
            auto &buffer_init_counter = mat_node_ptr->buffer_init_counter;
            auto &buffer_done_counter = mat_node_ptr->buffer_done_counter;
            const int num_src1_row       = shape_1.num_row();
            const size_t buffer_row_size = get_row_size(type_mid, ne10);

            SPY_ASSERT_FMT(num_src1_row * buffer_row_size <= param.buffer_span.size(), 
                "The size of buffer is less than that needed (buffer: {}, need: {})", param.buffer_span.size(), num_src1_row * buffer_row_size);

            while (true) {
                int cur_src1_row = buffer_init_counter++;
                if (cur_src1_row >= num_src1_row) { break; }

                const size_t i13 = cur_src1_row / (ne12 * ne11);
                const size_t i12 = cur_src1_row % (ne12 * ne11) / ne11;
                const size_t i11 = cur_src1_row %  ne11;

                const void *src1_row = operand_1.get<const void>({0, i11, i12, i13});
                void *buffer_row     = param.buffer_span.data() + cur_src1_row * buffer_row_size;
                auto_quantize_inner(type_1, src1_row, type_mid, buffer_row, ne10 / get_block_size(type_1));

                ++buffer_done_counter;
            }
            while (buffer_done_counter.load() < num_src1_row) { }

            // Compute
            for (size_t col_idx = param.tid; col_idx < num_dst; col_idx += param.concurrency) {
                const size_t i03 = col_idx / (ne02 * ne11 * ne01);
                const size_t i02 = col_idx % (ne02 * ne11 * ne01) / (ne11 * ne01);
                const size_t i11 = col_idx % (ne11 * ne01) / ne01;
                const size_t i01 = col_idx % (ne11 * ne01) % ne01;

                const void *src1_col = param.buffer_span.data() + (i11 + i02 * ne11 + i03 * ne11 * ne12) * buffer_row_size;
                const void *src0_row = operand_0.get<const void>({0, i01, i02, i03});
                float *dst_element   = result.get<float>({i01, i11, i02, i03});
                      *dst_element   = dot_func(src0_row, src1_col, ne00);
            }
        };


        switch (type_0) {
        case NumberType::Q8_0:
            matmul_with_buffer(NumberType::Q8_0);
            break;
        
        case NumberType::FP16:
            matmul_with_buffer(NumberType::FP16);
            break;

        default: /* Matmul without buffer */
            matmul_without_buffer();
            break;
        }

        return true;
    }
    
}  // namespace spy::cpu