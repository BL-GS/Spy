#include <memory>

#include "util/shell/logger.h"
#include "number/tensor.h"
#include "graph/graph.h"
#include "operator_impl.h"
#include "simd/vec_dot.h"
#include "simd/vec_convert.h"

namespace spy::cpu {

	struct BufferLatchControlHeader: public BufferControlHeader {
		std::atomic<int> src1_quantize_counter;
		std::atomic<int> src1_quantize_done;

		BufferLatchControlHeader() = default;
		BufferLatchControlHeader(int num_task, int num_src1_row):
			BufferControlHeader(num_task), src1_quantize_counter(num_src1_row), src1_quantize_done(num_src1_row) {}
		BufferLatchControlHeader(int num_task, int num_src1_row, CPUBackend *backend_ptr, int size):
			BufferControlHeader(num_task, backend_ptr, size), src1_quantize_counter(num_src1_row), src1_quantize_done(num_src1_row) {}

		~BufferLatchControlHeader() override = default;
	};

    inline constexpr NumberType target_buffer_type(NumberType type_0, NumberType type_1) {
        return type_0;
    }

	std::shared_ptr<ControlHeader> OperatorMatMulImpl::get_control_header(CPUBackend *backend_ptr, const DerivedOperatorNode *op_node) {
		const auto &operand_0	= op_node->input_data(0)->tensor;
		const auto &operand_1	= op_node->input_data(1)->tensor;

        const auto &shape_0     = operand_0.get_shape();
        const auto &shape_1     = operand_1.get_shape();

        const NumberType type_0 = shape_0.number_type;
        const NumberType type_1 = shape_1.number_type;

        const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
        const auto [ne10, ne11, ne12, ne13] = shape_1.elements;

        const int num_task = ne03 * ne02 * ne11 * ne01;

        const NumberType type_mid    = target_buffer_type(type_0, type_1);
        const int num_src1_row       = shape_1.num_row();
        const size_t buffer_row_size = get_row_size(type_mid, ne10);
        const size_t buffer_size     = num_src1_row * buffer_row_size;
        return std::make_shared<BufferLatchControlHeader>(num_task, num_src1_row, backend_ptr, buffer_size);
	}

    OperatorResult OperatorMatMulImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, DerivedOperatorNode *op_node) {
		const auto &operand_0 = op_node->input_data(0)->tensor;
		const auto &operand_1 = op_node->input_data(1)->tensor;
		const auto &result    = op_node->output_data(0)->tensor;

        const auto &shape_0     = operand_0.get_shape();
        const auto &shape_1     = operand_1.get_shape();
        const auto &shape_res   = result.get_shape();

		const auto type_0 = shape_0.number_type;
		const auto type_1 = shape_1.number_type;

        const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
        const auto [ne10, ne11, ne12, ne13] = shape_1.elements;

		const size_t num_dst         = shape_res.total_element();

        auto *   header_ptr   = static_cast<BufferLatchControlHeader *>(param.header_ptr.get());
        const   auto buffer_span   = header_ptr->data_span;
        uint8_t *buffer_ptr        = buffer_span.data();
        const   size_t buffer_size = buffer_span.size_bytes();

        const auto matmul_without_buffer = [&](){
            spy_assert(type_0 == type_1, "Expect the operands to be of the same type: {}, {}", type_0, type_1);

            const auto dot_func = NumberTypeMapper::product_map([](const auto T_type_0, const auto T_type_1){
                return Dot<T_type_0, T_type_1>::exec_raw;
            }, type_0, type_1);

            for (size_t col_idx = param.tid; col_idx < num_dst; col_idx += param.concurrency) {
                const int64_t i03 = col_idx / (ne02 * ne11 * ne01);
                const int64_t i02 = col_idx % (ne02 * ne11 * ne01) / (ne11 * ne01);
                const int64_t i11 = col_idx % (ne11 * ne01) / ne01;
                const int64_t i01 = col_idx % (ne11 * ne01) % ne01;

                const void *src1_col = operand_1.get<const void>({0, i11, i02, i03});
                const void *src0_row = operand_0.get<const void>({0, i01, i02, i03});
                float *dst_element   = result.get<float>({i01, i11, i02, i03});
                      *dst_element   = dot_func(src0_row, src1_col, ne00);
            }

            if (op_node->input_data(0)->node_prop.layer_id == 0 && op_node->name == "KQV") {
                for (int64_t i1 = 0; i1 < ne11; ++i1) {
                    for (int64_t i0 = 0; i0 < 10; ++i0) {
                        printf("%.3f ", *result.get<float>({i0, i1, 0, 0}));
                    }
                    printf("\n");
                }
                printf("\n");
            }
        };

        const auto matmul_with_buffer = [&](NumberType type_mid){
            const auto dot_func = NumberTypeMapper::product_map([](const auto T_type_0, const auto T_type_1){
                return Dot<T_type_0, T_type_1>::exec_raw;
            }, type_0, type_mid);

            // Init buffer
            const int num_src1_row       = shape_1.num_row();
            const size_t buffer_row_size = get_row_size(type_mid, ne10);

            spy_assert(num_src1_row * buffer_row_size <= buffer_size,
                "The size of buffer is less than that needed (buffer: {}, need: {})", buffer_size, num_src1_row * buffer_row_size);

			while (true) {
				const int cur_src1_row = --header_ptr->src1_quantize_counter;
				if (cur_src1_row < 0) { break; }

                const int64_t i13 = cur_src1_row / (ne12 * ne11);
                const int64_t i12 = cur_src1_row % (ne12 * ne11) / ne11;
                const int64_t i11 = cur_src1_row %  ne11;

                const void *src1_row = operand_1.get<const void>({0, i11, i12, i13});
                void *buffer_row     = buffer_ptr + cur_src1_row * buffer_row_size;
                auto_quantize_inner(type_1, src1_row, type_mid, buffer_row, ne10 / get_block_size(type_1));

				--header_ptr->src1_quantize_done;
            }
			while (header_ptr->src1_quantize_done.load() != 0) { std::this_thread::yield(); }

            // Compute
            for (size_t col_idx = param.tid; col_idx < num_dst; col_idx += param.concurrency) {
                const int64_t i03 = col_idx / (ne02 * ne11 * ne01);
                const int64_t i02 = col_idx % (ne02 * ne11 * ne01) / (ne11 * ne01);
                const int64_t i11 = col_idx % (ne11 * ne01) / ne01;
                const int64_t i01 = col_idx % (ne11 * ne01) % ne01;

                const void *src1_col = buffer_ptr + (i11 + i02 * ne11 + i03 * ne11 * ne12) * buffer_row_size;
                const void *src0_row = operand_0.get<const void>({0, i01, i02, i03});
                float *dst_element   = result.get<float>({i01, i11, i02, i03});
                      *dst_element   = dot_func(src0_row, src1_col, ne00);
            }
        };

        const NumberType target_type = target_buffer_type(type_0, type_1);
        if (type_1 == target_type) {
            matmul_without_buffer();
        } else {
            matmul_with_buffer(target_type);
        }

        return { 0_op_end };
    }
    
}  // namespace spy::cpu