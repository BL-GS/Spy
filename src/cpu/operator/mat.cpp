#include <latch>
#include <memory>

#include "util/shell/logger.h"
#include "util/align.h"
#include "number/tensor.h"
#include "number/compute/dot.h"
#include "number/quantization.h"
#include "graph/graph.h"
#include "operator_impl.h"

namespace spy::cpu {

	struct BufferLatchControlHeader: public BufferControlHeader {
		Uninitialized<std::latch> latch;

		BufferLatchControlHeader() = default;
		BufferLatchControlHeader(int num_task): BufferControlHeader(num_task) {}
		BufferLatchControlHeader(int num_task, CPUBackend *backend_ptr, int size): BufferControlHeader(num_task, backend_ptr, size) { }

		~BufferLatchControlHeader() override = default;

		void init(const spy::cpu::OperatorEnvParam &param) override { latch.put_value(param.concurrency); }
	};

    inline constexpr NumberType target_buffer_type(NumberType type_0, NumberType type_1) {
        return type_0;
    }

	std::shared_ptr<ControlHeader> OperatorMatMulImpl::get_control_header(CPUBackend *backend_ptr, const OperatorNode *op_node) {
		const auto &operand_0	= op_node->input(0).tensor;
		const auto &operand_1	= op_node->input(1).tensor;

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
        return std::make_shared<BufferLatchControlHeader>(num_task, backend_ptr, buffer_size);
	}

    OperatorResult OperatorMatMulImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand_0 = op_node->input(0).tensor;
		const auto &operand_1 = op_node->input(1).tensor;
		const auto &result    = op_node->output(0).tensor;

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
            const auto dot_func = NumberTypeMapper::product_map([](const auto T_type_0, const auto T_type_1){
                return Dot<T_type_0, T_type_1>::exec_raw;
            }, type_0, type_mid);

            // Init buffer
            const int num_src1_row       = shape_1.num_row();
            const size_t buffer_row_size = get_row_size(type_mid, ne10);

            spy_assert(num_src1_row * buffer_row_size <= buffer_size,
                "The size of buffer is less than that needed (buffer: {}, need: {})", buffer_size, num_src1_row * buffer_row_size);

			const int avg_src1_row = div_ceil(num_src1_row, param.concurrency);
            for (int cur_src1_row = param.tid * avg_src1_row; cur_src1_row < num_src1_row; ++cur_src1_row) {
                const size_t i13 = cur_src1_row / (ne12 * ne11);
                const size_t i12 = cur_src1_row % (ne12 * ne11) / ne11;
                const size_t i11 = cur_src1_row %  ne11;

                const void *src1_row = operand_1.get<const void>({0, i11, i12, i13});
                void *buffer_row     = buffer_ptr + cur_src1_row * buffer_row_size;
                auto_quantize_inner(type_1, src1_row, type_mid, buffer_row, ne10 / get_block_size(type_1));
            }
			header_ptr->latch.value.arrive_and_wait();

            // Compute
            for (size_t col_idx = param.tid; col_idx < num_dst; col_idx += param.concurrency) {
                const size_t i03 = col_idx / (ne02 * ne11 * ne01);
                const size_t i02 = col_idx % (ne02 * ne11 * ne01) / (ne11 * ne01);
                const size_t i11 = col_idx % (ne11 * ne01) / ne01;
                const size_t i01 = col_idx % (ne11 * ne01) % ne01;

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