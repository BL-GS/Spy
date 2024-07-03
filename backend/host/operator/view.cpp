/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#include <cstring>
#include <array>
#include <magic_enum_fuse.hpp>

#include "number/tensor.h"
#include "graph/graph.h"
#include "operator/config.h"
#include "operator/view.h"
#include "operator_impl.h"
#include "simd/vec_convert.h"

namespace spy::cpu {

	std::shared_ptr<ControlHeader> OperatorNopImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        return nullptr;
    }

	OperatorResult OperatorNopImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
        return { 0_op_end };
    }

	std::shared_ptr<ControlHeader> OperatorGetRowImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        const auto &operand_1  = op_node->input(1).tensor;
        const auto &shape_1 = operand_1.get_shape();
		const auto [ne10, ne11, ne12, ne13] = shape_1.elements;
        const int num_task = ne12 * ne11 * ne10;
        return std::make_shared<ControlHeader>(num_task);
    }

	OperatorResult OperatorGetRowImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand_0 = op_node->input(0).tensor;
		const auto &operand_1 = op_node->input(1).tensor;
		const auto &result    = op_node->output(0).tensor;

        const auto &shape_0     = operand_0.get_shape();
        const auto &shape_1     = operand_1.get_shape();
        const auto &shape_res   = result.get_shape();

		const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
		const auto [ne10, ne11, ne12, ne13] = shape_1.elements;

        const size_t num_idx       = ne10 * ne11 * ne12;
        const size_t src0_num_row  = shape_0.num_row();
        const size_t src0_row_size = shape_0.row_size();
        const size_t res_row_size  = shape_res.row_size();

        const NumberType type_0   = shape_0.number_type;
        const NumberType type_res = shape_res.number_type;

        const int32_t *row_idx_ptr = operand_1.get<int32_t>();
        uint8_t *dst_ptr 		   = result.get<uint8_t>();

        for (size_t row_idx = param.tid; row_idx < num_idx; row_idx += param.concurrency) {
            const size_t i12 = row_idx / (ne10 * ne11);
            const size_t i11 = row_idx % (ne10 * ne11) / ne10; 

            const size_t selected_row_idx = row_idx_ptr[row_idx];
            spy_assert(selected_row_idx < src0_num_row, "Access invalid row idx: {} (num: {})", selected_row_idx, src0_num_row);

            const void *src_ptr = operand_0.get({0, selected_row_idx, i11, i12});

            if (type_0 == type_res) {
                std::memcpy(dst_ptr + src0_row_size * row_idx, src_ptr, src0_row_size);
            } else {
                using magic_enum::enum_fuse;

                switch (enum_fuse(type_0, type_res).value()) {
                case enum_fuse(NumberType::Q8_0, NumberType::FP32).value(): {
                    quantize_inner<NumberType::Q8_0, NumberType::FP32>(src_ptr, dst_ptr + res_row_size * row_idx, ne00 / get_block_size(NumberType::Q8_0));
                } break;
                default:
                    spy_assert(false, "Unimplemented get row operator {} -> {}", type_0, type_res);
                }
            }
        }

        return { 0_op_end };
    }

    static OperatorResult dup_execute(CPUBackend *backend_ptr, const OperatorEnvParam &param, Tensor &result, const Tensor &operand) {
        const auto &shape_operand = operand.get_shape();
        const auto &shape_res     = result.get_shape();

        const auto [ne0, ne1, ne2, ne3] = shape_res.elements;
        const auto [ne00, ne01, ne02, ne03] = shape_operand.elements;

        const size_t num_row      = shape_operand.num_row();
        const size_t src_row_size = shape_operand.row_size();

        const NumberType type_operand = shape_operand.number_type;
        const NumberType type_result  = shape_res.number_type;

        for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
            const size_t i03 = row_idx / (ne01 * ne02);
            const size_t i02 = row_idx % (ne01 * ne02) / ne01; 
            const size_t i01 = row_idx % ne01;

            const void *src_ptr     = operand.get({0, i01, i02, i03});

            if (operand.is_continuous()) {
                const void *src_end_ptr = operand.get({ne00, i01, i02, i03});

                if (shape_operand == shape_res) {
                    void *dst_ptr = result.get({0, i01, i02, i03});
                    if (type_operand == type_result) {
                        std::memcpy(dst_ptr, src_ptr, src_row_size);
                    } else {
                        using magic_enum::enum_fuse;

                        switch (enum_fuse(type_operand, type_result).value()) {
                        case enum_fuse(NumberType::FP32, NumberType::FP16).value():
                            std::transform(static_cast<const float *>(src_ptr), static_cast<const float *>(src_end_ptr), 
                                static_cast<uint16_t *>(dst_ptr), 
                                [](const float x){ return spy_fp32_to_fp16(x); }
                            );
                            // Quantizator<NumberType::FP32, NumberType::FP16>::transform(
                            //     static_cast<const float *>(src_ptr), static_cast<uint16_t *>(dst_ptr), ne00);
                            break;
                        case enum_fuse(NumberType::FP16, NumberType::FP32).value():
                            std::transform(static_cast<const uint16_t *>(src_ptr), static_cast<const uint16_t *>(src_end_ptr), 
                                static_cast<float *>(dst_ptr), 
                                [](const uint16_t x){ return LOOK_UP_TABLE.fp32(x); }
                            );
                            break;
                        default:
                            spy_assert(false, "Unimplemented dup operator for {} -> {}", type_operand, type_result);
                        }
                    }                     
                } else if (result.is_continuous()) {
                    const size_t dst_row_size = get_row_size(type_result, ne00);
                    uint8_t *dst_ptr = result.get<uint8_t>() + dst_row_size * row_idx;
                    
                    if (type_operand == type_result) {
                        std::memcpy(dst_ptr, src_ptr, src_row_size);
                    } else {
                        using magic_enum::enum_fuse;

                        switch (enum_fuse(type_operand, type_result).value()) {
                        case enum_fuse(NumberType::FP32, NumberType::FP16).value():
                            std::transform(static_cast<const float *>(src_ptr), static_cast<const float *>(src_end_ptr), 
                                reinterpret_cast<uint16_t *>(dst_ptr), 
                                [](const float x){ return spy_fp32_to_fp16(x); }
                            );
                            // Quantizator<NumberType::FP32, NumberType::FP16>::transform(
                            //     static_cast<const float *>(src_ptr), static_cast<uint16_t *>(dst_ptr), ne00);
                            break;
                        case enum_fuse(NumberType::FP16, NumberType::FP32).value():
                            std::transform(static_cast<const uint16_t *>(src_ptr), static_cast<const uint16_t *>(src_end_ptr), 
                                reinterpret_cast<float *>(dst_ptr), 
                                [](const uint16_t x){ return LOOK_UP_TABLE.fp32(x); }
                            );
                            break;
                        default:
                            spy_assert(false, "Unimplemented dup operator for {} -> {}", type_operand, type_result);
                        }
                    }      
                } else {
                    spy_assert(false);
                }
            } else {
                for (size_t i00 = 0; i00 < ne00; ++i00) {
                    using magic_enum::enum_fuse;

                    const size_t global_iter = i00 + ne00 * row_idx;
                    const size_t i0 = global_iter % ne0;
                    const size_t i1 = global_iter / ne0 % ne1;
                    const size_t i2 = global_iter % (ne2 * ne1 * ne0) / (ne1 * ne0);
                    const size_t i3 = global_iter / (ne2 * ne1 * ne0);

                    switch (enum_fuse(type_operand, type_result).value()) {
                    case enum_fuse(NumberType::FP32, NumberType::FP16).value(): {
                        const float *src_ptr = operand.get<float>({i00, i01, i02, i03});
                        uint16_t *dst_ptr = result.get<uint16_t>({i0, i1, i2, i3});
                        *dst_ptr = spy_fp32_to_fp16(*src_ptr);
                        break;                        
                    }
                    default:
                        spy_assert(false);
                    }
                    
                }
            }
        }

        return { 0_op_end };
    }

	std::shared_ptr<ControlHeader> OperatorDupImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        const auto &operand  = op_node->input(0).tensor;
        const auto &shape_operand = operand.get_shape();
        const int num_task = shape_operand.num_row();
        return std::make_shared<ControlHeader>(num_task);
    }

	OperatorResult OperatorDupImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand = op_node->input(0).tensor;
		      auto &result  = op_node->output(0).tensor;

        return dup_execute(backend_ptr, param, result, operand);
	}


	std::shared_ptr<ControlHeader> OperatorCopyImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        return OperatorDupImpl::get_control_header(backend_ptr, op_node);
    }

	OperatorResult OperatorCopyImpl::execute(CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand_0 = op_node->input(0).tensor;
		      auto &operand_1  = op_node->input(1).tensor;
			  auto &result  = op_node->output(0).tensor;

        const OperatorResult ret = dup_execute(backend_ptr, param, operand_1, operand_0);
		if (ret.status == OperatorStatus::Success) { result.set_data_ptr(operand_1.get()); }
		return ret;
    }

	std::shared_ptr<ControlHeader> OperatorReshapeImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        return nullptr;
    }

	OperatorResult OperatorReshapeImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand = op_node->input(0).tensor;
		auto &result  = op_node->output(0).tensor;

        result.set_data_ptr(operand.get());
        return { 0_op_end };
    }

	std::shared_ptr<ControlHeader> OperatorViewImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        return nullptr;
    }

	OperatorResult OperatorViewImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
        const auto &operand  = op_node->input(0).tensor;
        auto  &result        = op_node->output(0).tensor;
        const int64_t offset = static_cast<OperatorDefinition<OperatorType::View> *>(op_node)->offset;

        result.set_data_ptr(operand.get<uint8_t>() + offset);
        return { 0_op_end };
    }

	std::shared_ptr<ControlHeader> OperatorTransposeImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        return nullptr;
    }

    OperatorResult OperatorTransposeImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand = op_node->input(0).tensor;
		auto &result  = op_node->output(0).tensor;

        result.set_data_ptr(operand.get());
        return { 0_op_end };
    }

	std::shared_ptr<ControlHeader> OperatorPermuteImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        return nullptr;
    }

	OperatorResult OperatorPermuteImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand = op_node->input(0).tensor;
		auto &result  = op_node->output(0).tensor;

        result.set_data_ptr(operand.get());
        return { 0_op_end };
    }

	std::shared_ptr<ControlHeader> OperatorContiguousImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        const auto &operand  = op_node->input(0).tensor;
        const auto &shape_operand = operand.get_shape();
        const int num_task = shape_operand.num_row();
        return std::make_shared<ControlHeader>(num_task);
    }

	OperatorResult OperatorContiguousImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
        const auto &operand  = op_node->input(0).tensor;
        auto  &result        = op_node->output(0).tensor;

        const auto &shape_operand = operand.get_shape();

        const auto [ne00, ne01, ne02, ne03] = shape_operand.elements;

        char *dst_ptr = result.get<char>();

        const size_t num_row  = shape_operand.num_row();
        const size_t row_size = shape_operand.row_size();
        
        if (!shape_operand.is_transposed()) {
            for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
                const size_t i03 = row_idx / (ne01 * ne02);
                const size_t i02 = row_idx % (ne01 * ne02) / ne01;
                const size_t i01 = row_idx % ne01;

                const float *src_row = operand.get<const float>({0, i01, i02, i03});
                std::memcpy(dst_ptr + row_idx * row_size, src_row, row_size);
            }				
        } else {
            for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
                const size_t i03 = row_idx / (ne01 * ne02);
                const size_t i02 = row_idx % (ne01 * ne02) / ne01;
                const size_t i01 = row_idx % ne01;
                for (size_t i00 = 0; i00 < ne00; ++i00) {
                    const float *src = operand.get<const float>({i00, i01, i02, i03});
                          float *dst = result.get<float>({i00, i01, i02, i03});
                                *dst = *src;
                }
            }
        }

        return { 0_op_end };
    }

}  // namespace spy::cpu
