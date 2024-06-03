#include <cmath>
#include <numbers>

#include "number/lookup_table.h"
#include "number/tensor.h"
#include "graph/graph.h"
#include "operator/config.h"
#include "operator/non-linear.h"
#include "operator_impl.h"

namespace spy::cpu {

	template<class T>
    struct Silu {
        T operator()(const T val) { return LOOK_UP_TABLE.silu(spy_fp32_to_fp16(val)); }
    };

	template<class T>
	struct Relu {
		T operator()(const T val) { return std::max(val, static_cast<T>(0)); }
	};

	std::shared_ptr<ControlHeader> OperatorReluImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        const auto &operand  = op_node->get_input<DataNode>(0).tensor;
        const auto &shape_operand = operand.get_shape();
        const int num_task = shape_operand.num_row();
        return std::make_shared<ControlHeader>(num_task);
    }

	OperatorResult OperatorReluImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand = op_node->get_input<DataNode>(0).tensor;
		const auto &result  = op_node->get_output<DataNode>(0).tensor;

        const auto &shape_operand = operand.get_shape();
        const auto &shape_res     = result.get_shape();

        const auto [ne00, ne01, ne02, ne03] = shape_operand.elements;
        const size_t num_row = ne01 * ne02 * ne03;

        for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
            const size_t i03 = row_idx / (ne01 * ne02);
            const size_t i02 = row_idx % (ne01 * ne02) / ne01;
            const size_t i01 = row_idx % ne01;

            const float *src0_row_ptr = operand.get<const float>({0, i01, i02, i03});
            float *		 dst_row_ptr  = result.get<float>({0, i01, i02, i03});
            std::transform(src0_row_ptr, src0_row_ptr + ne00, dst_row_ptr, Relu<float>());            
        }

        return { 0_op_end };
    }

	std::shared_ptr<ControlHeader> OperatorSiluImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        const auto &operand  = op_node->get_input<DataNode>(0).tensor;
        const auto &shape_operand = operand.get_shape();
        const int num_task = shape_operand.num_row();
        return std::make_shared<ControlHeader>(num_task);
    }

	OperatorResult OperatorSiluImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand = op_node->get_input<DataNode>(0).tensor;
		const auto &result  = op_node->get_output<DataNode>(0).tensor;

        const auto &shape_operand = operand.get_shape();

        const auto [ne00, ne01, ne02, ne03] = shape_operand.elements;
        const size_t num_row = ne01 * ne02 * ne03;

        for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
            const size_t i03 = row_idx / (ne01 * ne02);
            const size_t i02 = row_idx % (ne01 * ne02) / ne01;
            const size_t i01 = row_idx % ne01;

            const float *src0_row_ptr = operand.get<const float>({0, i01, i02, i03});
            float *		 dst_row_ptr  = result.get<float>({0, i01, i02, i03});
            std::transform(src0_row_ptr, src0_row_ptr + ne00, dst_row_ptr, Silu<float>());
        }

        return { 0_op_end };
    }

	std::shared_ptr<ControlHeader> OperatorSoftmaxImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        const auto &operand  = op_node->get_input<DataNode>(0).tensor;
        const auto &shape_operand = operand.get_shape();
        const int num_task = shape_operand.num_row();
        return std::make_shared<ControlHeader>(num_task);
    }

	OperatorResult operator_softmax_execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
        const auto &operand = op_node->get_input<DataNode>(0).tensor;
        const auto &result  = op_node->get_output<DataNode>(0).tensor;
        const float scale  = static_cast<OperatorDefinition<OperatorType::Softmax> *>(op_node)->scale;

        const auto &shape_res     = result.get_shape();

        const auto [ne0, ne1, ne2, ne3] = shape_res.elements;

        const size_t num_row = ne1 * ne2 * ne3;

        for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
            const size_t i3 = row_idx / (ne1 * ne2);
            const size_t i2 = row_idx % (ne1 * ne2) / ne1;
            const size_t i1 = row_idx % ne1;

            const float *src_ptr = operand.get<float>({0, i1, i2, i3});
            float *dst_ptr 	     = result.get<float>({0, i1, i2, i3});

            for (size_t i0 = 0; i0 < ne0; ++i0) {
                dst_ptr[i0] = src_ptr[i0] * scale;
            }

            const auto  max_iter = std::max_element(dst_ptr, dst_ptr + ne0);
            const float max_src = *max_iter;
            spy_assert(!std::isnan(max_src));

            for (size_t i0 = 0; i0 < ne0; ++i0) {
                dst_ptr[i0] -= max_src;
            }

            for (size_t i0 = 0; i0 < ne0; ++i0) {
                dst_ptr[i0] = LOOK_UP_TABLE.exp(spy_fp32_to_fp16(dst_ptr[i0]));
            }
            const float sum = std::reduce(dst_ptr, dst_ptr + ne0);
            spy_assert(sum > 0.0F);
            const float sum_inv = 1.0F / sum;
            for (size_t i0 = 0; i0 < ne0; ++i0) { dst_ptr[i0] *= sum_inv; }
        }

        return { 0_op_end };
    }

    static OperatorResult operator_masked_softmax_execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
        const auto &operand = op_node->get_input<DataNode>(0).tensor;
        const auto mask    = op_node->get_input<DataNode>(1).tensor;
        const auto &result  = op_node->get_output<DataNode>(0).tensor;
        const float scale  = static_cast<OperatorDefinition<OperatorType::Softmax> *>(op_node)->scale;

        const auto &shape_res     = result.get_shape();

        const auto [ne0, ne1, ne2, ne3] = shape_res.elements;

        const size_t num_row = ne1 * ne2 * ne3;

        for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
            const size_t i3 = row_idx / (ne1 * ne2);
            const size_t i2 = row_idx % (ne1 * ne2) / ne1;
            const size_t i1 = row_idx % ne1;

            const float *src_ptr  = operand.get<float>({0, i1, i2, i3});
            const float *mask_ptr = mask.get<float>({0, i1, 0, 0});
            float *dst_ptr 	      = result.get<float>({0, i1, i2, i3});

            for (size_t i0 = 0; i0 < ne0; ++i0) {
                dst_ptr[i0] = src_ptr[i0] + mask_ptr[i0];
            }

            for (size_t i0 = 0; i0 < ne0; ++i0) {
                dst_ptr[i0] *= scale;
            }

            const auto  max_iter = std::max_element(dst_ptr, dst_ptr + ne0);
            const float max_src = *max_iter;
            spy_assert(!std::isnan(max_src));

            for (size_t i0 = 0; i0 < ne0; ++i0) {
                dst_ptr[i0] -= max_src;
            }

            for (size_t i0 = 0; i0 < ne0; ++i0) {
                if (mask_ptr[i0] == -INFINITY) { 
                    dst_ptr[i0] = 0.0F; 
                } else {
                    dst_ptr[i0] = LOOK_UP_TABLE.exp(spy_fp32_to_fp16(dst_ptr[i0]));
                }
            }
            const float sum = std::reduce(dst_ptr, dst_ptr + ne0);
            spy_assert(sum > 0.0F);
            const float sum_inv = 1.0F / sum;
            for (size_t i0 = 0; i0 < ne0; ++i0) { dst_ptr[i0] *= sum_inv; }
        }

        return { 0_op_end };
    }

    OperatorResult OperatorSoftmaxImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
        if (op_node->get_input().size() == 1) { return operator_softmax_execute(backend_ptr, param, op_node); }
        return operator_masked_softmax_execute(backend_ptr, param, op_node);
    }

	std::shared_ptr<ControlHeader> OperatorNormRMSImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        const auto &operand  = op_node->get_input<DataNode>(0).tensor;
        const auto &shape_operand = operand.get_shape();
        const int num_task = shape_operand.num_row();
        return std::make_shared<ControlHeader>(num_task);
    }

	OperatorResult OperatorNormRMSImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
		const auto &operand = op_node->get_input<DataNode>(0).tensor;
		const auto &result  = op_node->get_output<DataNode>(0).tensor;
        const float eps = static_cast<OperatorDefinition<OperatorType::NormRMS> *>(op_node)->eps;

        const auto &shape_res = result.get_shape();

        const auto [ne0, ne1, ne2, ne3] = shape_res.elements;

        const size_t num_row = ne1 * ne2 * ne3;

        for (size_t row_idx = param.tid; row_idx < num_row; row_idx += param.concurrency) {
            const size_t i3 = row_idx / (ne1 * ne2);
            const size_t i2 = row_idx % (ne1 * ne2) / ne1;
            const size_t i1 = row_idx % ne1;

            const float *src_ptr = operand.get<float>({0, i1, i2, i3});
            float *		 dst_ptr = result.get<float>({0, i1, i2, i3});

            const float sum   = std::transform_reduce(src_ptr, src_ptr + ne0, 0.0F, std::plus<float>{}, [](const float x){ return x * x; });
            const float mean  = sum / static_cast<float>(ne0);
            const float scale = 1.0F / std::sqrt(mean + eps);
            std::transform(src_ptr, src_ptr + ne0, dst_ptr, [scale](const float x){ return x * scale; });
        }

        return { 0_op_end };
    }
	

    static float rope_yarn_corr_dim(const int32_t num_dim, const int32_t num_origin_context, 
            const float num_rot, const float base) {
        return static_cast<float>(num_dim) * std::log(num_origin_context / (num_rot * 2 * std::numbers::pi)) / (2 * std::log(base));
    }

    static std::array<float, 2> rope_yarn_corr_dims(const int32_t num_dim, const int32_t num_origin_context, 
            const float freq_base, const float beta_fast, const float beta_slow) {
        const float start = std::floor(rope_yarn_corr_dim(num_dim, num_origin_context, beta_fast, freq_base));
        const float end   = std::ceil(rope_yarn_corr_dim(num_dim, num_origin_context, beta_slow, freq_base));

        return {
            std::max(0.0F, start),
            std::min(static_cast<float>(num_dim - 1), end)
        };
    }

    static float rope_yarn_ramp(const float low, const float high, const int i0) {
        const float y = (i0 / 2 - low) / std::max(0.001F, high - low);
        return 1.0F - std::clamp(0.0F, 1.0F, y);
    }

    static void rope_yarn(float theta_extrap, float freq_scale, std::array<float, 2> corr_dims, int64_t i0, float ext_factor, float mscale,
        float * cos_theta, float * sin_theta) {
        // Get n-d rotational scaling corrected for extrapolation
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        if (ext_factor != 0.0F) {
            const float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
            theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
            // Get n-d magnitude scaling corrected for interpolation
            mscale *= 1.0F + 0.1F * std::log(1.0F / freq_scale);
        }
        *cos_theta = std::cos(theta) * mscale;
        *sin_theta = std::sin(theta) * mscale;
    }

	std::shared_ptr<ControlHeader> OperatorRopeImpl::get_control_header([[maybe_unused]] CPUBackend *backend_ptr, const OperatorNode *op_node) {
        const auto &operand_0 = op_node->get_input<DataNode>(0).tensor;
        const auto &shape_0 = operand_0.get_shape();
        const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
        const int num_task = ne03 * ne02; 
        return std::make_shared<ControlHeader>(num_task);
    }

	OperatorResult OperatorRopeImpl::execute([[maybe_unused]] CPUBackend *backend_ptr, const OperatorEnvParam &param, OperatorNode *op_node) {
        const auto &operand_0    = op_node->get_input<DataNode>(0).tensor;
        const auto &operand_1    = op_node->get_input<DataNode>(1).tensor;
        const auto &result       = op_node->get_output<DataNode>(0).tensor;
        const auto rope_context = static_cast<OperatorDefinition<OperatorType::Rope> *>(op_node)->rope_context;

        const auto [mode, num_past, num_dim, num_context, num_origin_context,
        freq_base, freq_scale, extend_factor, attention_factor,
        beta_fast, beta_slow, xpos_base, xpos_down] = rope_context;

        const auto &shape_res= result.get_shape();

        float theta_scale = std::pow(freq_base, -2.0F / num_dim);
        const float inv_num_dim = -1.0F / num_dim;

        const float sin_sign = 1.0F;

        const auto corr_dims = 
                rope_yarn_corr_dims(num_dim, num_origin_context, freq_base, beta_fast, beta_slow);

        const auto [ne0, ne1, ne2, ne3] = shape_res.elements;

        const int32_t *src1_ptr = operand_1.get<int32_t>();

        const size_t num_matrix = ne2 * ne3;

        for (size_t matrix_idx = param.tid; matrix_idx < num_matrix; ++matrix_idx) {
            const size_t i3 = matrix_idx / ne2;
            const size_t i2 = matrix_idx % ne2;

            const int32_t p = src1_ptr[i2];

            float *cache = static_cast<float *>(alloca(ne0 * sizeof(float)));
            if (mode != ModelRopeType::GLM && mode != ModelRopeType::Neox) {
                const int32_t *pos_ptr = operand_1.get<const int32_t>({i2, 0, 0, 0});
                float theta = static_cast<float>(*pos_ptr);
                for (uint64_t i0 = 0; i0 < ne0; i0 += 2) {
                    rope_yarn(theta, freq_scale, corr_dims, i0, extend_factor, attention_factor, &cache[i0 + 0], &cache[i0 + 1]);
                    cache[i0 + 1] *= sin_sign;
                    theta 		  *= theta_scale;
                }
            }

            for (size_t i1 = 0; i1 < ne1; ++i1) {

                switch (mode) {
                    case ModelRopeType::GLM: {
                        float theta_base  = std::min(p, num_context - 2);
                        float block_theta = std::max(p - (num_context - 2), 0);

                        for (size_t i0 = 0; i0 < ne0 / 4; ++i0) {
                            const float cos_theta       = std::cos(theta_base);
                            const float sin_theta       = std::sin(theta_base) * sin_sign;
                            const float cos_block_theta = std::cos(block_theta);
                            const float sin_block_theta = std::sin(block_theta) * sin_sign;

                            theta_base  *= theta_scale;
                            block_theta *= theta_scale;


                            const float *const src_ptr = operand_0.get<float>({i0, i1, i2, i3});
                            float *const dst_ptr	   = result.get<float>({i0, i1, i2, i3});

                            const float x0 = src_ptr[0];
                            const float x1 = src_ptr[num_dim / 2];
                            const float x2 = src_ptr[num_dim];
                            const float x3 = src_ptr[num_dim / 2 + num_dim];

                            dst_ptr[0]						= x0 * cos_theta - x1 * sin_theta;
                            dst_ptr[num_dim / 2]			= x0 * sin_theta + x1 * cos_theta;
                            dst_ptr[num_dim]				= x2 * cos_block_theta - x3 * sin_block_theta;
                            dst_ptr[num_dim / 2 + num_dim] 	= x2 * sin_block_theta + x3 * cos_block_theta;
                        }
                        break;
                    }
                    case ModelRopeType::Neox:
                        spy_assert(false, "To be implemented");
                        break;

                    default: {
                        for (size_t i0 = 0; i0 < ne0; i0 += 2) {
                            const float cos_theta = cache[i0 + 0];
                            const float sin_theta = cache[i0 + 1];

                            float zeta = (xpos_base != 0.0F) ? std::pow((i0 + 0.4F * ne0) / (1.4F * ne0), p / xpos_base) : 1.0F;
                            if (xpos_down) { zeta = 1.0F / zeta; }

                            const float *const src_ptr = operand_0.get<float>({i0, i1, i2, i3});
                            float *const dst_ptr	   = result.get<float>({i0, i1, i2, i3});

                            const float x0 = src_ptr[0];
                            const float x1 = src_ptr[1];

                            dst_ptr[0] = x0 * cos_theta * zeta - x1 * sin_theta * zeta;
                            dst_ptr[1] = x0 * sin_theta * zeta + x1 * cos_theta * zeta;
                        }
                    }
                }
            }
        }

        return { 0_op_end };
    }

}  // namespace spy::cpu