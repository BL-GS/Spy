#include <cuda_runtime.h>
#include <cublas.h>

#include "util/shell/logger.h"
#include "number/number.h"
#include "gpu_mem.h"
#include "gpu_util.h"
#include "operator/detail/convert.h"
#include "operator/detail/matmul.h"


namespace spy::gpu {

    static void cuda_op_mul_mat_cublas(DeviceContext &ctx, 
        const Tensor &result, const Tensor &operand_0, const Tensor &operand_1, 
        cudaStream_t stream) {

        auto *pool_ptr = ctx.get_memory_pool();

        const auto [ne0, ne1, ne2, ne3]     = result.element_array();
        const auto [ne00, ne01, ne02, ne03] = operand_0.element_array();
        const auto [ne10, ne11, ne12, ne13] = operand_1.element_array();

        const NumberType type_0 = operand_0.get_number_type();
        const NumberType type_1 = operand_1.get_number_type();

        if ((type_0 == NumberType::FP16 || is_quantized(type_0)) && operand_0.is_continuous()) {
            // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
            DeviceUniquePointer<half> src0_as_f16(pool_ptr);
            if (type_0 != NumberType::FP16) {
                const size_t ne = ne01 * ne00;
                src0_as_f16.allocate(ne);
                cuda_op_convert_raw(ctx, NumberType::FP16, src0_as_f16.get(), type_0, operand_0.get(), ne);
            }
            const half *src0_ptr = (type_0 == NumberType::FP16) ?  operand_0.get<const half>() : src0_as_f16.get();

            DeviceUniquePointer<half> src1_as_f16(pool_ptr);
            if (type_0 != NumberType::FP16) {
                const size_t ne = ne11 * ne10;
                src1_as_f16.allocate(ne);
                cuda_op_convert_raw(ctx, NumberType::FP16, src1_as_f16.get(), type_1, operand_1.get(), ne);
            }
            const half *src1_ptr = (type_1 == NumberType::FP16) ?  operand_1.get<const half>() : src1_as_f16.get();

            DeviceUniquePointer<half> dst_f16(pool_ptr, ne01 * ne11);

            const half alpha_f16 = 1.0f;
            const half beta_f16  = 0.0f;

            gpu_check(cublasSetStream_v2(ctx.get_cublas_handle(), stream));
            gpu_check(cublasGemmEx(ctx.get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                &alpha_f16, src0_ptr,       CUDA_R_16F, ne00,
                            src1_ptr,       CUDA_R_16F, ne10,
                &beta_f16,   dst_f16.get(), CUDA_R_16F, ne0,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP)
            );

            cuda_op_convert_raw(ctx, NumberType::FP32, result.get(), NumberType::FP16, dst_f16.get(), ne01 * ne11);
        } else {
            DeviceUniquePointer<float> src0_as_f32(pool_ptr);
            if (type_0 != NumberType::FP32) {
                const size_t ne = ne01 * ne00;
                src0_as_f32.allocate(ne);
                cuda_op_convert_raw(ctx, NumberType::FP32, src0_as_f32.get(), type_0, operand_0.get(), ne);
            }
            const float *src0_ptr = (type_0 == NumberType::FP32) ?  operand_0.get<const float>() : src0_as_f32.get();

            DeviceUniquePointer<float> src1_as_f32(pool_ptr);
            if (type_0 != NumberType::FP32) {
                const size_t ne = ne11 * ne10;
                src1_as_f32.allocate(ne);
                cuda_op_convert_raw(ctx, NumberType::FP32, src1_as_f32.get(), type_1, operand_1.get(), ne);
            }
            const float *src1_ptr = (type_1 == NumberType::FP32) ?  operand_1.get<const float>() : src1_as_f32.get();

            const float alpha = 1.0f;
            const float beta  = 0.0f;

            gpu_check(cublasSetStream_v2(ctx.get_cublas_handle(), stream));
            gpu_check(cublasSgemm_v2(ctx.get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                &alpha, src0_ptr, ne00,
                src1_ptr, ne10,
                &beta, result.get<float>(), ne0)
            );
        }
    }


    void cuda_op_matmul(DeviceContext &ctx, const Tensor &result, const Tensor &operand_0, const Tensor &operand_1) {
        cuda_op_mul_mat_cublas(ctx, result, operand_0, operand_1, ctx.get_stream());
    }

} // namespace spy::gpu