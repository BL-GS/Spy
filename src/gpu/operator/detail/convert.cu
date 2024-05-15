#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <magic_enum.hpp>
#include <magic_enum_fuse.hpp>

#include "util/shell/logger.h"
#include "operator/detail/convert.h"

namespace spy::gpu {

    constexpr int CUDA_DEQUANTIZE_BLOCK_SIZE = 256;

    template<typename T_Result>
    static __global__ void dequantize_block_q4_0(T_Result * __restrict__ yy, const block_q4_0_t * __restrict__ vx, int nb32) {

        const int64_t i = blockIdx.x;

        // assume 32 threads
        const int64_t tid = threadIdx.x;
        const int64_t il  = tid / 8;
        const int64_t ir  = tid % 8;
        const int64_t ib  = 8 * i + ir;
        if (ib >= nb32) { return; }

        T_Result * y = yy + 256 * i + 32 * ir + 4 * il;

        const block_q4_0_t * x  = static_cast<const block_q4_0_t *>(vx) + ib;
        const float delta       = __half2float(x->delta);
        const float delta_min   = -8 * delta;

        const uint8_t * q = x->quants + 4 * il;

        for (int l = 0; l < 4; ++l) {
            y[l+ 0] = delta * (q[l] & 0xF) + delta_min;
            y[l+16] = delta * (q[l] >>  4) + delta_min;
        }
    }

    template<typename T_Result>
    static void dequantize_row_q4_0_cuda(T_Result * y, const block_q4_0_t * vx, const size_t k, cudaStream_t stream) {
        const int nb32 = k / 32;
        const int nb   = (k + 255) / 256;
        dequantize_block_q4_0<<<nb, 32, 0, stream>>>(y, vx, nb32);
    }

    template <typename T_Result, typename T_Operand>
    static __global__ void convert_unary(T_Result * __restrict__ y, const T_Operand * __restrict__ vx, const size_t k) {
        const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= k) { return; }

        y[i] = vx[i];
    }

    template <typename T_Result, typename T_Operand>
    static void convert_unary_cuda(T_Result * __restrict__ y, const T_Operand * __restrict__ vx, const size_t k, cudaStream_t stream) {
        const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
        convert_unary<T_Result, T_Operand><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(y, vx, k);
    }


    void cuda_op_convert_raw(DeviceContext &ctx, NumberType dst_type, void *dst_ptr, NumberType src_type, void *src_ptr, const size_t num) {
        using magic_enum::enum_name;
        using magic_enum::enum_fuse;

        cudaStream_t stream = ctx.get_stream();

        switch (enum_fuse(dst_type, src_type).value()) {
        case enum_fuse(NumberType::FP32, NumberType::FP16).value():
            convert_unary_cuda(static_cast<float *>(dst_ptr), static_cast<const half *>(src_ptr), num, stream);
            break;
        case enum_fuse(NumberType::FP16, NumberType::FP32).value():
            convert_unary_cuda(static_cast<half *>(dst_ptr), static_cast<const float *>(src_ptr), num, stream);
            break;
        case enum_fuse(NumberType::FP32, NumberType::Q4_0).value():
            dequantize_row_q4_0_cuda(static_cast<float *>(dst_ptr), static_cast<const block_q4_0_t *>(src_ptr), num, stream);
            break;
        case enum_fuse(NumberType::FP16, NumberType::Q4_0).value():
            dequantize_row_q4_0_cuda(static_cast<half *>(dst_ptr), static_cast<const block_q4_0_t *>(src_ptr), num, stream);
            break;
        default:
            spy_assert(false, "Unsupport type for convert: dst: {}; src: {}",
                enum_name(dst_type), enum_name(src_type)
            );
        }
    }

    void cuda_op_convert(DeviceContext &ctx, const Tensor &result, const Tensor &operand) {
        const size_t num = operand.total_element();
        cuda_op_convert_raw(ctx, 
            result.get_number_type(), result.get(), 
            operand.get_number_type(), operand.get(),
            num
        );
    }

} // namespace spy::gpu