#include <cstddef>
#include <cuda_runtime.h>
#include <magic_enum_fuse.hpp>

#include "util/shell/logger.h"
#include "number/tensor.h"
#include "gpu_device.h"
#include "operator/detail/binary_op.h"

namespace spy::gpu {

    static __device__ __forceinline__ float op_repeat(const float a, const float b) { return b; }

    static __device__ __forceinline__ float op_add(const float a, const float b) { return a + b; }

    static __device__ __forceinline__ float op_sub(const float a, const float b) { return a - b; }

    static __device__ __forceinline__ float op_mul(const float a, const float b) { return a * b; }

    static __device__ __forceinline__ float op_div(const float a, const float b) { return a / b; }

    template<float (*T_Func)(const float, const float), typename T_Operand_0, typename T_Operand_1, typename T_Result>
    static __global__ void k_bin_bcast(const T_Operand_0 * src0, const T_Operand_1 * src1, T_Result * dst,
            int ne0, int ne1, int ne2, int ne3,
            int ne10, int ne11, int ne12, int ne13,
            /*int s0, */ int s1,  int s2,  int s3,
            /*int s00,*/ int s01, int s02, int s03,
            /*int s10,*/ int s11, int s12, int s13) {
        const int i0s = blockDim.x * blockIdx.x + threadIdx.x;
        const int i1 = (blockDim.y * blockIdx.y + threadIdx.y);
        const int i2 = (blockDim.z * blockIdx.z + threadIdx.z) / ne3;
        const int i3 = (blockDim.z * blockIdx.z + threadIdx.z) % ne3;

        if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
            return;
        }

        const int i11 = i1 % ne11;
        const int i12 = i2 % ne12;
        const int i13 = i3 % ne13;

        const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
        const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
        const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

        const T_Operand_0 * src0_row = src0 + i_src0;
        const T_Operand_1 * src1_row = src1 + i_src1;
        T_Result * dst_row = dst + i_dst;

        for (int i0 = i0s; i0 < ne0; i0 += blockDim.x*gridDim.x) {
            const int i10 = i0 % ne10;
            dst_row[i0] = static_cast<T_Result>(T_Func(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]));
        }
    }

    template<float (*T_Func)(const float, const float), typename T_Operand_0, typename T_Operand_1, typename T_Result>
    static __global__ void k_bin_bcast_unravel(const T_Operand_0 * src0, const T_Operand_1 * src1, T_Result * dst,
            int ne0, int ne1, int ne2, int ne3,
            int ne10, int ne11, int ne12, int ne13,
            /*int s0, */ int s1,  int s2,  int s3,
            /*int s00,*/ int s01, int s02, int s03,
            /*int s10,*/ int s11, int s12, int s13) {

        const int i = blockDim.x*blockIdx.x + threadIdx.x;

        const int i3 = i / (ne1 * ne0) / ne2;
        const int i2 = i / (ne1 * ne0) % ne2;
        const int i1 = i / ne0 % ne1;
        const int i0 = i % ne0;

        if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
            return;
        }

        const int i11 = i1 % ne11;
        const int i12 = i2 % ne12;
        const int i13 = i3 % ne13;

        const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
        const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
        const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

        const T_Operand_0 * src0_row = src0 + i_src0;
        const T_Operand_1 * src1_row = src1 + i_src1;
        T_Result * dst_row = dst + i_dst;

        const int i10 = i0 % ne10;
        dst_row[i0] = static_cast<T_Result>(T_Func(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]));
    }

    template<float (*T_Func)(const float, const float)>
    struct bin_bcast_cuda {

        template<typename T_Result, typename T_Operand_0, typename T_Operand_1>
        static void execute(const Tensor &result, const Tensor &operand_0, const Tensor &operand_1, 
            T_Result *data_res, T_Operand_0 *data_0, T_Operand_1 *data_1,
            cudaStream_t stream) {

            using DimensionArray = Tensor::DimensionArray;

            // collapse dimensions until first broadcast dimension
            auto cne  = result.element_array();
            auto cne0 = operand_0.element_array();
            auto cne1 = operand_1.element_array();

            auto cnb    = result.size_array();
            auto cnb0   = operand_0.size_array();
            auto cnb1   = operand_1.size_array();

            DimensionArray nr;
            std::transform(cne1.begin(), cne1.end(), cne.begin(), nr.begin(), std::divides<size_t>());

            constexpr auto collapse = [](DimensionArray &cne) {
                cne[0] *= cne[1];
                cne[1] = cne[2];
                cne[2] = cne[3];
                cne[3] = 1;
            };

            constexpr auto collapse_nb = [](DimensionArray &cnb, const DimensionArray &cne) {
                cnb[1] *= cne[1];
                cnb[2] *= cne[2];
                cnb[3] *= cne[3];
            };

            if (result.is_continuous() && operand_0.is_continuous() && operand_1.is_continuous()) {
                for (int i = 0; i < 4; i++) {
                    if (nr[i] != 1) {
                        break;
                    }
                    if (i > 0) {
                        collapse_nb(cnb, cne);
                        collapse_nb(cnb0, cne0);
                        collapse_nb(cnb1, cne1);
                        collapse(cne);
                        collapse(cne0);
                        collapse(cne1);
                    }
                }
            }

            {
                const auto [ne0, ne1, ne2, ne3]     = cne;
                const auto [ne10, ne11, ne12, ne13] = cne1;
                const auto [nb0, nb1, nb2, nb3]     = cnb;
                const auto [nb00, nb01, nb02, nb03] = cnb0;
                const auto [nb10, nb11, nb12, nb13] = cnb1;

                const size_t s0 = nb0 / sizeof(T_Result);
                const size_t s1 = nb1 / sizeof(T_Result);
                const size_t s2 = nb2 / sizeof(T_Result);
                const size_t s3 = nb3 / sizeof(T_Result);

                const size_t s10 = nb10 / sizeof(T_Operand_1);
                const size_t s11 = nb11 / sizeof(T_Operand_1);
                const size_t s12 = nb12 / sizeof(T_Operand_1);
                const size_t s13 = nb13 / sizeof(T_Operand_1);

                const size_t s00 = nb00 / sizeof(T_Operand_0);
                const size_t s01 = nb01 / sizeof(T_Operand_0);
                const size_t s02 = nb02 / sizeof(T_Operand_0);
                const size_t s03 = nb03 / sizeof(T_Operand_0);

                spy_assert(nb0 % sizeof(T_Result) == 0);
                spy_assert(nb1 % sizeof(T_Result) == 0);
                spy_assert(nb2 % sizeof(T_Result) == 0);
                spy_assert(nb3 % sizeof(T_Result) == 0);

                spy_assert(nb00 % sizeof(T_Operand_0) == 0);
                spy_assert(nb01 % sizeof(T_Operand_0) == 0);
                spy_assert(nb02 % sizeof(T_Operand_0) == 0);
                spy_assert(nb03 % sizeof(T_Operand_0) == 0);

                spy_assert(nb10 % sizeof(T_Operand_1) == 0);
                spy_assert(nb11 % sizeof(T_Operand_1) == 0);
                spy_assert(nb12 % sizeof(T_Operand_1) == 0);
                spy_assert(nb13 % sizeof(T_Operand_1) == 0);

                spy_assert(s0 == 1);
                spy_assert(s00 == 1);
                spy_assert(s10 == 1);

                const int block_size = 128;

                const size_t hne0 = std::max(ne0 / 2ULL, 1ULL);

                dim3 block_dims;
                block_dims.x = std::min<unsigned int>(hne0, block_size);
                block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
                block_dims.z = std::min(std::min<unsigned int>(ne2*ne3, block_size / block_dims.x / block_dims.y), 64U);

                dim3 block_nums(
                    (hne0 + block_dims.x - 1) / block_dims.x,
                    (ne1 + block_dims.y - 1) / block_dims.y,
                    (ne2*ne3 + block_dims.z - 1) / block_dims.z
                );

                if (block_nums.z > 65535) {
                    // this is the maximum number of blocks in z dimension, fallback to 1D grid kernel
                    const int block_num = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
                    k_bin_bcast_unravel<T_Func><<<block_num, block_size, 0, stream>>>(
                        data_0, data_1, data_res,
                        ne0, ne1, ne2, ne3,
                        ne10, ne11, ne12, ne13,
                        /* s0, */ s1, s2, s3,
                        /* s00, */ s01, s02, s03,
                        /* s10, */ s11, s12, s13);
                } else {
                    k_bin_bcast<T_Func><<<block_nums, block_dims, 0, stream>>>(
                        data_0, data_1, data_res,
                        ne0, ne1, ne2, ne3,
                        ne10, ne11, ne12, ne13,
                        /* s0, */ s1, s2, s3,
                        /* s00, */ s01, s02, s03,
                        /* s10, */ s11, s12, s13);
                }
            }
        }
    };

    template<class T_Operator>
    static void cuda_op_bin_bcast(
        const Tensor &result, const Tensor &operand_0, const Tensor &operand_1, cudaStream_t stream) {

        const NumberType type_res = result.get_number_type();
        const NumberType type_0   = operand_0.get_number_type();
        const NumberType type_1   = operand_1.get_number_type();

        void *data_res = result.get();
        const void *data_0   = operand_0.get();
        const void *data_1   = operand_1.get();

        spy_assert(type_1 == NumberType::FP32);

        using magic_enum::enum_fuse;
        switch (enum_fuse(type_0, type_res).value()) {
        case enum_fuse(NumberType::FP32, NumberType::FP32).value():
            T_Operator::execute(operand_0, operand_1, result, 
                static_cast<float *>(data_res), static_cast<const float *>(data_0), static_cast<const float *>(data_1),
                stream
            );
            break;
        case enum_fuse(NumberType::FP16, NumberType::FP16).value():
            T_Operator::execute(operand_0, operand_1, result, 
                static_cast<half *>(data_res), static_cast<const half *>(data_0), static_cast<const float *>(data_1),
                stream
            );
            break;
        case enum_fuse(NumberType::FP16, NumberType::FP32).value():
            T_Operator::execute(operand_0, operand_1, result, 
                static_cast<float *>(data_res), static_cast<const half *>(data_0), static_cast<const float *>(data_1),
                stream
            );
            break;
        default:
            spy_abort("Unsupported type: result: {}, operand_0: {}, operand_1: {}", type_res, type_0, type_1);
        }
    }

    void cuda_op_repeat(DeviceContext &ctx, const Tensor &result, const Tensor &operand_0, const Tensor &operand_1) {
        cuda_op_bin_bcast<bin_bcast_cuda<op_repeat>>(result, operand_0, operand_1, ctx.get_stream());
    }

    void cuda_op_add(DeviceContext &ctx, const Tensor &result, const Tensor &operand_0, const Tensor &operand_1) {
        cuda_op_bin_bcast<bin_bcast_cuda<op_add>>(result, operand_0, operand_1, ctx.get_stream());
    }

    void cuda_op_sub(DeviceContext &ctx, const Tensor &result, const Tensor &operand_0, const Tensor &operand_1) {
        cuda_op_bin_bcast<bin_bcast_cuda<op_sub>>(result, operand_0, operand_1, ctx.get_stream());
    }

    void cuda_op_mul(DeviceContext &ctx, const Tensor &result, const Tensor &operand_0, const Tensor &operand_1) {
        cuda_op_bin_bcast<bin_bcast_cuda<op_mul>>(result, operand_0, operand_1, ctx.get_stream());
    }

    void cuda_op_div(DeviceContext &ctx, const Tensor &result, const Tensor &operand_0, const Tensor &operand_1) {
        cuda_op_bin_bcast<bin_bcast_cuda<op_div>>(result, operand_0, operand_1, ctx.get_stream());
    }

} // namespace spy::gpu

