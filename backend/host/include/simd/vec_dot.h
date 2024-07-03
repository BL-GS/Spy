#pragma once

#include <immintrin.h>
#include <numeric>

#include "number/number.h"
#include "simd/vec_util.h"

namespace spy::cpu {

    inline float vec_dot_fp32(const float *x, const float *y, const int num) {
        using namespace simd;

        const int outer_end = num / FP32::block_step;
        const int inner_end = FP32::block_step / FP32::block_len;

        FP32::block_t sum[FP32::block_arr] = { FP32::block_zero() };
        int i = 0;
        for (; i + FP32::block_step < num; i += FP32::block_step) {
            #pragma unroll
            for (int k = 0; k < FP32::block_arr; ++i) {
                const FP32::block_t x_blk   = FP32::block_load(x + i + k * FP32::block_len);
                const FP32::block_t y_blk   = FP32::block_load(y + i + k * FP32::block_len);

                sum[k] = FP32::block_madd(x_blk, y_blk, sum[k]);
            }
        }
        const float simd_res   = FP32::block_reduce(sum);

        float normal_res = 0.0F;
        for (; i < num; ++i) { normal_res += x[i] * y[i]; }

        return simd_res + normal_res;
    }
 
    inline float vec_dot_fp16(const uint16_t *x, const uint16_t *y, const int num) {
        using namespace simd;

        FP16::block_t sum[FP16::block_arr] = { FP16::block_zero() };
        FP16::block_t ax[FP16::block_arr];
        FP16::block_t ay[FP16::block_arr];

        int i = 0;
        for (; i + FP16::block_step < num; i += FP16::block_step) {
            for (int j = 0; j < FP16::block_arr; j++) {
                ax[j]  = FP16::block_load(x + i + j * FP16::block_len);
                ay[j]  = FP16::block_load(y + i + j * FP16::block_len);
                sum[j] = FP16::block_madd(ax[j], ay[j], sum[j]);
            }
        }
        const float simd_res = FP16::block_reduce(sum);
        // leftovers
        float normal_res = 0.0F;
        for (; i < num; ++i) { normal_res += spy_fp16_to_fp32(x[i]) * spy_fp16_to_fp32(y[i]); }

        return normal_res + simd_res;
    }

    inline float vec_dot_q8_0(const block_q8_0_t *x, const block_q8_0_t *y, const int num) {
        using namespace simd;

        // Initialize accumulator with zeros
        __m256 acc = _mm256_setzero_ps();
        // Main loop
        for (size_t i = 0; i < num / get_block_size(NumberType::Q8_0); ++i) {
            // Compute combined scale for the block
            const __m256  d  = _mm256_set1_ps(spy_fp16_to_fp32(x[i].delta) * spy_fp16_to_fp32(y[i].delta));
            const __m256i qx = _mm256_loadu_si256((const __m256i *)x[i].quants);
            const __m256i qy = _mm256_loadu_si256((const __m256i *)y[i].quants);
            const __m256  q  = mul_sum_i8_pairs_float(qx, qy);
            // Apply the scale, and accumulate
            acc = _mm256_fmadd_ps( d, q, acc );
        }

        return hsum_float_8(acc);
    }

    inline float vec_dot_q4_0_q8_0(const block_q4_0_t *x, const block_q8_0_t *y, const int num) {
        using namespace simd;

        // Initialize accumulator with zeros
        __m256 acc = _mm256_setzero_ps();

        // Main loop
        for (size_t i = 0; i < num / get_block_size(NumberType::Q8_0); ++i) {
            // Compute combined scale for the block
            const __m256 d = _mm256_set1_ps( spy_fp16_to_fp32(x[i].delta) * spy_fp16_to_fp32(y[i].delta) );

            const __m128i lowMask = _mm_set1_epi8(0xF);
            const __m128i off = _mm_set1_epi8(8);

            const __m128i tmp = _mm_loadu_si128((const __m128i *)x[i].quants);

            __m128i bx_0 = _mm_and_si128(lowMask, tmp);
            __m128i by_0 = _mm_loadu_si128((const __m128i *)y[i].quants);
            bx_0 = _mm_sub_epi8(bx_0, off);
            const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

            bx_0 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp, 4));
            by_0 = _mm_loadu_si128((const __m128i *)(y[i].quants + 16));
            bx_0 = _mm_sub_epi8(bx_0, off);
            const __m128i i32_1 = mul_sum_i8_pairs(bx_0, by_0);

            // Convert int32_t to float
            const __m256 p = _mm256_cvtepi32_ps(_mm256_set_m128i(i32_0, i32_1));

            // Apply the scale, and accumulate
            acc = _mm256_add_ps(_mm256_mul_ps( d, p ), acc);
        }

        return hsum_float_8(acc);
    }


	template<NumberType T_lhs_type, NumberType T_rhs_type>
	struct Dot {
	public:
		static constexpr NumberType LhsType = T_lhs_type;
		static constexpr NumberType RhsType = T_rhs_type;

		using LhsBlock = BlockType<LhsType>;
		using RhsBlock = BlockType<RhsType>;

	public:
		float operator()(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) { return exec(lhs, rhs, num); }

		static float exec(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) {
			spy_assert(false, "Unimplemented dot product: {} - {}", get_type_name(T_lhs_type), get_type_name(T_rhs_type));
			return 0.0F;
		}

		static float exec_raw(const void *lhs, const void *rhs, size_t num) {
			return exec(static_cast<const LhsBlock *>(lhs), static_cast<const RhsBlock *>(rhs), num);
		}
	};

	template<>
	struct Dot<NumberType::Q4_0, NumberType::Q8_0> {
	public:
		static constexpr NumberType LhsType = NumberType::Q4_0;
		static constexpr NumberType RhsType = NumberType::Q8_0;

		using LhsBlock = BlockType<LhsType>;
		using RhsBlock = BlockType<RhsType>;

	public:
		float operator()(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) { return exec(lhs, rhs, num); }

		static float exec(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) {
            return vec_dot_q4_0_q8_0(lhs, rhs, num);
		}

		static float exec_raw(const void *lhs, const void *rhs, size_t num) {
			return exec(static_cast<const LhsBlock *>(lhs), static_cast<const RhsBlock *>(rhs), num);
		}
	};

	template<>
	struct Dot<NumberType::Q8_0, NumberType::Q8_0> {
	public:
		static constexpr NumberType LhsType = NumberType::Q8_0;
		static constexpr NumberType RhsType = NumberType::Q8_0;

		using LhsBlock = BlockType<LhsType>;
		using RhsBlock = BlockType<RhsType>;

	public:
		float operator()(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) { return exec(lhs, rhs, num); }

		static float exec(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) {
            return vec_dot_q8_0(lhs, rhs, num);
		}

		static float exec_raw(const void *lhs, const void *rhs, size_t num) {
			return exec(static_cast<const LhsBlock *>(lhs), static_cast<const RhsBlock *>(rhs), num);
		}
	};

	template<>
	struct Dot<NumberType::FP16, NumberType::FP16> {
	public:
		static constexpr NumberType LhsType = NumberType::FP16;
		static constexpr NumberType RhsType = NumberType::FP16;

		using LhsBlock = BlockType<LhsType>;
		using RhsBlock = BlockType<RhsType>;

	public:
		float operator()(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) { return exec(lhs, rhs, num); }

		static float exec(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) {
            return vec_dot_fp16(lhs, rhs, num);
		}

		static float exec_raw(const void *lhs, const void *rhs, size_t num) {
			return exec(static_cast<const LhsBlock *>(lhs), static_cast<const RhsBlock *>(rhs), num);
		}
	};


	template<>
	struct Dot<NumberType::FP32, NumberType::FP32> {
	public:
		static constexpr NumberType LhsType = NumberType::FP32;
		static constexpr NumberType RhsType = NumberType::FP32;

		using LhsBlock = BlockType<LhsType>;
		using RhsBlock = BlockType<RhsType>;

	public:
		float operator()(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) { return exec(lhs, rhs, num); }

		static float exec(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) {
			return vec_dot_fp32(lhs, rhs, num);
		}

		static float exec_raw(const void *lhs, const void *rhs, size_t num) {
			return exec(static_cast<const LhsBlock *>(lhs), static_cast<const RhsBlock *>(rhs), num);
		}
	};

	template<NumberType T_lhs_type, NumberType T_rhs_type>
		requires (T_lhs_type == T_rhs_type && is_trivial(T_lhs_type))
	struct Dot<T_lhs_type, T_rhs_type> {
	public:
		static constexpr NumberType LhsType = T_lhs_type;
		static constexpr NumberType RhsType = T_rhs_type;

		using LhsBlock = BlockType<LhsType>;
		using RhsBlock = BlockType<RhsType>;

	public:
		float operator()(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) { return exec(lhs, rhs, num); }

		static float exec(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) {
			return std::transform_reduce(lhs, lhs + num, rhs, 0, std::multiplies<LhsBlock>(), std::plus<LhsBlock>());
		}

		static float exec_raw(const void *lhs, const void *rhs, size_t num) {
			return exec(static_cast<const LhsBlock *>(lhs), static_cast<const RhsBlock *>(rhs), num);
		}
	};

} // namespace spy