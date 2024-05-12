/*
 * @author: BL-GS 
 * @date:   24-4-13
 */

#pragma once

#include <numeric>

#include "util/logger.h"
#include "util/simd.h"
#include "number/number.h"
#include "number/compute/util.h"

namespace spy {

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
			// Initialize accumulator with zeros
			__m256 acc = _mm256_setzero_ps();

			// Main loop
			for (size_t i = 0; i < num / get_block_size(NumberType::Q8_0); ++i) {
				// Compute combined scale for the block
				const __m256 d = _mm256_set1_ps( spy_fp16_to_fp32(lhs[i].delta) * spy_fp16_to_fp32(rhs[i].delta) );

				const __m128i lowMask = _mm_set1_epi8(0xF);
				const __m128i off = _mm_set1_epi8(8);

				const __m128i tmp = _mm_loadu_si128((const __m128i *)lhs[i].quants);

				__m128i bx_0 = _mm_and_si128(lowMask, tmp);
				__m128i by_0 = _mm_loadu_si128((const __m128i *)rhs[i].quants);
				bx_0 = _mm_sub_epi8(bx_0, off);
				const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

				bx_0 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp, 4));
				by_0 = _mm_loadu_si128((const __m128i *)(rhs[i].quants + 16));
				bx_0 = _mm_sub_epi8(bx_0, off);
				const __m128i i32_1 = mul_sum_i8_pairs(bx_0, by_0);

				// Convert int32_t to float
				__m256 p = _mm256_cvtepi32_ps(_mm256_set_m128i(i32_0, i32_1));

				// Apply the scale, and accumulate
				acc = _mm256_add_ps(_mm256_mul_ps( d, p ), acc);
			}

			return hsum_float_8(acc);
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
			// Initialize accumulator with zeros
			__m256 acc = _mm256_setzero_ps();

			// Main loop
			for (size_t i = 0; i < num / get_block_size(NumberType::Q8_0); ++i) {
				// Compute combined scale for the block
				const __m256 d = _mm256_set1_ps(spy_fp16_to_fp32(lhs[i].delta) * spy_fp16_to_fp32(rhs[i].delta));
				__m256i qx = _mm256_loadu_si256((const __m256i *)lhs[i].quants);
				__m256i qy = _mm256_loadu_si256((const __m256i *)rhs[i].quants);

				const __m256 q = mul_sum_i8_pairs_float(qx, qy);
				// Apply the scale, and accumulate
				acc = _mm256_fmadd_ps( d, q, acc );
			}

			return hsum_float_8(acc);
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
			float res = 0.0F;
			// Initialize accumulator with zeros
			const int np = static_cast<int>(num & (~31));

			SPY_SIMD_F16_VEC sum[SPY_SIMD_F16_ARR] = { SPY_SIMD_F16_VEC_ZERO };
			SPY_SIMD_F16_VEC ax[SPY_SIMD_F16_ARR];
			SPY_SIMD_F16_VEC ay[SPY_SIMD_F16_ARR];

			for (int i = 0; i < np; i += SPY_SIMD_F16_STEP) {
				for (int j = 0; j < SPY_SIMD_F16_ARR; j++) {
					ax[j]  = SPY_SIMD_F16_VEC_LOAD(const_cast<LhsBlock *>(lhs) + i + j * SPY_SIMD_F16_EPR, j);
					ay[j]  = SPY_SIMD_F16_VEC_LOAD(const_cast<RhsBlock *>(rhs) + i + j * SPY_SIMD_F16_EPR, j);
					sum[j] = SPY_SIMD_F16_VEC_FMA(sum[j], ax[j], ay[j]);
				}
			}

			// reduce sum0..sum3 to sum0
			SPY_SIMD_F16_VEC_REDUCE(res, sum);

			// leftovers
			for (size_t i = np; i < num; ++i) {
				res += spy_fp16_to_fp32(lhs[i]) * spy_fp16_to_fp32(rhs[i]);
			}

			return res;
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

}  // namespace spy