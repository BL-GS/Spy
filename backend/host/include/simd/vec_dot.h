#pragma once

#include <immintrin.h>

#include "number/number.h"
#include "simd/vec_util.h"

namespace spy::cpu {

	template<NumberType T_lhs_type, NumberType T_rhs_type>
	struct Dot {
	public:
		static constexpr NumberType LhsType = T_lhs_type;
		static constexpr NumberType RhsType = T_rhs_type;

		using LhsBlock = BlockType<LhsType>;
		using RhsBlock = BlockType<RhsType>;

	public:
		float operator()(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) { return exec(lhs, rhs, num); }

		static float exec([[maybe_unused]]const LhsBlock *lhs, [[maybe_unused]]const RhsBlock *rhs, [[maybe_unused]]size_t num) {
			spy_abort("Unimplemented dot product: {} - {}", get_type_name(T_lhs_type), get_type_name(T_rhs_type));
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
		static float exec(const LhsBlock *lhs, const RhsBlock *rhs, size_t num_element) {
			using namespace simd;

			const size_t num_block = num_element / get_block_size(NumberType::Q8_0);

			// Initialize accumulator with zeros
			__m256 acc = _mm256_setzero_ps();

			// Main loop
			for (size_t i = 0; i < num_block; ++i) {
				// Compute combined scale for the block
				const __m256 d = _mm256_set1_ps( spy_fp16_to_fp32(lhs[i].delta) * spy_fp16_to_fp32(rhs[i].delta) );

				const __m128i low_mask = _mm_set1_epi8(0xF);
				const __m128i off = _mm_set1_epi8(8);

				const __m128i tmp = _mm_loadu_si128((const __m128i *)lhs[i].quants);

				__m128i bx_0 = _mm_and_si128(low_mask, tmp);
				__m128i by_0 = _mm_loadu_si128((const __m128i *)rhs[i].quants);
				bx_0 = _mm_sub_epi8(bx_0, off);
				const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

				bx_0 = _mm_and_si128(low_mask, _mm_srli_epi64(tmp, 4));
				by_0 = _mm_loadu_si128((const __m128i *)(rhs[i].quants + 16));
				bx_0 = _mm_sub_epi8(bx_0, off);
				const __m128i i32_1 = mul_sum_i8_pairs(bx_0, by_0);

				// Convert int32_t to float
				const __m256 p = _mm256_cvtepi32_ps(_mm256_set_m128i(i32_0, i32_1));

				// Apply the scale, and accumulate
				acc = _mm256_add_ps(_mm256_mul_ps( d, p ), acc);
			}

			return hsum_float_8(acc);
		}

		float operator()(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) { return exec(lhs, rhs, num); }

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
		static float exec(const LhsBlock *lhs, const RhsBlock *rhs, size_t num_element) {
			using namespace simd;

			const size_t num_block = num_element / get_block_size(NumberType::Q8_0);

			// Initialize accumulator with zeros
			__m256 acc = _mm256_setzero_ps();
			// Main loop
			for (size_t i = 0; i < num_block; ++i) {
				// Compute combined scale for the block
				const __m256  d  = _mm256_set1_ps(spy_fp16_to_fp32(lhs[i].delta) * spy_fp16_to_fp32(rhs[i].delta));
				const __m256i qx = _mm256_loadu_si256((const __m256i *)lhs[i].quants);
				const __m256i qy = _mm256_loadu_si256((const __m256i *)rhs[i].quants);
				const __m256  q  = mul_sum_i8_pairs_float(qx, qy);
				// Apply the scale, and accumulate
				acc = _mm256_fmadd_ps( d, q, acc );
			}

			return hsum_float_8(acc);
		}

		float operator()(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) { return exec(lhs, rhs, num); }

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
		static float exec(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) {
			using namespace simd;

			FP16::block_t sum[FP16::block_arr] = { FP16::block_zero() };
			FP16::block_t ax[FP16::block_arr];
			FP16::block_t ay[FP16::block_arr];

			size_t i = 0;
			for (; i + FP16::block_step < num; i += FP16::block_step) {
				for (int j = 0; j < FP16::block_arr; j++) {
					ax[j]  = FP16::block_load(lhs + i + j * FP16::block_len);
					ay[j]  = FP16::block_load(rhs + i + j * FP16::block_len);
					sum[j] = FP16::block_madd(ax[j], ay[j], sum[j]);
				}
			}
			const float simd_res = FP16::block_reduce(sum);
			// leftovers
			float normal_res = 0.0F;
			for (; i < num; ++i) { normal_res += spy_fp16_to_fp32(lhs[i]) * spy_fp16_to_fp32(rhs[i]); }

			return normal_res + simd_res;
		}

		float operator()(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) { return exec(lhs, rhs, num); }

		static float exec_raw(const void *lhs, const void *rhs, size_t num) {
			return exec(static_cast<const LhsBlock *>(lhs), static_cast<const RhsBlock *>(rhs), num);
		}
	};

	template<>
	struct Dot<NumberType::FP16, NumberType::FP32> {
	public:
		static constexpr NumberType LhsType = NumberType::FP32;
		static constexpr NumberType RhsType = NumberType::FP32;

		using LhsBlock = BlockType<LhsType>;
		using RhsBlock = BlockType<RhsType>;

	public:
		static float exec(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) {
			using namespace simd;

			FP32::block_t sum[FP32::block_arr] = { FP32::block_zero() };
			size_t i = 0;
			for (; i + FP32::block_step < num; i += FP32::block_step) {
				for (int k = 0; k < FP32::block_arr; ++k) {
					const FP32::block_t x_blk   = FP32::block_load(lhs + i + k * FP32::block_len);
					const FP32::block_t y_blk   = FP32::block_load(rhs + i + k * FP32::block_len);

					sum[k] = FP32::block_madd(x_blk, y_blk, sum[k]);
				}
			}
			const float simd_res   = FP32::block_reduce(sum);

			float normal_res = 0.0F;
			for (; i < num; ++i) { normal_res += lhs[i] * rhs[i]; }

			return simd_res + normal_res;
		}

		float operator()(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) { return exec(lhs, rhs, num); }

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
		static float exec(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) {
			using namespace simd;

			FP32::block_t sum[FP32::block_arr] = { FP32::block_zero() };
			size_t i = 0;
			for (; i + FP32::block_step < num; i += FP32::block_step) {
				for (int k = 0; k < FP32::block_arr; ++k) {
					const FP32::block_t x_blk   = FP32::block_load(lhs + i + k * FP32::block_len);
					const FP32::block_t y_blk   = FP32::block_load(rhs + i + k * FP32::block_len);

					sum[k] = FP32::block_madd(x_blk, y_blk, sum[k]);
				}
			}
			const float simd_res   = FP32::block_reduce(sum);

			float normal_res = 0.0F;
			for (; i < num; ++i) { normal_res += lhs[i] * rhs[i]; }

			return simd_res + normal_res;
		}

		float operator()(const LhsBlock *lhs, const RhsBlock *rhs, size_t num) { return exec(lhs, rhs, num); }

		static float exec_raw(const void *lhs, const void *rhs, size_t num) {
			return exec(static_cast<const LhsBlock *>(lhs), static_cast<const RhsBlock *>(rhs), num);
		}
	};

} // namespace spy