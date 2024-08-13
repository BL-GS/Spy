#pragma once

#include <simde/x86/avx2.h>

#include "number/number.h"
#include "simd/vec_util.h"

namespace spy::cpu {

	namespace detail {

		template<class T_Impl>
		struct DotImpl {
			static constexpr NumberType LhsType = T_Impl::LhsType;
			static constexpr NumberType RhsType = T_Impl::RhsType;

			using LhsBlock = BlockType<LhsType>;
			using RhsBlock = BlockType<RhsType>;

			float operator()(const LhsBlock *lhs, const RhsBlock *rhs, int64_t num) {
				return T_Impl::exec(lhs, rhs, num);
			}

			static float exec_raw(const void *lhs, const void *rhs, int64_t num) {
				return T_Impl::exec(static_cast<const LhsBlock *>(lhs), static_cast<const RhsBlock *>(rhs), num);
			}
		};

		template<NumberType T_lhs_type, NumberType T_rhs_type>
		struct Dot {
		public:
			static constexpr NumberType LhsType = T_lhs_type;
			static constexpr NumberType RhsType = T_rhs_type;

		public:
			static float exec([[maybe_unused]]const auto *lhs, [[maybe_unused]]const auto *rhs, [[maybe_unused]]int64_t num) {
				spy_abort("Unimplemented dot product: {} - {}", get_type_name(T_lhs_type), get_type_name(T_rhs_type));
				return 0.0F;
			}
		};

		template<>
		struct Dot<NumberType::Q4_0, NumberType::Q8_0> {
		public:
			static constexpr NumberType LhsType = NumberType::Q4_0;
			static constexpr NumberType RhsType = NumberType::Q8_0;

		public:
			static float exec(const block_q4_0_t *lhs, const block_q8_0_t *rhs, int64_t num_element) {
				using namespace simd;

				const int64_t num_block = num_element / get_block_size(NumberType::Q8_0);

				// Initialize accumulator with zeros
				simde__m256 acc = simde_mm256_setzero_ps();

				// Main loop
				for (int64_t i = 0; i < num_block; ++i) {
					// Compute combined scale for the block
					const simde__m256 d         = simde_mm256_set1_ps( spy_fp16_to_fp32(lhs[i].delta) * spy_fp16_to_fp32(rhs[i].delta) );

					const simde__m128i low_mask = simde_mm_set1_epi8(0xF);
					const simde__m128i off      = simde_mm_set1_epi8(8);

					const simde__m128i tmp      = simde_mm_loadu_si128(lhs[i].quants);

					simde__m128i bx_0           = simde_mm_and_si128(low_mask, tmp);
					simde__m128i by_0           = simde_mm_loadu_si128(rhs[i].quants);
					bx_0                        = simde_mm_sub_epi8(bx_0, off);
					const simde__m128i i32_0    = mul_sum_i8_pairs(bx_0, by_0);

					bx_0 = simde_mm_and_si128(low_mask, simde_mm_srli_epi64(tmp, 4));
					by_0 = simde_mm_loadu_si128(rhs[i].quants + 16);
					bx_0 = simde_mm_sub_epi8(bx_0, off);
					const simde__m128i i32_1 = mul_sum_i8_pairs(bx_0, by_0);

					// Convert int32_t to float
					const simde__m256 p = simde_mm256_cvtepi32_ps(simde_mm256_set_m128i(i32_0, i32_1));

					// Apply the scale, and accumulate
					acc = simde_mm256_fmadd_ps(d, p, acc);
				}

				return hsum_float_8(acc);
			}
		};

		template<>
		struct Dot<NumberType::Q8_0, NumberType::Q8_0> {
		public:
			static constexpr NumberType LhsType = NumberType::Q8_0;
			static constexpr NumberType RhsType = NumberType::Q8_0;

		public:
			static float exec(const block_q8_0_t *lhs, const block_q8_0_t *rhs, int64_t num_element) {
				using namespace simd;

				const int64_t num_block = num_element / get_block_size(NumberType::Q8_0);

				// Initialize accumulator with zeros
				simde__m256 acc = simde_mm256_setzero_ps();
				// Main loop
				for (int64_t i = 0; i < num_block; ++i) {
					// Compute combined scale for the block
					const simde__m256  d  = simde_mm256_set1_ps(spy_fp16_to_fp32(lhs[i].delta) * spy_fp16_to_fp32(rhs[i].delta));
					const simde__m256i qx = simde_mm256_loadu_si256(lhs[i].quants);
					const simde__m256i qy = simde_mm256_loadu_si256(rhs[i].quants);
					const simde__m256  q  = mul_sum_i8_pairs_float(qx, qy);
					// Apply the scale, and accumulate
					acc = simde_mm256_fmadd_ps(d, q, acc);
				}

				return hsum_float_8(acc);
			}
		};

		template<>
		struct Dot<NumberType::FP16, NumberType::FP16> {
		public:
			static constexpr NumberType LhsType = NumberType::FP16;
			static constexpr NumberType RhsType = NumberType::FP16;

		public:
			static float exec(const uint16_t *lhs, const uint16_t *rhs, int64_t num) {
				using namespace simd;

				FP16::block_t sum[FP16::block_arr] = { FP16::block_zero() };
				FP16::block_t ax[FP16::block_arr];
				FP16::block_t ay[FP16::block_arr];

				int64_t i = 0;
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
		};

		template<>
		struct Dot<NumberType::FP16, NumberType::FP32> {
		public:
			static constexpr NumberType LhsType = NumberType::FP16;
			static constexpr NumberType RhsType = NumberType::FP32;

		public:
			static float exec(const uint16_t *lhs, const float *rhs, int64_t num) {
				using namespace simd;

				FP32::block_t sum[FP32::block_arr] = { FP32::block_zero() };
				int i = 0;
				for (; i + FP32::block_step < num; i += FP32::block_step) {
					for (int k = 0; k < FP32::block_arr; ++k) {
						const FP16::block_t x_blk   = FP16::block_load(lhs + i + k * FP16::block_len);
						const FP32::block_t y_blk   = FP32::block_load(rhs + i + k * FP32::block_len);

						sum[k] = FP32::block_madd(x_blk, y_blk, sum[k]);
					}
				}
				const float simd_res   = FP32::block_reduce(sum);

				float normal_res = 0.0F;
				for (; i < num; ++i) { normal_res += spy_fp16_to_fp32(lhs[i]) * rhs[i]; }

				return simd_res + normal_res;
			}
		};


		template<>
		struct Dot<NumberType::FP32, NumberType::FP32> {
		public:
			static constexpr NumberType LhsType = NumberType::FP32;
			static constexpr NumberType RhsType = NumberType::FP32;

		public:
			static float exec(const float *lhs, const float *rhs, int64_t num) {
				using namespace simd;

				FP32::block_t sum[FP32::block_arr] = { FP32::block_zero() };
				int i = 0;
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
		};

	} // namespace detail

	template<NumberType T_lhs_type, NumberType T_rhs_type>
	using Dot = detail::DotImpl<detail::Dot<T_lhs_type, T_rhs_type>>;

}  // namespace spy::cpu