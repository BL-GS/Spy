#pragma once

#include <immintrin.h>

#include "util/logger.h"
#include "number/number.h"
#include "number/compute/util.h"
#include "number/quantization_impl/type.h"

namespace spy {
	
	template<>
	struct Quantizator<NumberType::FP32, NumberType::FP16> {
	public:
		static constexpr NumberType FromType = NumberType::FP32;
		static constexpr NumberType ToType	 = NumberType::FP16;

		using FromMetadata              = NumberMetadata<FromType>;
		using ToMetadata                = NumberMetadata<ToType>;
		using FromBlock             	= BlockType<FromType>;
		using ToBlock               	= BlockType<ToType>;

		static constexpr size_t FROM_BLOCK_SIZE = FromMetadata::BLOCK_SIZE;
		static constexpr size_t TO_BLOCK_SIZE   = ToMetadata::BLOCK_SIZE;
		static constexpr size_t FROM_TYPE_SIZE  = FromMetadata::TYPE_SIZE;
		static constexpr size_t TO_TYPE_SIZE    = ToMetadata::TYPE_SIZE;

	public:
		static void transform(const FromBlock * __restrict from_ptr, ToBlock * __restrict to_ptr, size_t num_from) {
			for (size_t i = 0; i < num_from; ++i) {
				to_ptr[i] = spy_fp32_to_fp16(from_ptr[i]);
			}
			// int64_t i = 0;
			// for (; i + 7 < num_from; i += 8) {
			// 	__m256 x_vec = _mm256_loadu_ps(from_ptr + i);
			// 	__m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
			// 	_mm_storeu_si128(reinterpret_cast<__m128i *>(to_ptr + i), y_vec);
			// }
			// for(; i + 3 < num_from; i += 4) {
			// 	__m128 x_vec = _mm_loadu_ps(from_ptr + i);
			// 	__m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
			// 	_mm_storel_epi64(reinterpret_cast<__m128i *>(to_ptr + i), y_vec);
			// }
		}
	};

	template<>
	struct Quantizator<NumberType::FP32, NumberType::Q4_0> {
	public:
		static constexpr NumberType FromType = NumberType::FP32;
		static constexpr NumberType ToType	 = NumberType::Q4_0;

		using FromMetadata              = NumberMetadata<FromType>;
		using ToMetadata                = NumberMetadata<ToType>;
		using FromBlock             	= BlockType<FromType>;
		using ToBlock               	= BlockType<ToType>;

		static constexpr size_t FROM_BLOCK_SIZE = FromMetadata::BLOCK_SIZE;
		static constexpr size_t TO_BLOCK_SIZE   = ToMetadata::BLOCK_SIZE;
		static constexpr size_t FROM_TYPE_SIZE  = FromMetadata::TYPE_SIZE;
		static constexpr size_t TO_TYPE_SIZE    = ToMetadata::TYPE_SIZE;

		static constexpr int NUM_FROM_UNIT = ToMetadata::NUM_BEFORE_DEQUANTIZATION;

	public:
		static void transform(const FromBlock * __restrict from_ptr, ToBlock * __restrict to_ptr, size_t num_from) {
			const size_t num_from_block = num_from / NUM_FROM_UNIT;
			const size_t num_left 		= num_from % NUM_FROM_UNIT;
			SPY_ASSERT_FMT(num_left == 0, "Expect the quantization source ({}) to be aligned with the target block ({}).", num_from, NUM_FROM_UNIT);

			for (int i = 0; i < num_from_block; ++i) {
				float abs_max = 0.0F;
				float max     = 0.0F;
				for (int j = 0; j < NUM_FROM_UNIT; ++j) {
					const float v     = from_ptr[i * NUM_FROM_UNIT + j];
					const float fab_v = std::fabs(v);
					if (abs_max < fab_v) {
						abs_max = fab_v;
						max 	= v;
					}
				}

				const float delta  = max / -8;
				const float idelta = (max == 0.0F) ? (1.0F / delta) : 0.0F;

				to_ptr[i].delta = spy_fp32_to_fp16(delta);
				
				for (int j = 0; j < NUM_FROM_UNIT / 2; ++j) {
					const float x0 = from_ptr[i * NUM_FROM_UNIT + j] * idelta;
					const float x1 = from_ptr[i * NUM_FROM_UNIT + (NUM_FROM_UNIT / 2) + j] * idelta;

					const uint8_t xi0 = std::min<int8_t>(15, static_cast<int8_t>(x0 + 8.5F));
					const uint8_t xi1 = std::min<int8_t>(15, static_cast<int8_t>(x1 + 8.5F));

					to_ptr[i].quants[j]  = xi0;
					to_ptr[i].quants[j] |= (xi1 << 4);
				}

			}
		}
	};

	template<>
	struct Quantizator<NumberType::FP32, NumberType::Q4_1> {
	public:
		static constexpr NumberType FromType = NumberType::FP32;
		static constexpr NumberType ToType	 = NumberType::Q4_1;

		using FromMetadata              = NumberMetadata<FromType>;
		using ToMetadata                = NumberMetadata<ToType>;
		using FromBlock             	= BlockType<FromType>;
		using ToBlock               	= BlockType<ToType>;

		static constexpr size_t FROM_BLOCK_SIZE = FromMetadata::BLOCK_SIZE;
		static constexpr size_t TO_BLOCK_SIZE   = ToMetadata::BLOCK_SIZE;
		static constexpr size_t FROM_TYPE_SIZE  = FromMetadata::TYPE_SIZE;
		static constexpr size_t TO_TYPE_SIZE    = ToMetadata::TYPE_SIZE;

		static constexpr int NUM_FROM_UNIT = ToMetadata::NUM_BEFORE_DEQUANTIZATION;

	public:
		static void transform(const FromBlock * __restrict from_ptr, ToBlock * __restrict to_ptr, size_t num_from) {
			const size_t num_from_block = num_from / NUM_FROM_UNIT;
			const size_t num_left 		= num_from % NUM_FROM_UNIT;
			SPY_ASSERT_FMT(num_left == 0, "Expect the quantization source ({}) to be aligned with the target block ({}).", num_from, NUM_FROM_UNIT);

			for (int i = 0; i < num_from_block; ++i) {
				float min = std::numeric_limits<float>::min();
				float max = std::numeric_limits<float>::max();

				for (int j = 0; j < NUM_FROM_UNIT; ++j) {
					const float v = from_ptr[i * NUM_FROM_UNIT + j];
					min = std::min(min, v);
					max = std::max(max, v);
				}

				const float delta  = (max - min) / 0b111;
				const float idelta = (delta == 0.0F) ? (1.0F / delta) : 0.0F;

				to_ptr[i].delta = spy_fp32_to_fp16(delta);
				to_ptr[i].min 	= spy_fp32_to_fp16(min);
				
				for (int j = 0; j < NUM_FROM_UNIT / 2; ++j) {
					const float x0 = from_ptr[i * NUM_FROM_UNIT + j] * idelta;
					const float x1 = from_ptr[i * NUM_FROM_UNIT + (NUM_FROM_UNIT / 2) + j] * idelta;

					const uint8_t xi0 = std::min<int8_t>(15, static_cast<int8_t>(x0 + 0.5F));
					const uint8_t xi1 = std::min<int8_t>(15, static_cast<int8_t>(x1 + 0.5F));

					to_ptr[i].quants[j]  = xi0;
					to_ptr[i].quants[j] |= (xi1 << 4);
				}

			}
		}
	};

	template<>
	struct Quantizator<NumberType::FP32, NumberType::Q8_0> {
	public:
		static constexpr NumberType FromType = NumberType::FP32;
		static constexpr NumberType ToType	 = NumberType::Q8_0;

		using FromMetadata              = NumberMetadata<FromType>;
		using ToMetadata                = NumberMetadata<ToType>;
		using FromBlock             	= BlockType<FromType>;
		using ToBlock               	= BlockType<ToType>;

		static constexpr size_t FROM_BLOCK_SIZE = FromMetadata::BLOCK_SIZE;
		static constexpr size_t TO_BLOCK_SIZE   = ToMetadata::BLOCK_SIZE;
		static constexpr size_t FROM_TYPE_SIZE  = FromMetadata::TYPE_SIZE;
		static constexpr size_t TO_TYPE_SIZE    = ToMetadata::TYPE_SIZE;

		static constexpr int NUM_FROM_UNIT = ToMetadata::NUM_BEFORE_DEQUANTIZATION;

	public:
		static void transform(const FromBlock * __restrict from_ptr, ToBlock * __restrict to_ptr, size_t num_from) {
			const size_t num_from_block = num_from / NUM_FROM_UNIT;
			const size_t num_left 		= num_from % NUM_FROM_UNIT;
			SPY_ASSERT_FMT(num_left == 0, "Expect the quantization source ({}) to be aligned with the target block ({}).", num_from, NUM_FROM_UNIT);

			for (int i = 0; i < num_from_block; ++i) {
				// Load elements into 4 AVX vectors
				__m256 v0 = _mm256_loadu_ps( from_ptr );
				__m256 v1 = _mm256_loadu_ps( from_ptr + 8 );
				__m256 v2 = _mm256_loadu_ps( from_ptr + 16 );
				__m256 v3 = _mm256_loadu_ps( from_ptr + 24 );
				from_ptr += 32;

				// Compute max(abs(e)) for the block
				const __m256 signBit = _mm256_set1_ps( -0.0F );
				__m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
				maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
				maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
				maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

				__m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
				max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
				max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
				const float maxScalar = _mm_cvtss_f32( max4 );

				// Quantize these floats
				const float d = maxScalar / 127.F;
				to_ptr[i].delta = spy_fp32_to_fp16(d);
				const float id = ( maxScalar != 0.0F ) ? 127.0F / maxScalar : 0.0F;
				const __m256 mul = _mm256_set1_ps( id );

				// Apply the multiplier
				v0 = _mm256_mul_ps( v0, mul );
				v1 = _mm256_mul_ps( v1, mul );
				v2 = _mm256_mul_ps( v2, mul );
				v3 = _mm256_mul_ps( v3, mul );

				// Round to nearest integer
				v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
				v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
				v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
				v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

				// Convert floats to integers
				__m256i i0 = _mm256_cvtps_epi32( v0 );
				__m256i i1 = _mm256_cvtps_epi32( v1 );
				__m256i i2 = _mm256_cvtps_epi32( v2 );
				__m256i i3 = _mm256_cvtps_epi32( v3 );

				// Convert int32 to int16
				i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
				i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
				// Convert int16 to int8
				i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

				// We got our precious signed bytes, but the order is now wrong
				// These AVX2 pack instructions process 16-byte pieces independently
				// The following instruction is fixing the order
				const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
				i0 = _mm256_permutevar8x32_epi32( i0, perm );

				_mm256_storeu_si256(reinterpret_cast<__m256i *>(to_ptr[i].quants), i0);
			}
		}
	};
	
	template<>
	struct Quantizator<NumberType::FP32, NumberType::Q8_1> {
	public:
		static constexpr NumberType FromType = NumberType::FP32;
		static constexpr NumberType ToType	 = NumberType::Q8_1;

		using FromMetadata              = NumberMetadata<FromType>;
		using ToMetadata                = NumberMetadata<ToType>;
		using FromBlock             	= BlockType<FromType>;
		using ToBlock               	= BlockType<ToType>;

		static constexpr size_t FROM_BLOCK_SIZE = FromMetadata::BLOCK_SIZE;
		static constexpr size_t TO_BLOCK_SIZE   = ToMetadata::BLOCK_SIZE;
		static constexpr size_t FROM_TYPE_SIZE  = FromMetadata::TYPE_SIZE;
		static constexpr size_t TO_TYPE_SIZE    = ToMetadata::TYPE_SIZE;

		static constexpr int NUM_FROM_UNIT = ToMetadata::NUM_BEFORE_DEQUANTIZATION;

	public:
		static void transform(const FromBlock * __restrict from_ptr, ToBlock * __restrict to_ptr, size_t num_from) {
			const size_t num_from_block = num_from / NUM_FROM_UNIT;
			const size_t num_left 		= num_from % NUM_FROM_UNIT;
			SPY_ASSERT_FMT(num_left == 0, "Expect the quantization source ({}) to be aligned with the target block ({}).", num_from, NUM_FROM_UNIT);

			for (int i = 0; i < num_from_block; ++i) {
				// Load elements into 4 AVX vectors
				__m256 v0 = _mm256_loadu_ps( from_ptr );
				__m256 v1 = _mm256_loadu_ps( from_ptr + 8 );
				__m256 v2 = _mm256_loadu_ps( from_ptr + 16 );
				__m256 v3 = _mm256_loadu_ps( from_ptr + 24 );
				from_ptr += 32;

				// Compute max(abs(e)) for the block
				const __m256 signBit = _mm256_set1_ps( -0.0F );
				__m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
				maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
				maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
				maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

				__m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
					   max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
					   max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
				const float maxScalar = _mm_cvtss_f32( max4 );

				// Quantize these floats
				const  float delta = maxScalar / 127.0F;
				to_ptr[i].delta    = spy_fp32_to_fp16(delta);
				const  float id    = ( maxScalar != 0.0F ) ? 127.F / maxScalar : 0.0F;
				const  __m256 mul  = _mm256_set1_ps( id );

				// Apply the multiplier
				v0 = _mm256_mul_ps( v0, mul );
				v1 = _mm256_mul_ps( v1, mul );
				v2 = _mm256_mul_ps( v2, mul );
				v3 = _mm256_mul_ps( v3, mul );

				// Round to nearest integer
				v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
				v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
				v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
				v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

				// Convert floats to integers
				__m256i i0 = _mm256_cvtps_epi32( v0 );
				__m256i i1 = _mm256_cvtps_epi32( v1 );
				__m256i i2 = _mm256_cvtps_epi32( v2 );
				__m256i i3 = _mm256_cvtps_epi32( v3 );

				// Compute the sum of the quants and set y[i].s
				to_ptr[i].sum = spy_fp32_to_fp16(delta * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3))));

				// Convert int32 to int16
				i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
				i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
													// Convert int16 to int8
				i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

				// We got our precious signed bytes, but the order is now wrong
				// These AVX2 pack instructions process 16-byte pieces independently
				// The following instruction is fixing the order
				const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
				i0 = _mm256_permutevar8x32_epi32( i0, perm );

				_mm256_storeu_si256(reinterpret_cast<__m256i *>(to_ptr[i].quants), i0);
			}
		}
	};

} // namespace spy