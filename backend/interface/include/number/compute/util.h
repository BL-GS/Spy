#pragma once

#include <immintrin.h>

namespace spy {

	// horizontally add 8 floats
	static inline float hsum_float_8(const __m256 x) {
		__m128 res = _mm256_extractf128_ps(x, 1);
		res = _mm_add_ps(res, _mm256_castps256_ps128(x));
		res = _mm_add_ps(res, _mm_movehl_ps(res, res));
		res = _mm_add_ss(res, _mm_movehdup_ps(res));
		return _mm_cvtss_f32(res);
	}

	// horizontally add 8 int32_t
	static inline int hsum_i32_8(const __m256i a) {
		const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
		const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
		const __m128i sum64 = _mm_add_epi32(hi64, sum128);
		const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
		return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
	}

	// horizontally add 4 int32_t
	static inline int hsum_i32_4(const __m128i a) {
		const __m128i hi64 = _mm_unpackhi_epi64(a, a);
		const __m128i sum64 = _mm_add_epi32(hi64, a);
		const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
		return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
	}

	// multiply int8_t, add results pairwise twice
	static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
		// Get absolute values of x vectors
		const __m128i ax = _mm_sign_epi8(x, x);
		// Sign the values of the y vectors
		const __m128i sy = _mm_sign_epi8(y, x);
		// Perform multiplication and create 16-bit values
		const __m128i dot = _mm_maddubs_epi16(ax, sy);
		const __m128i ones = _mm_set1_epi16(1);
		return _mm_madd_epi16(ones, dot);
	}

	// add int16_t pairwise and return as float vector
	static inline __m256 sum_i16_pairs_float(const __m256i x) {
		const __m256i ones = _mm256_set1_epi16(1);
		const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
		return _mm256_cvtepi32_ps(summed_pairs);
	}

	static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
		// Perform multiplication and create 16-bit values
		const __m256i dot = _mm256_maddubs_epi16(ax, sy);
		return sum_i16_pairs_float(dot);
	}

	// multiply int8_t, add results pairwise twice and return as float vector
	static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
		// Get absolute values of x vectors
		const __m256i ax = _mm256_sign_epi8(x, x);
		// Sign the values of the y vectors
		const __m256i sy = _mm256_sign_epi8(y, x);
		return mul_sum_us8_pairs_float(ax, sy);
	}

} // namespace spy