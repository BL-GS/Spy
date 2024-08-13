#pragma once

#include <cstdint>
#include <simde/x86/fma.h>
#include <simde/x86/f16c.h>
#include <simde/x86/avx2.h>

namespace spy::cpu::simd {

#ifndef SPY_AVX2
	#warning "It may be of awful performance if using CPU without AVX2"
#endif

    struct FP32 {
        using block_t = simde__m256;

        static constexpr int  block_len  = 8;
        static constexpr int  block_step = 32; 
        static constexpr int  block_arr  = block_step / block_len;

        static constexpr auto block_zero  = simde_mm256_setzero_ps;
        static constexpr auto block_load  = simde_mm256_loadu_ps;
        static constexpr auto block_store = simde_mm256_storeu_ps;

        static constexpr auto block_add  = simde_mm256_add_ps;
        static constexpr auto block_sub  = simde_mm256_sub_ps;
        static constexpr auto block_mul  = simde_mm256_mul_ps;
        static constexpr auto block_div  = simde_mm256_div_ps;
        static constexpr auto block_madd =  simde_mm256_fmadd_ps;

        static float block_reduce(block_t x[block_arr]) {
            int offset = block_arr >> 1;
            for (int i = 0; i < offset; ++i) { x[i] = _mm256_add_ps(x[i], x[offset+i]); }
            offset >>= 1;
            for (int i = 0; i < offset; ++i) { x[i] = _mm256_add_ps(x[i], x[offset+i]); }
            offset >>= 1;
            for (int i = 0; i < offset; ++i) { x[i] = _mm256_add_ps(x[i], x[offset+i]); }
            const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]), _mm256_extractf128_ps(x[0], 1));
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            return _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
        }
    };

    struct FP16 {
        using block_t = simde__m256;

        static constexpr int  block_len  = 8;
        static constexpr int  block_step = 32; 
        static constexpr int  block_arr  = block_step / block_len;

        static constexpr auto block_zero = simde_mm256_setzero_ps;
        static constexpr auto block_load  = +[](const uint16_t *x) { return simde_mm256_cvtph_ps(simde_mm_loadu_si128((const __m128i *)x)); };
        static constexpr auto block_store = +[](uint16_t *x, simde__m256 y) { simde_mm_storeu_si128((simde__m128i *)(x), simde_mm256_cvtps_ph(y, 0)); };

		static constexpr auto block_add  = simde_mm256_add_ps;
		static constexpr auto block_sub  = simde_mm256_sub_ps;
		static constexpr auto block_mul  = simde_mm256_mul_ps;
		static constexpr auto block_div  = simde_mm256_div_ps;
		static constexpr auto block_madd =  simde_mm256_fmadd_ps;

        static float block_reduce(block_t x[block_arr]) {
            int offset = block_arr >> 1;
            for (int i = 0; i < offset; ++i) { x[i] = _mm256_add_ps(x[i], x[offset+i]); }
            offset >>= 1;
            for (int i = 0; i < offset; ++i) { x[i] = _mm256_add_ps(x[i], x[offset+i]); }
            offset >>= 1;
            for (int i = 0; i < offset; ++i) { x[i] = _mm256_add_ps(x[i], x[offset+i]); }
            const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]), _mm256_extractf128_ps(x[0], 1));
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            return _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
        }
    };

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