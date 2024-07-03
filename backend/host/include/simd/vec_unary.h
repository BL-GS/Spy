#pragma once

#include <cmath>
#include <immintrin.h>

#include "number/lookup_table.h"

namespace spy::cpu {

    template<auto T_simd_func, auto T_func>
    inline void simd_vec_unary_fp32(const float *src, float *dst, int num) {
        int idx = 0;
#if defined(__AVX2__)
        for (; idx + 8 <= num; idx += 8) {
            const __m256 src_block = _mm256_load_ps(src + idx);
            const __m256 res_block = T_simd_func(src_block);
            _mm256_store_ps(dst + idx, res_block);
        }
#elif defined(__AVX__)
        for (; idx + 4 <= num; idx += 4) {
            const __m128 src_block = _mm_load_ps(src + idx);
            const __m128 res_block = T::simd_func(src_block);
            _mm_store_ps(dst + idx, res_block);
        }
#endif
        for (; idx < num; ++idx) { dst[idx] = T_func(src[idx]); }
    }

    inline void vec_sqrt_fp32(const float *src, float *dst, int num) {
        constexpr auto simd_func = +[](__m256 x_block){ return _mm256_sqrt_ps(x_block); };
        constexpr auto func = +[](float x){ return std::sqrt(x); };
        simd_vec_unary_fp32<simd_func, func>(src, dst, num);
    }

    inline void vec_relu_fp32(const float *src, float *dst, int num) {
        for (int i = 0; i < num; ++i) { dst[i] = src[i] > 0.0f ? src[i] : 0.0f; }
    }

    inline void vec_silu_fp32(const float *src, float *dst, int num) {
        for (int i = 0; i < num; ++i) { dst[i] = spy_fp16_to_fp32(LOOK_UP_TABLE.silu(spy_fp32_to_fp16(num))); }
    }

    inline void vec_gelu_fp32(const float *src, float *dst, int num) {
        for (int i = 0; i < num; ++i) { dst[i] = spy_fp16_to_fp32(LOOK_UP_TABLE.gelu(spy_fp32_to_fp16(num))); }
    }

    inline void vec_exp_fp32(const float *src, float *dst, int num) {
        for (int i = 0; i < num; ++i) { dst[i] = spy_fp16_to_fp32(LOOK_UP_TABLE.exp(spy_fp32_to_fp16(num))); }
    }

} // namespace