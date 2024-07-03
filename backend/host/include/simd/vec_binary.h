#pragma once

#include <cmath>
#include <immintrin.h>

namespace spy::cpu {

    template<auto T_simd_func, auto T_func>
    inline void simd_vec_binary_fp32(const float *x, const float *y, float *dst, int num) {
        int idx = 0;
#if defined(__AVX2__)
        for (; idx + 8 <= num; idx += 8) {
            const __m256 x_block = _mm256_loadu_ps(x + idx);
            const __m256 y_block = _mm256_loadu_ps(y + idx);
            const __m256 res_block = T_simd_func(x_block, y_block);
            _mm256_store_ps(dst + idx, res_block);
        }
#elif defined(__AVX__)
        for (; idx + 4 <= num; idx += 4) {
            const __m128 x_block = _mm_load_ps(x + idx);
            const __m128 y_block = _mm_load_ps(y + idx);
            const __m128 res_block = T::simd_func(x_block, y_block);
            _mm_store_ps(dst + idx, res_block);
        }
#endif
        for (; idx < num; ++idx) { dst[idx] = T_func(x[idx], y[idx]); }
    }

    template<class T = float>
    inline void vec_add(const T *x, const T *y, T *dst, int num) { for (int i = 0; i < num; ++i) { dst[i] = x[i] + y[i]; } }

    template<class T = float>
    inline void vec_sub(const T *x, const T *y, T *dst, int num) { for (int i = 0; i < num; ++i) { dst[i] = x[i] - y[i]; } }

    template<class T = float>
    inline void vec_mul(const T *x, const T *y, T *dst, int num) { for (int i = 0; i < num; ++i) { dst[i] = x[i] * y[i]; } }

    template<class T = float>
    inline void vec_div(const T *x, const T *y, T *dst, int num) { for (int i = 0; i < num; ++i) { dst[i] = x[i] / y[i]; } }

    template<class T = int>
    inline void vec_mod(const T *x, const T *y, T *dst, int num) { for (int i = 0; i < num; ++i) { dst[i] = x[i] % y[i]; } }

} // namespace