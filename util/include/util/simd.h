#pragma once

#include <immintrin.h>

namespace spy {

#ifdef __AVX__

#define SPY_SIMD_SIMD

// F32 AVX

#define SPY_SIMD_F32_STEP 32
#define SPY_SIMD_F32_EPR  8

#define SPY_SIMD_F32x8         __m256
#define SPY_SIMD_F32x8_ZERO    _mm256_setzero_ps()
#define SPY_SIMD_F32x8_SET1(x) _mm256_set1_ps(x)
#define SPY_SIMD_F32x8_LOAD    _mm256_loadu_ps
#define SPY_SIMD_F32x8_STORE   _mm256_storeu_ps
#if defined(__FMA__)
    #define SPY_SIMD_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#else
    #define SPY_SIMD_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
#endif
#define SPY_SIMD_F32x8_ADD     _mm256_add_ps
#define SPY_SIMD_F32x8_MUL     _mm256_mul_ps
#define SPY_SIMD_F32x8_REDUCE(res, x)                                 \
do {                                                              \
    int offset = SPY_SIMD_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    const __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = (float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));             \
} while (0)

#define SPY_SIMD_F32_VEC        SPY_SIMD_F32x8
#define SPY_SIMD_F32_VEC_ZERO   SPY_SIMD_F32x8_ZERO
#define SPY_SIMD_F32_VEC_SET1   SPY_SIMD_F32x8_SET1
#define SPY_SIMD_F32_VEC_LOAD   SPY_SIMD_F32x8_LOAD
#define SPY_SIMD_F32_VEC_STORE  SPY_SIMD_F32x8_STORE
#define SPY_SIMD_F32_VEC_FMA    SPY_SIMD_F32x8_FMA
#define SPY_SIMD_F32_VEC_ADD    SPY_SIMD_F32x8_ADD
#define SPY_SIMD_F32_VEC_MUL    SPY_SIMD_F32x8_MUL
#define SPY_SIMD_F32_VEC_REDUCE SPY_SIMD_F32x8_REDUCE

// F16 AVX

#define SPY_SIMD_F16_STEP 32
#define SPY_SIMD_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#define SPY_SIMD_F32Cx8             __m256
#define SPY_SIMD_F32Cx8_ZERO        _mm256_setzero_ps()
#define SPY_SIMD_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#if defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#define SPY_SIMD_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(x)))
#define SPY_SIMD_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#else
static inline __m256 __avx_f32cx8_load(uint16_t *x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = spy_fp16_to_fp32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline void __avx_f32cx8_store(uint16_t *x, __m256 y) {
    float arr[8];

    _mm256_storeu_ps(arr, y);

    for (int i = 0; i < 8; i++)
        x[i] = spy_fp32_to_fp16(arr[i]);
}
#define SPY_SIMD_F32Cx8_LOAD(x)     __avx_f32cx8_load(x)
#define SPY_SIMD_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#endif

#define SPY_SIMD_F32Cx8_FMA         SPY_SIMD_F32x8_FMA
#define SPY_SIMD_F32Cx8_ADD         _mm256_add_ps
#define SPY_SIMD_F32Cx8_MUL         _mm256_mul_ps
#define SPY_SIMD_F32Cx8_REDUCE      SPY_SIMD_F32x8_REDUCE

#define SPY_SIMD_F16_VEC                SPY_SIMD_F32Cx8
#define SPY_SIMD_F16_VEC_ZERO           SPY_SIMD_F32Cx8_ZERO
#define SPY_SIMD_F16_VEC_SET1           SPY_SIMD_F32Cx8_SET1
#define SPY_SIMD_F16_VEC_LOAD(p, i)     SPY_SIMD_F32Cx8_LOAD(p)
#define SPY_SIMD_F16_VEC_STORE(p, r, i) SPY_SIMD_F32Cx8_STORE(p, r[i])
#define SPY_SIMD_F16_VEC_FMA            SPY_SIMD_F32Cx8_FMA
#define SPY_SIMD_F16_VEC_ADD            SPY_SIMD_F32Cx8_ADD
#define SPY_SIMD_F16_VEC_MUL            SPY_SIMD_F32Cx8_MUL
#define SPY_SIMD_F16_VEC_REDUCE         SPY_SIMD_F32Cx8_REDUCE

#define SPY_SIMD_F32_ARR (SPY_SIMD_F32_STEP/SPY_SIMD_F32_EPR)
#define SPY_SIMD_F16_ARR (SPY_SIMD_F16_STEP/SPY_SIMD_F16_EPR)

#endif

} // namespace spy