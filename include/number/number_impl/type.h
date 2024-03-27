#pragma once

#include <cstddef>

namespace spy {

	enum class NumberType: int {
		// Quantized
		FP32 = 0,
		FP16 = 1,
		Q4_0 = 2,
		Q4_1 = 3,
		Q5_0 = 6,
		Q5_1 = 7,
		Q8_0 = 8,
		Q8_1 = 9,
		// K-quant
		Q2_K = 10,
		Q3_K = 11,
		Q4_K = 12,
		Q5_K = 13,
		Q6_K = 14,
		Q8_K = 15,
		// Imatrix K-quant
		IQ2_XXS = 16,
		IQ2_XS  = 17,
		IQ3_XXS = 18,
		IQ1_S   = 19,
		IQ4_NL  = 20,
		IQ3_S   = 21,
		IQ2_S   = 22,
		IQ4_XS  = 23,
		IQ1_M   = 29,
		// Unquantized
		INT8  = 24,
		INT16 = 25,
		INT32 = 26,
		INT64 = 27,
		FP64  = 28
	};

    
#ifdef QK_K_64
    constexpr size_t SIZE_SUPER_BLOCK = 64;
    constexpr size_t SIZE_K_SCALE     = 4;
    constexpr size_t IQ3S_N_SCALE     = 2;
#else
    constexpr size_t SIZE_SUPER_BLOCK = 256;
    constexpr size_t SIZE_K_SCALE     = 12;
    constexpr size_t IQ3S_N_SCALE     = SIZE_SUPER_BLOCK / 64;
#endif


    template<NumberType T_type>
	struct NumberMetadata { };

} // namespace spy