#pragma once

#include <cstddef>

#include "util/enum.h"
#include "util/logger.h"

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
		// Unquantized
		INT8  = 24,
		INT16 = 25,
		INT32 = 26,
		INT64 = 27,
		FP64  = 28,
		IQ1_M = 29
	};

	template<class T_Enum, std::underlying_type_t<T_Enum> T_end = 256, std::underlying_type_t<T_Enum> T_begin = 0>
		requires std::is_enum_v<T_Enum>
	using EnumMapperAlias = EnumMapper<T_Enum, static_cast<T_Enum>(T_end), static_cast<T_Enum>(T_begin)>;

	using NumberTypeMapper = EnumMapperAlias<NumberType, 29, 0>;

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
	struct NumberMetadata {
		static constexpr NumberType       NUMBER_TYPE       = T_type;
		static constexpr std::string_view NAME              = "unknown";
		static constexpr bool             IS_QUANTIZATION   = false;

		using BlockType                                     = int;
		static constexpr int TYPE_SIZE                      = 0;
		static constexpr int BLOCK_SIZE                     = 0;
	};

} // namespace spy