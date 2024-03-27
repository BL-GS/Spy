#pragma once

#include <cstdint>
#include <string_view>

#include "number/number_impl/type.h"

namespace spy {

	template<>
	struct NumberMetadata<NumberType::IQ2_XXS> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::IQ2_XXS;
		static constexpr std::string_view NAME              = "iq2_xxs";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = SIZE_SUPER_BLOCK;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 8;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
			uint16_t delta;
			uint16_t quants[NUM_BEFORE_DEQUANTIZATION];
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_iq2_xxs_t  = NumberMetadata<NumberType::IQ2_XXS>;
	using block_iq2_xxs_t = meta_iq2_xxs_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::IQ2_XS> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::IQ2_XS;
		static constexpr std::string_view NAME              = "iq2_xs";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = SIZE_SUPER_BLOCK;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 8;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
			uint16_t delta;
			uint16_t quants[NUM_BEFORE_DEQUANTIZATION];
            uint8_t  scales[K / 32];
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_iq2_xs_t  = NumberMetadata<NumberType::IQ2_XS>;
	using block_iq2_xs_t = meta_iq2_xs_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::IQ2_S> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::IQ2_S;
		static constexpr std::string_view NAME              = "iq2_s";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = SIZE_SUPER_BLOCK;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 8;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
			uint16_t delta;
			uint8_t  quants[K / 4];
            uint8_t  quants_high[K / 32];
            uint8_t  scales[K / 32];
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_iq2_s_t  = NumberMetadata<NumberType::IQ2_S>;
	using block_iq2_s_t = meta_iq2_s_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::IQ3_XXS> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::IQ3_XXS;
		static constexpr std::string_view NAME              = "iq3_xxs";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = SIZE_SUPER_BLOCK;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 8;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
			uint16_t delta;
			uint8_t  quants[NUM_BEFORE_DEQUANTIZATION * 3];
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_iq3_xxs_t  = NumberMetadata<NumberType::IQ3_XXS>;
	using block_iq3_xxs_t = meta_iq3_xxs_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::IQ3_S> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::IQ3_S;
		static constexpr std::string_view NAME              = "iq3_s";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = SIZE_SUPER_BLOCK;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 8;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
			uint16_t delta;
			uint8_t  quants[K / 4];
            uint8_t  quants_high[K / 32];
            uint8_t  signs[K / 8];
            uint8_t  scales[IQ3S_N_SCALE];
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_iq3_s_t  = NumberMetadata<NumberType::IQ3_S>;
	using block_iq3_s_t = meta_iq3_s_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::IQ1_S> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::IQ1_S;
		static constexpr std::string_view NAME              = "iq1_s";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = SIZE_SUPER_BLOCK;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 8;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
            uint16_t delta;
			uint8_t  quants[K / 8];
            uint8_t  quants_high[K / 32];
		};

		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_iq1_s_t  = NumberMetadata<NumberType::IQ1_S>;
	using block_iq1_s_t = meta_iq1_s_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::IQ1_M> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::IQ1_M;
		static constexpr std::string_view NAME              = "iq1_m";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = SIZE_SUPER_BLOCK;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 8;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

#ifdef QK_K_64
		struct BlockType {
			uint8_t  quants[K / 8];
            uint8_t  quants_high[K / 16];
            uint16_t delta;
            uint8_t  scales[K / 32];
		};
#else
		struct BlockType {
			uint8_t  quants[K / 8];
            uint8_t  quants_high[K / 16];
            uint8_t  scales[K / 32];
		};
#endif
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_iq1_m_t  = NumberMetadata<NumberType::IQ1_M>;
	using block_iq1_m_t = meta_iq1_m_t::BlockType;

    union iq1_m_scale_t {
        uint16_t f16;
        uint16_t u16;
    };


	template<>
	struct NumberMetadata<NumberType::IQ4_NL> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::IQ4_NL;
		static constexpr std::string_view NAME              = "iq4_nl";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = SIZE_SUPER_BLOCK;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 2;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
            uint16_t delta;
			uint8_t  quants[K / 2];
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_iq4_nl_t  = NumberMetadata<NumberType::IQ4_NL>;
	using block_iq4_nl_t = meta_iq4_nl_t::BlockType;


	template<>
	struct NumberMetadata<NumberType::IQ4_XS> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::IQ4_XS;
		static constexpr std::string_view NAME              = "iq4_xs";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = SIZE_SUPER_BLOCK;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
#ifdef QK_K_64
		static constexpr int R                              = 2;
#else
		static constexpr int R                              = 8;
#endif
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

#ifdef QK_K_64
		struct BlockType {
            uint16_t delta;
			uint8_t  quants[K / 2];
		};
#else
		struct BlockType {
            uint16_t delta;
            uint16_t scale_high;
            uint8_t  scale_low[K / 64];
			uint8_t  quants[K / 2];
		};
#endif
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_iq4_xs_t  = NumberMetadata<NumberType::IQ4_XS>;
	using block_iq4_xs_t = meta_iq4_xs_t::BlockType;

} // namespace spy