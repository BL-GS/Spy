#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "number/number_impl/type.h"

namespace spy {

	template<>
	struct NumberMetadata<NumberType::Q2_K> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::Q2_K;
		static constexpr std::string_view NAME              = "q2_k";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = SIZE_SUPER_BLOCK;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 4;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
			uint8_t scales[K / 16];
			uint8_t quants[NUM_BEFORE_DEQUANTIZATION];
            union {
                struct {
                    uint16_t delta;
                    uint16_t min;
                };
                uint32_t delta_min;
            };
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_q2_k_t  = NumberMetadata<NumberType::Q2_K>;
	using block_q2_k_t = meta_q2_k_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::Q3_K> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::Q3_K;
		static constexpr std::string_view NAME              = "q3_k";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = SIZE_SUPER_BLOCK;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 4;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

#ifdef QK_K_64
		struct BlockType {
            uint8_t  high_mask[K / 8];
			uint8_t  quants[NUM_BEFORE_DEQUANTIZATION];
            uint8_t  scales[2];
            uint16_t delta;
		};
#else   
		struct BlockType {
            uint8_t  high_mask[K / 8];
			uint8_t  quants[NUM_BEFORE_DEQUANTIZATION];
            uint8_t  scales[12];
            uint16_t delta;
		};
#endif
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_q3_k_t  = NumberMetadata<NumberType::Q3_K>;
	using block_q3_k_t = meta_q3_k_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::Q4_K> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::Q4_K;
		static constexpr std::string_view NAME              = "q4_k";
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

#ifdef QK_K_64
		struct BlockType {
            uint16_t delta;
            uint8_t  scales[2];
            uint8_t  quants[K / 2];
		};
#else   
		struct BlockType {
            union {
                struct {
                    uint16_t delta;
                    uint16_t min;
                };
                uint32_t delta_min;
            };
            uint8_t scales[SIZE_K_SCALE];
            uint8_t quants[NUM_BEFORE_DEQUANTIZATION];
		};
#endif
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_q4_k_t  = NumberMetadata<NumberType::Q4_K>;
	using block_q4_k_t = meta_q4_k_t::BlockType;


	template<>
	struct NumberMetadata<NumberType::Q5_K> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::Q5_K;
		static constexpr std::string_view NAME              = "q5_k";
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

#ifdef QK_K_64
		struct BlockType {
            uint16_t delta;
            int8_t   scales[K / 16];
            uint8_t  quants_high[K / 8];
            uint8_t  quants[NUM_BEFORE_DEQUANTIZTION];
		};
#else   
		struct BlockType {
            union {
                struct {
                    uint16_t delta;
                    uint16_t min;
                };
                uint32_t delta_min;
            };
            uint8_t scales[SIZE_K_SCALE];
            uint8_t quants_high[K / 8];
            uint8_t quants[NUM_BEFORE_DEQUANTIZATION];
		};
#endif
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_q5_k_t  = NumberMetadata<NumberType::Q5_K>;
	using block_q5_k_t = meta_q5_k_t::BlockType;

    template<>
	struct NumberMetadata<NumberType::Q6_K> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::Q6_K;
		static constexpr std::string_view NAME              = "q6_k";
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
            uint8_t  quants_low[NUM_BEFORE_DEQUANTIZATION];
            uint8_t  quants_high[K / 4];
            uint8_t  scales[K / 16];
            uint16_t delta;
		};

		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_q6_k_t  = NumberMetadata<NumberType::Q6_K>;
	using block_q6_k_t = meta_q6_k_t::BlockType;


    template<>
	struct NumberMetadata<NumberType::Q8_K> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::Q8_K;
		static constexpr std::string_view NAME              = "q8_k";
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
            float    delta;
            uint8_t  quants[K];
            uint8_t  sums[K / 16];
		};

		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_q8_k_t  = NumberMetadata<NumberType::Q8_K>;
	using block_q8_k_t = meta_q8_k_t::BlockType;

} // namespace spy