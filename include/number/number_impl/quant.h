#pragma once

#include <cstdint>
#include <string_view>

#include "number/number_impl/type.h"

namespace spy {
    
	template<>
	struct NumberMetadata<NumberType::Q4_0> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::Q4_0;
		static constexpr std::string_view NAME              = "q4_0";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = 32;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 2;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
			uint16_t delta;
			uint8_t  quants[NUM_BEFORE_DEQUANTIZATION];
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_q4_0_t  = NumberMetadata<NumberType::Q4_0>;
	using block_q4_0_t = meta_q4_0_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::Q4_1> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::Q4_1;
		static constexpr std::string_view NAME              = "q4_1";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = 32;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 2;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
			union {
				struct {
					uint16_t delta;
					uint16_t min;
				};
				uint32_t delta_min;
			};
			int8_t  quants[NUM_BEFORE_DEQUANTIZATION];
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_q4_1_t  = NumberMetadata<NumberType::Q4_1>;
	using block_q4_1_t = meta_q4_1_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::Q5_0> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::Q5_0;
		static constexpr std::string_view NAME              = "q5_0";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = 32;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 2;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
            uint16_t delta;
            uint8_t  quant_high[4];
            uint8_t  quants[NUM_BEFORE_DEQUANTIZATION];
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_q5_0_t  = NumberMetadata<NumberType::Q5_0>;
	using block_q5_0_t = meta_q5_0_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::Q5_1> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::Q5_1;
		static constexpr std::string_view NAME              = "q5_1";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = 32;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 2;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
            union {
                struct {
                    int16_t delta;
                    int16_t min;
                };
                uint32_t delta_min;
            };
            uint8_t quant_high[4];
            uint8_t quants[NUM_BEFORE_DEQUANTIZATION];
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_q5_1_t  = NumberMetadata<NumberType::Q5_1>;
	using block_q5_1_t = meta_q5_1_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::Q8_0> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::Q8_0;
		static constexpr std::string_view NAME              = "q8_0";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = 32;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 1;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
			uint16_t delta;
			int8_t   quants[NUM_BEFORE_DEQUANTIZATION];
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_q8_0_t  = NumberMetadata<NumberType::Q8_0>;
	using block_q8_0_t = meta_q8_0_t::BlockType;

	template<>
	struct NumberMetadata<NumberType::Q8_1> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::Q8_1;
		static constexpr std::string_view NAME              = "q8_1";
		static constexpr bool             IS_QUANTIZATION   = true;
		/// The number of values after dequantization
		static constexpr int K                              = 32;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = K;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 1;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = K / R;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = K / (4 * R);
		static constexpr int NUM_INTEGER                    = I;

		struct BlockType {
			union {
				struct {
					uint16_t delta;
					uint16_t sum; // delta * sum(quants)
				};
				uint16_t delta_sum;
			};
			int8_t  quants[NUM_BEFORE_DEQUANTIZATION];
		};
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_q8_1_t  = NumberMetadata<NumberType::Q8_1>;
	using block_q8_1_t = meta_q8_1_t::BlockType;

} // namespace spy