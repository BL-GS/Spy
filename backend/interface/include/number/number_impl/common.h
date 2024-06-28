#pragma once

#include <cstdint>
#include <string_view>

#include "number/number_impl/type.h"

namespace spy {
    

	template<>
	struct NumberMetadata<NumberType::INT8> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::INT8;
		static constexpr std::string_view NAME              = "i8";
		static constexpr bool             IS_QUANTIZATION   = false;
		/// The number of values after dequantization
		static constexpr int K                              = 1;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = 1;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 1;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = 1;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = 1;
		static constexpr int NUM_INTEGER                    = 1;

		using BlockType                                     = uint8_t;
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_int8_t  = NumberMetadata<NumberType::INT8>;

	template<>
	struct NumberMetadata<NumberType::INT16> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::INT16;
		static constexpr std::string_view NAME              = "i16";
		static constexpr bool             IS_QUANTIZATION   = false;
		/// The number of values after dequantization
		static constexpr int K                              = 1;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = 1;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 1;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = 1;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = 1;
		static constexpr int NUM_INTEGER                    = 1;

		using BlockType                                     = uint16_t;
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_int16_t  = NumberMetadata<NumberType::INT16>;

	template<>
	struct NumberMetadata<NumberType::INT32> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::INT32;
		static constexpr std::string_view NAME              = "i32";
		static constexpr bool             IS_QUANTIZATION   = false;
		/// The number of values after dequantization
		static constexpr int K                              = 1;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = 1;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 1;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = 1;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = 1;
		static constexpr int NUM_INTEGER                    = 1;

		using BlockType                                     = uint32_t;
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_int32_t  = NumberMetadata<NumberType::INT32>;

	template<>
	struct NumberMetadata<NumberType::INT64> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::INT64;
		static constexpr std::string_view NAME              = "i64";
		static constexpr bool             IS_QUANTIZATION   = false;
		/// The number of values after dequantization
		static constexpr int K                              = 1;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = 1;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 1;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = 1;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = 1;
		static constexpr int NUM_INTEGER                    = 1;

		using BlockType                                     = uint64_t;
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_int64_t  = NumberMetadata<NumberType::INT64>;

	template<>
	struct NumberMetadata<NumberType::FP16> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::FP16;
		static constexpr std::string_view NAME              = "f16";
		static constexpr bool             IS_QUANTIZATION   = false;
		/// The number of values after dequantization
		static constexpr int K                              = 1;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = 1;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 1;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = 1;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = 1;
		static constexpr int NUM_INTEGER                    = 1;

		using BlockType                                     = uint16_t;
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_fp16_t  = NumberMetadata<NumberType::FP16>;

	template<>
	struct NumberMetadata<NumberType::FP32> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::FP32;
		static constexpr std::string_view NAME              = "f32";
		static constexpr bool             IS_QUANTIZATION   = false;
		/// The number of values after dequantization
		static constexpr int K                              = 1;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = 1;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 1;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = 1;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = 1;
		static constexpr int NUM_INTEGER                    = 1;

		using BlockType                                     = float;
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_fp32_t  = NumberMetadata<NumberType::FP32>;

	template<>
	struct NumberMetadata<NumberType::FP64> {
		static constexpr NumberType       NUMBER_TYPE       = NumberType::FP64;
		static constexpr std::string_view NAME              = "f64";
		static constexpr bool             IS_QUANTIZATION   = false;
		/// The number of values after dequantization
		static constexpr int K                              = 1;
		static constexpr int NUM_AFTER_DEQUANTIZATION       = 1;
		/// The number of values after dequantization / The number of values before dequantization
		static constexpr int R                              = 1;
		static constexpr int NUM_BEFORE_DEQUANTIZATION      = 1;
		/// The number of 32-bit Integers before dequantization
		static constexpr int I                              = 1;
		static constexpr int NUM_INTEGER                    = 1;

		using BlockType                                     = double;
		static constexpr int TYPE_SIZE                      = sizeof(BlockType);
		static constexpr int BLOCK_SIZE                     = K;
	};
	using meta_fp64_t  = NumberMetadata<NumberType::FP64>;

} // namespace spy