#pragma once

#include "number/number.h"
#include "number/lookup_table.h"
#include "number/quantization_impl/type.h"

namespace spy {

    template<>
	struct Quantizator<NumberType::Q8_0, NumberType::FP32> {
		using FromMetadata              = NumberMetadata<NumberType::Q8_0>;
		using ToMetadata                = NumberMetadata<NumberType::FP32>;
		using FromType                  = BlockType<NumberType::Q8_0>;
		using ToType                    = BlockType<NumberType::FP32>;

		static constexpr size_t FROM_BLOCK_SIZE = FromMetadata::BLOCK_SIZE;
		static constexpr size_t TO_BLOCK_SIZE   = ToMetadata::BLOCK_SIZE;
		static constexpr size_t FROM_TYPE_SIZE  = FromMetadata::TYPE_SIZE;
		static constexpr size_t TO_TYPE_SIZE    = ToMetadata::TYPE_SIZE;

		static constexpr int FROM_NUM = FromMetadata::NUM_BEFORE_DEQUANTIZATION;
		static constexpr int TO_NUM   = FromMetadata::NUM_AFTER_DEQUANTIZATION;

		static constexpr void transform(const FromType *from_ptr, ToType *to_ptr) {
			for (int i = 0; i < FROM_NUM; ++i) {
				const float delta = LOOK_UP_TABLE.fp32(from_ptr->delta);
				const float quant = static_cast<float>(from_ptr->quants[i]); 
				to_ptr[i]         = quant * delta;
			}
		}

		static constexpr void transform(const FromType *from_ptr, ToType *to_ptr, size_t num) {
			for (size_t i = 0; i < num; ++i) {
				transform(from_ptr + i, to_ptr + i * TO_NUM);
			}
		}
	};
    
} // namespace spy