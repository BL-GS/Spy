#pragma once

#include "number/number.h"
#include "util/shell/logger.h"

namespace spy {

    template<NumberType From_type, NumberType To_type>
	struct Quantizator { 
	public:
		static constexpr NumberType FromType = From_type;
		static constexpr NumberType ToType	 = To_type;

		using FromMetadata              = NumberMetadata<FromType>;
		using ToMetadata                = NumberMetadata<ToType>;
		using FromBlock             	= BlockType<FromType>;
		using ToBlock               	= BlockType<ToType>;

		static constexpr size_t FROM_BLOCK_SIZE = FromMetadata::BLOCK_SIZE;
		static constexpr size_t TO_BLOCK_SIZE   = ToMetadata::BLOCK_SIZE;
		static constexpr size_t FROM_TYPE_SIZE  = FromMetadata::TYPE_SIZE;
		static constexpr size_t TO_TYPE_SIZE    = ToMetadata::TYPE_SIZE;

	public:
		static void transform(const FromBlock * __restrict from_ptr, ToBlock * __restrict to_ptr, size_t num_from) {
			if constexpr (FromType == ToType) {
				std::memcpy(to_ptr, from_ptr, num_from);
			} else {
				throw SpyUnimplementedException("Unimplemented quantization");
			}
		}
	};



} // namespace spy