#pragma once

#include "number/number.h"
#include "number/quantization_impl/type.h"
#include "number/quantization_impl/q4_0.h"
#include "number/quantization_impl/q4_1.h"
#include "number/quantization_impl/q8_0.h"
#include "number/quantization_impl/fp32.h"

namespace spy {

	template<NumberType T_from, NumberType T_to>
	inline void quantize_inner(const void *src, void *dst, size_t num) {
		using FromType = NumberMetadata<T_from>::BlockType;
		using ToType   = NumberMetadata<T_to>::BlockType;

		Quantizator<T_from, T_to>::transform(
				static_cast<const FromType *>(src),
				static_cast<ToType *>(dst),
				num
		);
	}

	inline static void auto_quantize_inner(NumberType type_0, const void *src, NumberType type_1, void *dst, size_t num) {
		const auto transform_func = NumberTypeMapper::product_map([](const auto T_type_0, const auto T_type_1){
			return quantize_inner<T_type_0, T_type_1>;
		}, type_0, type_1);
		transform_func(src, dst, num);
	}

}  // namespace spy