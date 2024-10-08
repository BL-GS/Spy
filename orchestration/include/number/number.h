/*
 * @author: BL-GS 
 * @date:   24-3-21
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <string_view>
#include <simde/simde-f16.h>

#include "number/number_impl/type.h"
#include "number/number_impl/common.h"
#include "number/number_impl/quant.h"
#include "number/number_impl/k-quant.h"
#include "number/number_impl/imatrix.h"

namespace spy {

	/*
	 * Utilities
	 */

	template<NumberType T_type>
	using BlockType = typename NumberMetadata<T_type>::BlockType;

	/*!
	 * @brief Get the block size in bytes of the number.
	 * Especially for quantized number, there are several elements clustered as a block. This function returns the size of the block in bytes.
	 * As for unquantized number, it acts like `sizeof(element)`
	 */
	inline constexpr int64_t get_type_size(NumberType number_type) {
		return NumberTypeMapper::map(
			[](const auto T_number_type){ return NumberMetadata<T_number_type>::TYPE_SIZE; }, 
			[](const auto T_number_type){ spy_abort("Unknown number type: {}", T_number_type); },
			number_type
		);
	}

	/*!
	 * @brief Get the number of elements in the number block.
	 * Especially for quantized number, there are several elements sharing one metadata (e.g. scale), and these elements are clustered as a block.
	 * This function returns the number of elements in a block. 
	 * As for unquantized number, it returns 1.
	 */
	inline constexpr int64_t get_block_size(NumberType number_type) {
		return NumberTypeMapper::map(
			[](const auto T_number_type){ return NumberMetadata<T_number_type>::BLOCK_SIZE; }, 
			[](const auto T_number_type){ spy_abort("Unknown number type: {}", T_number_type); },
			number_type
		);
	}

	/*!
	 * @brief Get the size in bytes of the data row.
	 * Especially for quantized number, there are several elements clustered as a block.
	 * And we need to consider the number of block (num_element / get_block_size(number_type)) and the size in bytes of block.
	 * As for unquantized number, it acts like `num_element` * `sizeof(element)`
	 */
	inline constexpr int64_t get_row_size(NumberType number_type, int64_t num_element) {
		return num_element * get_type_size(number_type) / get_block_size(number_type);
	}

	/*!
	 * @brief Get the name of the number type
	 */
	inline constexpr std::string_view get_type_name(NumberType number_type) {
		return NumberTypeMapper::map(
			[](const auto T_number_type){ return NumberMetadata<T_number_type>::NAME; }, 
			[](const auto T_number_type){ spy_abort("Unknown number type: {}", T_number_type); },
			number_type
		);
	}


	template<NumberType T_number_type>
	struct IsQuantizedExtractor {
		constexpr auto operator()() const { return NumberMetadata<T_number_type>::IS_QUANTIZATION; };
	};

	/*!
	 * @brief Whether the number type is quantized
	 */
	inline constexpr bool is_quantized(NumberType number_type) {
		return NumberTypeMapper::map(
			[](const auto T_number_type){ return NumberMetadata<T_number_type>::IS_QUANTIZATION; }, 
			[](const auto T_number_type){ spy_abort("Unknown nunmber type: {}", T_number_type); }, 
			number_type
		);
	}

	/*!
	 * @brief Whether the number type is trivial. In other word, CPU can compute with them naturally
	 */
	inline constexpr bool is_trivial(NumberType number_type) {
		switch (number_type) {
		case NumberType::FP64:
		case NumberType::FP32:
		case NumberType::INT64:
		case NumberType::INT32:
		case NumberType::INT16:
		case NumberType::INT8:
			return true;
		default:
			return false;
		}
	}

}  // namespace spy