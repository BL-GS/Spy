/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstdint>
#include <magic_enum.hpp>
#include <fmt/format.h>

#include "util/exception.h"
#include "util/type/enum.h"
#include "util/shell/logger.h"

namespace spy {

	enum class OpAssertOption: bool {
		/// Whether the shapes of tensors match the operator
		Shape		= true, 
		/// Whether the result of operator is valid
		Numeric		= true
	};

#define SPY_OP_ASSERT(type, expression, ...) 	spy_assert((expression) && static_cast<bool>(OpAssertOption:: type), __VA_ARGS__)

#define SPY_OP_SHAPE_ASSERT(expression, ...) 	SPY_OP_ASSERT(Shape, expression, __VA_ARGS__)

#define SPY_OP_NUMERIC_ASSERT(expression, ...)	SPY_OP_ASSERT(Numeric, expression, __VA_ARGS__)

#define SPY_OP_ASSERT_FMT(type, expression, ...) 	spy_assert((expression) && static_cast<bool>(OpAssertOption:: type), __VA_ARGS__)

#define SPY_OP_SHAPE_ASSERT_FMT(expression, ...) 	SPY_OP_ASSERT_FMT(Shape, expression, __VA_ARGS__)

#define SPY_OP_NUMERIC_ASSERT_FMT(expression, ...)	SPY_OP_ASSERT_FMT(Numeric, expression, __VA_ARGS__)


	enum class OperatorType: int32_t {
		/* input/output */
		Input,
		/* 1-operand operator */
		Nop, Dup, Copy, View, Reshape, MaskedCopy, MaskedSelect, GetRow, Transpose, Permute, Contiguous,
		Relu, Silu, Gelu, Max, Min, Softmax,
		/* 2-operand operator */
		Add, Sub, Mul, Div, Mod, MatMul,
		/* mul-operand operator */
		Sum,
		/* summarization operator */
		Norm, NormRMS, Rope,
		/* quantization / dequantization */
		Quantize
	};

#define OPERATOR_TYPE_MAP(map)              \
		map(OperatorType::Input)			\
		map(OperatorType::Nop)              \
		map(OperatorType::Dup)              \
		map(OperatorType::Copy)             \
		map(OperatorType::View)             \
		map(OperatorType::Reshape)          \
		map(OperatorType::MaskedCopy)       \
		map(OperatorType::MaskedSelect)     \
		map(OperatorType::GetRow)           \
		map(OperatorType::Transpose)        \
		map(OperatorType::Permute)          \
		map(OperatorType::Contiguous)       \
		map(OperatorType::Relu)             \
		map(OperatorType::Silu)             \
		map(OperatorType::Gelu)             \
		map(OperatorType::Max)              \
		map(OperatorType::Min)              \
		map(OperatorType::Softmax)          \
                                            \
		map(OperatorType::Add)              \
		map(OperatorType::Sub)              \
		map(OperatorType::Mul)              \
		map(OperatorType::Div)              \
		map(OperatorType::Mod)              \
		map(OperatorType::MatMul)           \
                                            \
		map(OperatorType::Sum)              \
                                            \
		map(OperatorType::Norm)             \
		map(OperatorType::NormRMS)          \
		map(OperatorType::Rope)             \
	                                        \
		map(OperatorType::Quantize)

	template<template<OperatorType> class T_func>
	inline constexpr auto operator_type_switch(const OperatorType op_type) {
#define OPERATOR_TYPE_CASE(type) \
		case type: return T_func<type>()();

		switch (op_type) {
			OPERATOR_TYPE_MAP(OPERATOR_TYPE_CASE)
		}
#undef OPERATOR_TYPE_CASE
		spy_abort("Unknown type of number");
		spy_unreachable();
	}

	template<template<OperatorType> class T_func, class ...Args>
	inline constexpr auto operator_type_switch(const OperatorType op_type, Args &&...args) {
#define OPERATOR_TYPE_CASE(type) \
		case type: return T_func<type>()(std::forward<Args>(args)...);

		switch (op_type) {
			OPERATOR_TYPE_MAP(OPERATOR_TYPE_CASE)
		}
#undef OPERATOR_TYPE_CASE
		spy_abort("Unknown type of number");
		spy_unreachable();
	}

	inline constexpr bool is_view(OperatorType op_type) { 
		switch (op_type) {
		case OperatorType::View:
		case OperatorType::Reshape:
		case OperatorType::Transpose:
		case OperatorType::Permute:
		case OperatorType::Copy:
			return true;
		default:
			return false;
		}
	}

	inline constexpr bool is_nop(OperatorType op_type) {
		return op_type == OperatorType::Nop;
	}

	/*!
	 * @brief Definition of operator
	 * @note When implementing the definition, it should be derived from OperatorNode.
	 * @note The common type definition do not derive from OperatorNode to check the integration at compile time.
	 */
	template<OperatorType T_op_type>
	struct OperatorDefinition { };

	template<OperatorType T_op_type>
	struct OperatorNodeImpl { };


}  // namespace spy

SPY_ENUM_FORMATTER(spy::OperatorType);