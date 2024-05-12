/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstdint>
#include <magic_enum.hpp>
#include <fmt/format.h>

#include "util/exception.h"
#include "backend/config.h"

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
	constexpr auto operator_type_switch(const OperatorType op_type) {
#define OPERATOR_TYPE_CASE(type) \
		case type: return T_func<type>()();

		switch (op_type) {
			OPERATOR_TYPE_MAP(OPERATOR_TYPE_CASE)
		}

		spy_assert<true>(false, "Unknown type of number");
#undef OPERATOR_TYPE_CASE
	}

	template<template<OperatorType> class T_func, class ...Args>
	constexpr auto operator_type_switch(const OperatorType op_type, Args &&...args) {
#define OPERATOR_TYPE_CASE(type) \
		case type: return T_func<type>()(std::forward<Args>(args)...);

		switch (op_type) {
			OPERATOR_TYPE_MAP(OPERATOR_TYPE_CASE)
		}

		spy_assert<true>(false, "Unknown type of number");
#undef OPERATOR_TYPE_CASE
	}

	template<OperatorType T_op_type>
	struct OperatorNodeImpl { };

	class SpyUnimplementedOperatorException: protected SpyUnimplementedException {
	public:
		SpyUnimplementedOperatorException(BackendType backend_type, OperatorType op_type):
			SpyUnimplementedException(fmt::format("backend: {}; operator: {}", magic_enum::enum_name(backend_type), magic_enum::enum_name(op_type))) {}
	};

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

}  // namespace spy