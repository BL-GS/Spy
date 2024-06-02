#pragma once

#include "util/shell/logger.h"
#include "operator/type.h"
#include "operator/config.h"
#include "graph/graph.h"

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::Add> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Add;

	public:
		OperatorDefinition() = default;

		OperatorDefinition(NodeCredit credit): OperatorNode(credit, TYPE) {}

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
            spy_assert(input.size() == 2, "Expect the number of operands to be 2 (cur: {})", input.size());

			const Tensor &operand_0 = input[0]->tensor;
			const Tensor &operand_1 = input[1]->tensor;
			spy_assert(operand_0.get_number_type() == NumberType::FP32);
			spy_assert(operand_1.get_number_type() == NumberType::FP32);

			const auto &shape_0     = operand_0.get_shape();
			const auto &shape_1     = operand_1.get_shape();
			const auto  type_0   = operand_0.get_number_type();
			const auto  type_1   = operand_1.get_number_type();
			
			spy_assert(shape_0 == shape_1 || Shape::can_repeat(shape_0, shape_1), 
					"Operands should be of the same shape or repeatable shape (operand1: {}, operand2: {})", 
					shape_0.to_string(), shape_1.to_string());
			spy_assert(type_0 == type_1, 
					"Operands should be of the same type (operand1: {}, operand2: {})", 
					magic_enum::enum_name(type_0), magic_enum::enum_name(type_1));

			const Shape shape_res     = shape_0;
			return { shape_res, nullptr };
		}
    };


    template<>
    struct OperatorDefinition<OperatorType::Sub> final: OperatorNode {
	public:
		static constexpr OperatorType TYPE = OperatorType::Sub;

	public:
		OperatorDefinition() = default;

		OperatorDefinition(NodeCredit credit): OperatorNode(credit, TYPE) {}

	
	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
            spy_assert(input.size() == 2, "Expect the number of operands to be 2 (cur: {})", input.size());

			const Tensor &operand_0 = input[0]->tensor;
			const Tensor &operand_1 = input[1]->tensor;
			spy_assert(operand_0.get_number_type() == NumberType::FP32);
			spy_assert(operand_1.get_number_type() == NumberType::FP32);

			const auto &shape_0     = operand_0.get_shape();
			const auto &shape_1     = operand_1.get_shape();
			const auto  type_0   = operand_0.get_number_type();
			const auto  type_1   = operand_1.get_number_type();
			
			spy_assert(shape_0 == shape_1 || Shape::can_repeat(shape_0, shape_1), 
					"Operands should be of the same shape or repeatable shape (operand1: {}, operand2: {})", 
					shape_0.to_string(), shape_1.to_string());
			spy_assert(type_0 == type_1, 
					"Operands should be of the same type (operand1: {}, operand2: {})", 
					magic_enum::enum_name(type_0), magic_enum::enum_name(type_1));

			const Shape shape_res     = shape_0;
			return { shape_res, nullptr };
		}
    };


    template<>
    struct OperatorDefinition<OperatorType::Mul> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Mul;

	public:
		OperatorDefinition() = default;

		OperatorDefinition(NodeCredit credit): OperatorNode(credit, TYPE) {}


	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
            spy_assert(input.size() == 2, "Expect the number of operands to be 2 (cur: {})", input.size());

			const Tensor &operand_0 = input[0]->tensor;
			const Tensor &operand_1 = input[1]->tensor;
			spy_assert(operand_0.get_number_type() == NumberType::FP32);
			spy_assert(operand_1.get_number_type() == NumberType::FP32);

			const auto &shape_0     = operand_0.get_shape();
			const auto &shape_1     = operand_1.get_shape();
			const auto  type_0   = operand_0.get_number_type();
			const auto  type_1   = operand_1.get_number_type();
			
			spy_assert(shape_0 == shape_1 || Shape::can_repeat(shape_0, shape_1), 
					"Operands should be of the same shape or repeatable shape (operand1: {}, operand2: {})", 
					shape_0.to_string(), shape_1.to_string());
			spy_assert(type_0 == type_1, 
					"Operands should be of the same type (operand1: {}, operand2: {})", 
					magic_enum::enum_name(type_0), magic_enum::enum_name(type_1));

			const Shape shape_res     = shape_0;
			return { shape_res, nullptr };
		}
    };


    template<>
    struct OperatorDefinition<OperatorType::Div> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Div;

	public:
		OperatorDefinition() = default;

		OperatorDefinition(NodeCredit credit): OperatorNode(credit, TYPE) {}


	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
            spy_assert(input.size() == 2, "Expect the number of operands to be 2 (cur: {})", input.size());

			const Tensor &operand_0 = input[0]->tensor;
			const Tensor &operand_1 = input[1]->tensor;
			spy_assert(operand_0.get_number_type() == NumberType::FP32);
			spy_assert(operand_1.get_number_type() == NumberType::FP32);

			const auto &shape_0     = operand_0.get_shape();
			const auto &shape_1     = operand_1.get_shape();
			const auto  type_0   = operand_0.get_number_type();
			const auto  type_1   = operand_1.get_number_type();
			
			spy_assert(shape_0 == shape_1 || Shape::can_repeat(shape_0, shape_1), 
					"Operands should be of the same shape or repeatable shape (operand1: {}, operand2: {})", 
					shape_0.to_string(), shape_1.to_string());
			spy_assert(type_0 == type_1, 
					"Operands should be of the same type (operand1: {}, operand2: {})", 
					magic_enum::enum_name(type_0), magic_enum::enum_name(type_1));

			const Shape shape_res     = shape_0;
			return { shape_res, nullptr };
		}
    };

} // namespace spy