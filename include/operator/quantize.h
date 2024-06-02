#pragma once

#include "util/shell/logger.h"
#include "operator/type.h"
#include "operator/config.h"
#include "graph/graph.h"

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::Quantize> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Quantize;

	public:
		NumberType target_type = NumberType::FP32;

	public:
		OperatorDefinition() = default;

		OperatorDefinition(NodeCredit credit, NumberType target_type): 
				OperatorNode(credit, TYPE), target_type(target_type) {}

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
			spy_assert(input.size() == 1, "Expect the number of operands to be 1 (cur: {})", input.size());

			const Tensor &operand        = input[0]->tensor;
			const size_t target_dim      = operand.get_dim();
			const auto   target_elements = operand.element_array();
			const Shape  target_shape(target_dim, target_elements, target_type);
			return { target_shape, nullptr };
		}
    };

} // namespace spy