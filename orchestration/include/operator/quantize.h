#pragma once

#include "util/shell/logger.h"
#include "operator/type.h"
#include "operator/config.h"
#include "graph/data_node.h"
#include "graph/op_node.h"

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::Quantize> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Quantize;

	public:
		NumberType target_type = NumberType::FP32;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

		OperatorDefinition(NumberType target_type):
				OperatorNode(TYPE), target_type(target_type) {}

	    ~OperatorDefinition() noexcept = default;

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
			spy_assert(num_input() == 1, "Expect the number of operands to be 1 (cur: {})", num_input());

			const Tensor &operand 			= input<DataNode>(0)->tensor;
			const size_t target_dim      	= operand.get_dim();
			const auto   target_elements 	= operand.element_array();
			const Shape  target_shape(target_dim, target_elements, target_type);
			return { target_shape, nullptr };
		}
    };

} // namespace spy