#pragma once

#include "number/tensor.h"
#include "operator/type.h"

namespace spy {

	/*!
	 * @brief Definition of operator
	 * @note When implementing the definition, it should be derived from OperatorNode.
	 * @note The common type definition do not derive from OperatorNode to check the integration at compile time.
	 */
	template<OperatorType T_op_type>
	struct OperatorDefinition {
	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce() const { 
			spy_assert(false, "The definition of operator {} hasn't been implemented.", T_op_type);
			return {};
		}
	};

}  // namespace spy