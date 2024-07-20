#pragma once

#include "operator/type.h"

namespace spy {

	/*!
	 * @brief Definition of operator
	 * @note When implementing the definition, it should be derived from OperatorNode.
	 * @note The common type definition do not derive from OperatorNode to check the integration at compile time.
	 */
	template<OperatorType T_op_type>
	struct OperatorDefinition { };

}  // namespace spy