#pragma once

#include <cstdint>
#include <span>

#include "number/tensor.h"
#include "model/vocab/type.h"
#include "operator/type.h"

namespace spy {

    struct RopeContext {
		ModelRopeType 	mode; 

		int32_t 		num_past;
		int32_t 		num_dim; 
		int32_t 		num_context; 
		int32_t 		num_origin_context;

		float 			freq_base; 
		float 			freq_scale; 
		float 			extend_factor; 
		float 			attention_factor;
		float 			beta_fast; 
		float 			beta_slow; 
		float 			xpos_base; 
		bool 			xpos_down;
	};

	struct OperatorEnvParam {
		/// The concurrency of operator execution
		int concurrency;
		/// The id of thread executing the operator
		int tid;
		/// The buffer
		std::span<uint8_t> buffer_span;
	};

	/*!
	 * @brief Definition of operator
	 * @note When implementing the definition, it should be drived from OperatorNode.
	 * @note The common type defintion do not derive from OperatorNode to check the integration at compile time.
	 */
	template<OperatorType T_op_type>
	struct OperatorDefinition {
	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const { 
			SPY_ASSERT_FMT(false, "The definition of operator {} hasn't been implemented.", magic_enum::enum_name(T_op_type));
			return {};
		}
	};

}  // namespace spy