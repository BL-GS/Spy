#pragma once

#include <atomic>
#include <magic_enum.hpp>

#include "util/shell/logger.h"
#include "operator/type.h"
#include "operator/config.h"
#include "graph/graph.h"

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::MatMul> final: OperatorNode {
	public:
		static constexpr OperatorType TYPE = OperatorType::MatMul;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;

	public: /* Interface for graph deduction */
		/*! 
		 * @brief Deduce the result tensor with proper shape
		 * @return The tensor with the expected shape
		 */
		Tensor deduce_result() const {
			spy_assert(num_input() == 2, "Expect the number of operands to be 2 (cur: {})", num_input());

			const Tensor &operand_0 = input(0).tensor;
			const Tensor &operand_1 = input(1).tensor;

			const auto &shape_0     = operand_0.get_shape();
			const auto &shape_1     = operand_1.get_shape();
			
			const auto [ne00, ne01, ne02, ne03] = shape_0.elements;
			const auto [ne10, ne11, ne12, ne13] = shape_1.elements;

			spy_assert(ne00 == ne10 && ne02 == ne12 && ne03 == ne13, 
					"Operands should be of the same shape (operand1: {}, operand2: {})", shape_0, shape_1);
			spy_assert(shape_1.number_type == NumberType::FP32, 
						"Expect the type of operand 1 to be fp32 (cur: {})", shape_1.number_type);

			auto num_element = shape_1.elements;
			num_element[0] = shape_0.elements[1];

            constexpr NumberType type_res = NumberType::FP32;
			const Shape shape_res(shape_0.dim, num_element, type_res);	
            
			return { shape_res, nullptr };
		}
    };


} // namespace spy