#pragma once

#include <magic_enum.hpp>

#include "util/log/logger.h"
#include "operator/type.h"
#include "graph/data_node.h"
#include "graph/op_node.h"
#include "graph/graph.h"

#ifndef OPERATOR_HEADER_MACRO
	#warning "Do not include mat.h manually, please use operator/operator.h instead."
#endif // OPERATOR_HEADER_MACRO

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::MatMul> final: OperatorNode {
	public:
		static constexpr OperatorType TYPE = OperatorType::MatMul;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;

	public: /* Interface for graph deduction and data propogation */
		DataNode * deduce(Graph &graph, const DataNodeProperty &prop, DataNode *lhs_ptr, DataNode *rhs_ptr) {
			add_input(lhs_ptr, rhs_ptr);
			DataNode *output_node_ptr = std::addressof(graph.alloc_node<DataNode>());
			output_node_ptr->name = name + "-out";
			output_node_ptr->set_prop(prop);
			add_output(output_node_ptr);
			return output_node_ptr;
		}

		void propagate() override {
			assert_num_input(2);
			assert_num_output(1);

			const Tensor &in_0 = input_data(0)->tensor;
			const Tensor &in_1 = input_data(1)->tensor;

			const auto [ne00, ne01, ne02, ne03] = in_0.elements();
			const auto [ne10, ne11, ne12, ne13] = in_1.elements();

			spy_assert(ne00 == ne10 && ne02 == ne12 && ne03 == ne13, 
					"invalid dimensions of operands (operand1: {}, operand2: {})", 
					in_0.shape, in_1.shape
			);

			auto *out_node = output<DataNode>(0);
			out_node->view_src = nullptr;

			Tensor &out = out_node->tensor;
			out.shape = Shape({ ne01, ne11, ne12, ne13 }, NumberType::FP32);
		}
    };
	using MatMulOpDef = OperatorDefinition<OperatorType::MatMul>;

} // namespace spy