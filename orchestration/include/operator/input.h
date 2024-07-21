#pragma once

#include "graph/op_node.h"
#include "graph/graph.h"
#include "operator/type.h"
#include "operator/config.h"
#include "operator/parameter.h"

#ifndef OPERATOR_HEADER_MACRO
	#warning "Do not include non-linear.h manually, please use operator/operator.h instead."
#endif // OPERATOR_HEADER_MACRO

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::Input> final: OperatorNode {
    public:
		struct Param { Shape shape; };

		using ParameterWrapper    = OperatorParameter<Param>;
		using ParameterRefPointer = ParameterWrapper::RefPointer;

		static constexpr OperatorType TYPE = OperatorType::Reshape;

	public:
		ParameterWrapper params;

    public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

		OperatorDefinition(const Param &params): OperatorNode(TYPE), params(params) {}

		OperatorDefinition(ParameterRefPointer ref_ptr): OperatorNode(TYPE), params(ref_ptr) {}

	    ~OperatorDefinition() noexcept = default;

	public: /* Interface for graph deduction */
        /*!
         * @brief Resolve input nodes and generate output nodes
         * @return Output nodes
         */
		DataNode *deduce(Graph &graph, const DataNodeProperty &prop) {
			DataNode *output_node_ptr = std::addressof(graph.alloc_node<DataNode>());
			output_node_ptr->name = name;
			output_node_ptr->set_prop(prop);
			add_output(output_node_ptr);
			return output_node_ptr;
		}


		/*! 
		 * @brief Validate the metadata of inputs and propagate to generate the metadata of the output nodes
		 * @return Output nodes
		 */
		void propagate() override {
			assert_num_input(0);
			assert_num_output(1);

			const Tensor &in = input_data(0)->tensor;
			auto *out_node = output<DataNode>(0);
			out_node->view_src = input_data(0);

			Tensor &out = out_node->tensor;
			const Shape &target_shape = params.get_val().shape;
			spy_assert(target_shape.total_element() == in.total_element(),
				"invalid shape for reshape: {}, which contains different number of elements from input: {}",
				target_shape, in.shape
			);
			out.shape = target_shape;
		}
    };
	using InputOpDef = OperatorDefinition<OperatorType::Input>;
	using InputParam = InputOpDef::Param;

} // namespace spy