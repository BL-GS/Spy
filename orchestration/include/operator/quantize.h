#pragma once

#include "util/log/logger.h"
#include "operator/type.h"
#include "operator/parameter.h"
#include "graph/graph.h"
#include "graph/data_node.h"
#include "graph/op_node.h"

#ifndef OPERATOR_HEADER_MACRO
	#warning "Do not include quantize.h manually, please use operator/operator.h instead."
#endif // OPERATOR_HEADER_MACRO

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::Quantize> final: OperatorNode {
    public:
		struct Param { NumberType target_type; };

		using ParameterWrapper    = OperatorParameter<Param>;
		using ParameterRefPointer = ParameterWrapper::RefPointer;

		static constexpr OperatorType TYPE = OperatorType::Quantize;

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
		DataNode *deduce(Graph &graph, const DataNodeProperty &prop, DataNode *in_node_ptr) {
			add_input(in_node_ptr);
			DataNode *output_node_ptr = std::addressof(graph.alloc_node<DataNode>());
			output_node_ptr->name = name + "-out";
			output_node_ptr->set_prop(prop);
			add_output(output_node_ptr);
			return output_node_ptr;
		}


		/*! 
		 * @brief Validate the metadata of inputs and propagate to generate the metadata of the output nodes
		 * @return Output nodes
		 */
		void propagate() override {
			assert_num_input(1);
			assert_num_output(1);
			const Param &cur_param = params.track_ref_if_needed();

			const Tensor &in = input_data(0)->tensor;
			const size_t target_dim      = in.dim();
			const auto   target_elements = in.elements();
			const NumberType target_type = cur_param.target_type;
			const Shape  target_shape(target_dim, target_elements, target_type);

			auto *out_node = output<DataNode>(0);
			out_node->view_src = nullptr;

			Tensor &out = out_node->tensor;
			out.shape = target_shape;
		}
    };
	using QuantizeOpDef = OperatorDefinition<OperatorType::Quantize>;
	using QuantizeParam = QuantizeOpDef::Param;

} // namespace spy