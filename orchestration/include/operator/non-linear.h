#pragma once

#include "util/log/logger.h"
#include "operator/type.h"
#include "graph/data_node.h"
#include "graph/op_node.h"
#include "operator/common.h"
#include "operator/parameter.h"

#ifndef OPERATOR_HEADER_MACRO
	#warning "Do not include non-linear.h manually, please use operator/operator.h instead."
#endif // OPERATOR_HEADER_MACRO

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::Relu> final: OperatorUnaryNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Relu;

	public:
		OperatorDefinition(): OperatorUnaryNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;
    };
	using ReluOpDef = OperatorDefinition<OperatorType::Relu>;


    template<>
    struct OperatorDefinition<OperatorType::Silu> final: OperatorUnaryNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Silu;

	public:
	    OperatorDefinition(): OperatorUnaryNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;
    };
	using SiluOpDef = OperatorDefinition<OperatorType::Silu>;


    template<>
    struct OperatorDefinition<OperatorType::Softmax> final: OperatorNode {
	public:
		struct Param { float scale; };

		using ParameterWrapper    = OperatorParameter<Param>;
		using ParameterRefPointer = ParameterWrapper::RefPointer;

		static constexpr OperatorType TYPE = OperatorType::Softmax;

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
		 * @param input_node_ptr The input tensor
		 * @param pos_node_ptr The position for rotation
         * @return Output nodes
         */
		DataNode *deduce(Graph &graph, const DataNodeProperty &prop, DataNode *input_node_ptr, DataNode *mask_node_ptr) {
			add_input(input_node_ptr, mask_node_ptr);
			DataNode *output_node_ptr = std::addressof(graph.alloc_node<DataNode>());
			output_node_ptr->name = name + "-out";
			output_node_ptr->set_prop(prop);
			add_output(output_node_ptr);
			return output_node_ptr;
		}

	    /*!
         * @brief Resolve input nodes and generate output nodes
		 * @param input_node_ptr The input tensor
         * @return Output nodes
         */
		DataNode *deduce(Graph &graph, const DataNodeProperty &prop, DataNode *input_node_ptr) {
			add_input(input_node_ptr);
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
			assert_num_input(2);
			assert_num_output(1);
			params.track_ref_if_needed();

			const Tensor &in  = input_data(0)->tensor;

			auto *out_node = output_data(0);
			out_node->view_src = nullptr;

			Tensor &out = out_node->tensor;
			out.shape = in.shape;
		}
    };
	using SoftmaxOpDef = OperatorDefinition<OperatorType::Softmax>;
	using SoftmaxParam = SoftmaxOpDef::Param;


    template<>
    struct OperatorDefinition<OperatorType::NormRMS> final: OperatorNode {
	public:
		struct Param { float eps; };

		using ParameterWrapper    = OperatorParameter<Param>;
		using ParameterRefPointer = ParameterWrapper::RefPointer;

		static constexpr OperatorType TYPE = OperatorType::NormRMS;

    public:
		ParameterWrapper params;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

		OperatorDefinition(const Param &params): OperatorNode(TYPE), params(params) {}

		OperatorDefinition(ParameterRefPointer ref_ptr): OperatorNode(TYPE), params(ref_ptr) {}

	    ~OperatorDefinition() noexcept = default;

	public:
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
			params.track_ref_if_needed();

			const Tensor &in = input_data(0)->tensor;

			auto *out_node = output<DataNode>(0);
			out_node->view_src = nullptr;

			Tensor &out = out_node->tensor;
			out.shape = in.shape;
		}
    };
	using NormRMSOpDef = OperatorDefinition<OperatorType::NormRMS>;
	using NormRMSParam = NormRMSOpDef::Param;


    template<>
    struct OperatorDefinition<OperatorType::Rope> final: OperatorNode {
    public:
		enum class RopeType : int {
			None = -1,
			Norm = 0,
			Neox = 2,
			GLM  = 4,
		};

		struct Param {
			RopeType 		mode; 

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

		using ParameterWrapper    = OperatorParameter<Param>;
		using ParameterRefPointer = ParameterWrapper::RefPointer;

		static constexpr OperatorType TYPE = OperatorType::Rope;

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
		 * @param input_node_ptr The input tensor
		 * @param pos_node_ptr The position for rotation
         * @return Output nodes
         */
		DataNode *deduce(Graph &graph, const DataNodeProperty &prop, DataNode *input_node_ptr, DataNode *pos_node_ptr) {
			add_input(input_node_ptr, pos_node_ptr);
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
			assert_num_input(2);
			assert_num_output(1);
			params.track_ref_if_needed();

			const Tensor &in  = input_data(0)->tensor;

			spy_assert(in.dim() >= 3, "invalid dimension of input tensor: {}(expect: >=3)", in.dim());

			auto *out_node = output_data(0);
			out_node->view_src = nullptr;

			Tensor &out = out_node->tensor;
			out.shape = in.shape;
		}
    };

	using RopeOpDef = OperatorDefinition<OperatorType::Rope>;
	using RopeType  = RopeOpDef::RopeType;
	using RopeParam = RopeOpDef::Param;

} // namespace spy

SPY_ENUM_FORMATTER(spy::RopeType);