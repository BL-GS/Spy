#pragma once

#include "operator/common.h"
#include "util/shell/logger.h"
#include "operator/type.h"
#include "operator/parameter.h"
#include "graph/graph.h"
#include "graph/op_node.h"
#include "graph/data_node.h"

#ifndef OPERATOR_HEADER_MACRO
	#warning "Do not include view.h manually, please use operator/operator.h instead."
#endif // OPERATOR_HEADER_MACRO

namespace spy {

    template<>
    struct OperatorDefinition<OperatorType::Nop> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Nop;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

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

			const Tensor &in = input_data(0)->tensor;
			auto *out_node = output<DataNode>(0);
			out_node->view_src = input_data(0);

			Tensor &out = out_node->tensor;
			out.shape = in.shape;
		}
    };
	using NopOpDef = OperatorDefinition<OperatorType::Nop>;


    template<>
    struct OperatorDefinition<OperatorType::GetRow> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::GetRow;

	public:
		OperatorDefinition(): OperatorNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;

	public: /* Interface for graph deduction */
        /*!
         * @brief Resolve input nodes and generate output nodes
         * @return Output nodes
         */
		DataNode *deduce(Graph &graph, const DataNodeProperty &prop, DataNode *in_node_ptr, DataNode *index_node_ptr) {
			add_input(in_node_ptr, index_node_ptr);
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

			const Tensor &in 	   = input_data(0)->tensor;
			const Tensor &index_in = input_data(1)->tensor;

			const size_t in_dim    = in.dim();
			const size_t index_dim = index_in.dim();

			const Shape &in_shape    = in.shape;
			const Shape &index_shape = index_in.shape;

			auto *out_node = output<DataNode>(0);
			Tensor &out = out_node->tensor;
			out_node->view_src = nullptr;

			if (index_dim == in_dim - 1) { // Specify each row
				for (size_t i = 1; i < index_dim; ++i) {
					spy_assert(index_shape.elements[i] == in_shape.elements[i + 1], 
						"invalid shape of index: {}", index_shape
					);
				}
			} else if (index_dim != 1) { // Broadcast index
				spy_assert(false, "invalid dimension of index: {}(expect: {} or {})", 
					index_dim, in_dim - 1, 1);
			}

			auto target_elements = in.elements();
			target_elements[1] = index_shape.elements[0];

			const Shape target_shape(in_dim, target_elements, NumberType::FP32);
			out.shape = target_shape;
		}      
    };
	using GetRowOpDef = OperatorDefinition<OperatorType::GetRow>;


    template<>
    struct OperatorDefinition<OperatorType::Dup> final: OperatorNode {
	public:
		static constexpr OperatorType TYPE = OperatorType::Dup;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

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

			const Tensor &in = input_data(0)->tensor;
			auto *out_node = output<DataNode>(0);
			out_node->view_src = nullptr;

			Tensor &out = out_node->tensor;
			out.shape = in.shape;
		}
    };
	using DupOpDef = OperatorDefinition<OperatorType::Dup>;


    template<>
    struct OperatorDefinition<OperatorType::Copy> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Copy;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

	    ~OperatorDefinition() noexcept = default;

	public: /* Interface for graph deduction */
        /*!
         * @brief Resolve input nodes and generate output nodes
         * @return Output nodes
         */
		DataNode *deduce(Graph &graph, const DataNodeProperty &prop, DataNode *src_node_ptr, DataNode *dst_node_ptr) {
			add_input(src_node_ptr, dst_node_ptr);
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

			const Tensor &src = input_data(0)->tensor;
			const Tensor &dst = input_data(1)->tensor;
			spy_assert(src.shape.total_element() == dst.shape.total_element(), 
				"invalid shape of the second input tensor: {} (expect: {})",
				dst.shape, src.shape
			);

			auto *out_node = output<DataNode>(0);
			out_node->view_src = input_data(0);

			Tensor &out = out_node->tensor;
			out.shape = src.shape;
		}
    };
	using CopyOpDef = OperatorDefinition<OperatorType::Copy>;


    template<>
    struct OperatorDefinition<OperatorType::Reshape> final: OperatorNode {
    public:
		struct Param { Shape new_shape; };

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
			auto *out_node = output<DataNode>(0);
			out_node->view_src = input_data(0);

			Tensor &out = out_node->tensor;
			const Shape &target_shape = cur_param.new_shape;
			spy_assert(target_shape.total_element() == in.total_element(),
				"invalid shape for reshape: {}, which contains different number of elements from input: {}",
				target_shape, in.shape
			);
			out.shape = target_shape;
		}
    };
	using ReshapeOpDef = OperatorDefinition<OperatorType::Reshape>;
	using ReshapeParam = ReshapeOpDef::Param;


    template<>
    struct OperatorDefinition<OperatorType::View> final: OperatorNode {
    public:
		struct Param { 
			int64_t offset;
			Shape 	new_shape; 
		};

		using ParameterWrapper    = OperatorParameter<Param>;
		using ParameterRefPointer = ParameterWrapper::RefPointer;

        static constexpr int64_t	INVALID_OFFSET 	= std::numeric_limits<int64_t>::max();
		static constexpr OperatorType TYPE          = OperatorType::View;

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
			auto *out_node = output<DataNode>(0);
			out_node->view_src = input_data(0);

			Tensor &out = out_node->tensor;
			const Shape &target_shape   = cur_param.new_shape;
			// TODO: assert
			out.shape = target_shape;
		}
    };
	using ViewOpDef = OperatorDefinition<OperatorType::View>;
	using ViewParam = ViewOpDef::Param;


    template<>
    struct OperatorDefinition<OperatorType::Transpose> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Transpose;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

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

			const Tensor &in = input_data(0)->tensor;
			auto *out_node = output<DataNode>(0);
			out_node->view_src = nullptr;

			Shape target_shape = in.shape;
			std::swap(target_shape.elements[0], target_shape.elements[1]);
			std::swap(target_shape.bytes[0], target_shape.bytes[1]);

			Tensor &out = out_node->tensor;
			out.shape = target_shape;
		}
    };
	using TransposeOpDef = OperatorDefinition<OperatorType::Transpose>;

    
    template<>
    struct OperatorDefinition<OperatorType::Permute> final: OperatorNode {
    public:
		struct Param { 
			std::array<int64_t, MAX_DIM> axis;
		};

		using ParameterWrapper    = OperatorParameter<Param>;
		using ParameterRefPointer = ParameterWrapper::RefPointer;

		static constexpr OperatorType TYPE = OperatorType::Permute;

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
			auto *out_node = output<DataNode>(0);
			out_node->view_src = nullptr;

			Shape target_shape = in.shape;
			const auto &axis = cur_param.axis;
			for (int i = 0; i < in.dim(); ++i) {
				target_shape.elements[i] = in.shape.elements[axis[i]];
				target_shape.bytes[i]    = in.shape.bytes[axis[i]];
			}

			Tensor &out = out_node->tensor;
			out.shape = target_shape;
		}
    };
	using PermuteOpDef = OperatorDefinition<OperatorType::Permute>;
	using PermuteParam = PermuteOpDef::Param;


    template<>
    struct OperatorDefinition<OperatorType::Contiguous> final: OperatorNode {
    public:
		static constexpr OperatorType TYPE = OperatorType::Contiguous;

	public:
	    OperatorDefinition(): OperatorNode(TYPE) {}

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

			const Tensor &in = input_data(0)->tensor;
			auto *out_node = output<DataNode>(0);
			out_node->view_src = nullptr;

			Tensor &out = out_node->tensor;
			out.shape = Shape(in.dim(), in.elements(), in.type());
		}
    };
	using ContiguousOpDef = OperatorDefinition<OperatorType::Contiguous>;

} // namespace spy