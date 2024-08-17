#pragma once

#include <magic_enum.hpp>
#include <fmt/core.h>

#include "util/log/logger.h"
#include "graph/data_node.h"
#include "graph/op_node.h"
#include "graph/graph.h"

namespace spy {

	struct OperatorUnaryNode: OperatorNode {
	public:
		OperatorUnaryNode(OperatorType type): OperatorNode(type) {}

		~OperatorUnaryNode() noexcept = default;

	public:
		auto &set_name(std::string_view name) {
			this->name = fmt::format("{}: {}", magic_enum::enum_name(op_type), name);
			return *this;
		}

		auto &set_input(DataNode *in_node_ptr) {
			add_input(in_node_ptr);
			return *this;
		}

        /*!
         * @brief Resolve input nodes and generate output nodes
         * @return Output nodes
         */
		DataNode *deduce(Graph &graph, const DataNodeProperty &prop) {
			DataNode *output_node_ptr = std::addressof(graph.alloc_node<DataNode>());
			output_node_ptr->name = name + "-out";
			output_node_ptr->set_prop(prop);
			add_output(output_node_ptr);
			return output_node_ptr;
		}


		/*! 
		 * @brief Validate the metadata of inputs and propagate to generate the metadata of the output nodes
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

	struct OperatorBinaryNode: OperatorNode {
	public:
		OperatorBinaryNode(OperatorType type): OperatorNode(type) {}

		~OperatorBinaryNode() noexcept = default;

	public:
		auto &set_name(std::string_view name) {
			this->name = fmt::format("{}: {}", magic_enum::enum_name(op_type), name);
			return *this;
		}

		auto &set_input(DataNode *lhs_ptr, DataNode *rhs_ptr) {
			add_input(lhs_ptr, rhs_ptr);
			return *this;
		}

        /*!
         * @brief Resolve input nodes and generate output nodes
         * @return Output nodes
         */
		DataNode *deduce(Graph &graph, const DataNodeProperty &prop) {
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

			const Tensor &in_0 = input_data(0)->tensor;
			const Tensor &in_1 = input_data(1)->tensor;

			spy_assert(Shape::can_repeat(in_0.shape, in_1.shape), 
					"Operands should be of repeatable (in_0: {}, in_1: {})", 
					in_0.shape, in_1.shape
			);

			auto *out_node = output<DataNode>(0);
			out_node->view_src = nullptr;

			Tensor &out = out_node->tensor;
			out.shape = in_0.shape;
		}
	};

} // namespace spy