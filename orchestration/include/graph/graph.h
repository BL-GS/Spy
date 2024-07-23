/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstddef>
#include <queue>
#include <magic_enum.hpp>

#include "util/shell/logger.h"
#include "graph/basic_node.h"
#include "graph/data_node.h"
#include "graph/op_node.h"

namespace spy {

	class GraphStorage;

	struct Graph: PropertyInterface {
	public:
		GraphID id 					= INVALID_GRAPH_ID;

		GraphStorage *storage_ptr 	= nullptr;
		/// The entry of the graph, which is used for dependency analysis
		std::vector<OperatorNode *> entry_point_array;

	public:
		Graph(GraphID id, GraphStorage &graph_storage): id(id), storage_ptr(std::addressof(graph_storage)) {}

		virtual ~Graph() = default;

	public:
		template<class T_Node = BasicNode>
			requires std::is_base_of_v<BasicNode, T_Node>
		T_Node &get_node(NodeID id) const;

		template<class T_Node = BasicNode, class ...Args>
			requires std::is_base_of_v<BasicNode, T_Node>
		T_Node &alloc_node(Args &&...args) const;

		void connect(BasicNode &from_node, BasicNode &to_node) {
			spy_assert(from_node.id != INVALID_NODE_ID && to_node.id != INVALID_NODE_ID, "invalid id (from: {}, to: {})",
				from_node.id, to_node.id);
			spy_assert(from_node.id != to_node.id, "trying to generate a ring (id: {})", from_node.id);
			spy_assert(from_node.graph_id == to_node.graph_id, "trying to connect nodes from different graph (from: {}, to: {})",
				from_node.graph_id, to_node.graph_id);

			from_node.add_output(std::addressof(to_node));
			to_node.add_input(std::addressof(from_node));
		}

		void propagate() const;

	public:
		std::map<std::string_view, std::string> property() const override;

	};

	class GraphStorage {
	public:
		static constexpr NodeID OP_NODE_ID_MASK = 0xFF00'0000;

	private:
		std::vector<std::unique_ptr<DataNode>> 		data_node_array_;
		std::vector<std::unique_ptr<OperatorNode>> 	op_node_array_;

	public:
		GraphStorage() = default;

		~GraphStorage() = default;

		GraphStorage(GraphStorage &&other) = default;

	public:
		template<class T_Node, class ...Args> 
			requires std::is_base_of_v<BasicNode, T_Node>
		T_Node &alloc_node(const Graph &graph, Args &&...args) {
			if constexpr (std::is_base_of_v<OperatorNode, T_Node>) {
				const  NodeID new_id     = op_node_array_.size() | OP_NODE_ID_MASK;
				T_Node &new_node         = *static_cast<T_Node *>(std::to_address(op_node_array_.emplace_back(std::make_unique<T_Node>(std::forward<Args>(args)...))));
					new_node.id       = new_id;
					new_node.graph_id = graph.id;
				return new_node;
			} else {
				const  NodeID new_id     = data_node_array_.size();
				T_Node &new_node         = *static_cast<T_Node *>(std::to_address(data_node_array_.emplace_back(std::make_unique<T_Node>(std::forward<Args>(args)...))));
					new_node.id       = new_id;
					new_node.graph_id = graph.id;
				return new_node;
			}
		}

		bool is_data_node(NodeID id) const { return (id & OP_NODE_ID_MASK) == 0; }

		bool is_op_node(NodeID id)   const { return (id & OP_NODE_ID_MASK) != 0; }

	public:
		BasicNode &operator[](NodeID id) { return node(id); }

		BasicNode &node(NodeID id) { 
			spy_assert_debug(id != INVALID_NODE_ID, "invalid node id");
			if (is_data_node(id)) { return *data_node_array_[id]; }
			return *op_node_array_[id ^ OP_NODE_ID_MASK]; 
		}

		size_t num_data_node() const { return data_node_array_.size(); }

		size_t num_op_node()   const { return op_node_array_.size();   }

		size_t num_node() 	   const { return num_data_node() + num_op_node(); }

	public:
		std::vector<int> get_data_input_count() const {
			std::vector<int> input_count(num_data_node(), 0);
			for (auto &node_ptr: data_node_array_) {
				const NodeID node_id	= node_ptr->id;
				const size_t num_input  = node_ptr->num_input();
				input_count[node_id] 	= num_input;
			}
			return input_count;
		}

		std::vector<int> get_data_output_count() const {
			std::vector<int> output_count(num_data_node(), 0);
			for (auto &node_ptr: data_node_array_) {
				const NodeID node_id	 = node_ptr->id;
				const size_t num_output  = node_ptr->num_output();
				output_count[node_id] 	 = num_output;
			}
			return output_count;
		}

		std::vector<int> get_op_input_count() const {
			std::vector<int> input_count(num_op_node(), 0);
			for (auto &node_ptr: op_node_array_) {
				const NodeID node_id	= node_ptr->id;
				const NodeID op_id		= node_id ^ OP_NODE_ID_MASK;
				const size_t num_input  = node_ptr->num_input();
				input_count[op_id] 		= num_input;
			}
			return input_count;
		}

		std::vector<int> get_op_output_count() const {
			std::vector<int> output_count(num_op_node(), 0);
			for (auto &node_ptr: op_node_array_) {
				const NodeID node_id	 = node_ptr->id;
				const NodeID op_id		 = node_id ^ OP_NODE_ID_MASK;
				const size_t num_output  = node_ptr->num_output();
				output_count[op_id] 	 = num_output;
			}
			return output_count;
		}
	};

	template<class T_Node>
		requires std::is_base_of_v<BasicNode, T_Node>
	T_Node &Graph::get_node(NodeID id) const {
		T_Node &node = static_cast<T_Node &>(storage_ptr->node(id));
		spy_assert(id == node.graph_id, "try to get a node(graph_id: {}) out of the graph(id: {})",
			node.graph_id, id);
		return node;
	}

	template<class T_Node, class ...Args>
		requires std::is_base_of_v<BasicNode, T_Node>
	T_Node &Graph::alloc_node(Args &&...args) const {
		return storage_ptr->alloc_node<T_Node>(*this, std::forward<Args>(args)...);
	}

}  // namespace spy