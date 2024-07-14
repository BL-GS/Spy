/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstddef>
#include <magic_enum.hpp>

#include "util/shell/logger.h"
#include "graph/basic_node.h"

namespace spy {

	class GraphStorage;

	struct Graph: PropertyInterface {
	public:
		GraphID id 					= INVALID_GRAPH_ID;

		GraphStorage *storage_ptr 	= nullptr;
		/// The entry of the graph, which is used for dependency analysis
		const BasicNode *entry_point = nullptr;

	public:
		Graph(GraphID id, GraphStorage &graph_storage): id(id), storage_ptr(std::addressof(graph_storage)) {}

		~Graph() = default;

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

	public:
		std::map<std::string_view, std::string> property() const override;
	};

	class GraphStorage: PropertyInterface {
	private:
		std::vector<Graph> graph_array_;

		std::vector<std::unique_ptr<BasicNode>> node_array_;

	public:
		GraphStorage() = default;

		~GraphStorage() = default;

		GraphStorage(GraphStorage &&other) = default;

	public:
		template<class T_Node, class ...Args> 
			requires std::is_base_of_v<BasicNode, T_Node>
		T_Node &alloc_node(Graph &graph, Args &&...args) {
			const  NodeID new_id     = node_array_.size();
			T_Node &new_node         = node_array_.emplace_back(std::forward<Args>(args)...);
			       new_node.id       = new_id;
			       new_node.graph_id = graph.id;
			return new_node;
		}

	public:
		BasicNode &operator[](NodeID id) { return node(id); }

		BasicNode &node(NodeID id) { 
			spy_assert_debug(id != INVALID_NODE_ID, "invalid node id");
			spy_assert_debug(id < num_node(), "invalid node id");
			return *node_array_[id]; 
		}

		size_t num_node() const { return node_array_.size(); }

	public:
		std::vector<int> get_input_count() const {
			std::vector<int> input_count(num_node(), 0);
			for (auto &node_ptr: node_array_) {
				const NodeID node_id	= node_ptr->id;
				const size_t num_input  = node_ptr->num_input();
				input_count[node_id] 	= num_input;
			}
			return input_count;
		}

		std::vector<int> get_output_count() const {
			std::vector<int> output_count(num_node(), 0);
			for (auto &node_ptr: node_array_) {
				const NodeID node_id	 = node_ptr->id;
				const size_t num_output  = node_ptr->num_output();
				output_count[node_id] 	 = num_output;
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
		return storage_ptr->alloc_node<T_Node>(this, std::forward<Args>(args)...);
	}

}  // namespace spy