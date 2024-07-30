/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstddef>
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

		std::vector<NodeID> input_node_id_array_;

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

		void register_input(NodeID id) { input_node_id_array_.push_back(id); }

		bool is_data_node(NodeID id) const { return (id & OP_NODE_ID_MASK) == 0; }

		bool is_op_node(NodeID id)   const { return (id & OP_NODE_ID_MASK) != 0; }

	public:
		BasicNode &operator[](NodeID id) { return node(id); }

		BasicNode &node(NodeID id) const { 
			spy_assert_debug(id != INVALID_NODE_ID, "invalid node id");
			if (is_data_node(id)) { return *data_node_array_[id]; }
			return *op_node_array_[id2idx(id)]; 
		}

		size_t num_data_node() const { return data_node_array_.size(); }

		size_t num_op_node()   const { return op_node_array_.size();   }

		size_t num_node() 	   const { return num_data_node() + num_op_node(); }

		static uint32_t id2idx(NodeID id) { return id & (~OP_NODE_ID_MASK); }

	public:
		void propagate() const;

		template<class T_StartIter, class T_EndIter>
		void get_data_input_count(T_StartIter start_iter, T_EndIter end_iter) const {
			size_t count = 0;
			for (auto iter = start_iter; iter != end_iter; ++iter) {
				spy_assert(count < num_data_node(), 
					"the range of ouput({}) is smaller than #num: {}", 
					count + 1, num_data_node()
				);

				const auto &node_ptr   = data_node_array_[count];
				const size_t num_input = node_ptr->num_input();
				      *iter            = num_input;

				++count;
			}
		}

		std::vector<int> get_data_input_count() const {
			std::vector<int> input_count(num_data_node(), 0);
			get_data_input_count(input_count.begin(), input_count.end());
			return input_count;
		}

		template<class T_StartIter, class T_EndIter>
		void get_data_output_count(T_StartIter start_iter, T_EndIter end_iter) const {
			size_t count = 0;
			for (auto iter = start_iter; iter != end_iter; ++iter) {
				spy_assert(count < num_data_node(), 
					"the range of ouput({}) is smaller than #num: {}", 
					count + 1, num_data_node()
				);

				const auto &node_ptr    = data_node_array_[count];
				const size_t num_output = node_ptr->num_output();
				      *iter             = num_output;

				++count;
			}
		}

		std::vector<int> get_data_output_count() const {
			std::vector<int> output_count(num_data_node(), 0);
			get_data_output_count(output_count.begin(), output_count.end());
			return output_count;
		}


		template<class T_StartIter, class T_EndIter>
		void get_op_input_count(T_StartIter start_iter, T_EndIter end_iter) const {
			size_t count = 0;
			for (auto iter = start_iter; iter != end_iter; ++iter) {
				spy_assert(count < num_op_node(), 
					"the range of ouput({}) is smaller than #num: {}", 
					count + 1, num_op_node()
				);
				
				const auto &node_ptr   = op_node_array_[count];
				const size_t num_input = node_ptr->num_input();
				      *iter            = num_input;

				++count;
			}
		}

		std::vector<int> get_op_input_count() const {
			std::vector<int> input_count(num_op_node(), 0);
			get_op_input_count(input_count.begin(), input_count.end());
			return input_count;
		}

		template<class T_StartIter, class T_EndIter>
		void get_op_output_count(T_StartIter start_iter, T_EndIter end_iter) const {
			size_t count = 0;
			for (auto iter = start_iter; iter != end_iter; ++iter) {
				spy_assert(count < num_op_node(), 
					"the range of ouput({}) is smaller than #num: {}", 
					count + 1, num_op_node()
				);
				
				const auto &node_ptr    = op_node_array_[count];
				const size_t num_output = node_ptr->num_output();
				      *iter             = num_output;

				++count;
			}
		}

		std::vector<int> get_op_output_count() const {
			std::vector<int> output_count(num_op_node(), 0);
			get_op_output_count(output_count.begin(), output_count.end());
			return output_count;
		}
	};

	template<class T_Node>
		requires std::is_base_of_v<BasicNode, T_Node>
	T_Node &Graph::get_node(NodeID node_id) const {
		T_Node &node = static_cast<T_Node &>(storage_ptr->node(node_id));
		spy_assert(this->id == node.graph_id, "try to get a node(graph_id: {}) out of the graph(id: {})",
			node.graph_id, this->id);
		return node;
	}

	template<class T_Node, class ...Args>
		requires std::is_base_of_v<BasicNode, T_Node>
	T_Node &Graph::alloc_node(Args &&...args) const {
		return storage_ptr->alloc_node<T_Node>(*this, std::forward<Args>(args)...);
	}

}  // namespace spy