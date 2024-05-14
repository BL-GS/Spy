/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <magic_enum.hpp>

#include "util/shell/logger.h"
#include "number/tensor.h"
#include "operator/type.h"
#include "graph/type.h"

namespace spy {

	struct NodeCredit {
		bool 		is_data;
		uint32_t	node_id;

		bool operator==(const NodeCredit &other) const { return is_data == other.is_data && node_id == other.node_id; }
	};

	struct DataNode;
	struct OperatorNode;
	class Graph;

	struct DataNode {
		friend class Graph;
	public:
		using NodeArray = std::vector<OperatorNode *>;

	public: /* Content */
		NodeCredit  	credit;
		/// The name of the node
		std::string 	name;
		/// The metadata of tensor
		Tensor 			tensor;
		/// The source of view
		DataNode *		view_src 		= nullptr;
		/// The type of data
		DataNodeType 	data_type	= DataNodeType::Variable;

	protected:
		NodeArray		output;

	public:
		DataNode() = default;

		template<class ...Args>
		DataNode(NodeCredit credit, std::string_view name, DataNodeType data_type, Args &&...args) : 
			credit(credit), name(name), tensor(std::forward<Args>(args)...), data_type(data_type) {}

		DataNode(const DataNode &other) = default;

		~DataNode() noexcept = default;

	public:
		/*!
		 * @brief Connect with output node
		 * @param out_node_ptr: The pointer of the output node
		 */
		void output_connect(OperatorNode *out_node_ptr) { output.push_back(out_node_ptr);  }

		/*!
		 * @brief Get all output nodes
		 */
		const NodeArray &get_output() const { return output; }

		/*!
		 * @brief Get a output node
		 */
		template<class T>
		T &get_output(size_t idx) 					const { return *static_cast<T *>(output[idx]);  }

	public: /* Utilities */
		/*!
		 * @brief Get the credit of this node, which denote the position of the node in the graph
		 */
		NodeCredit  get_credit() const { return credit;  }

		/*!
		 * @brief Get the name of this node.
		 */
		std::string get_name()   const { return name; }
	};

	struct OperatorNode {
		friend class Graph;
	public:
		using NodeArray = std::vector<DataNode *>;

	public: /* Content */
		NodeCredit  	credit;
		/// The name of the node
		std::string 	name;
		/// The type of operation
		OperatorType 	op_type;
		/// Input nodes
		NodeArray   	input;
		/// Output nodes
		NodeArray  		output;

	public:
		OperatorNode() : op_type(OperatorType::Nop) {}

		OperatorNode(NodeCredit credit, std::string_view name, OperatorType op_type): credit(credit), name(name), op_type(op_type) {}

		OperatorNode(const OperatorNode &other) = default;

		~OperatorNode() noexcept = default;

	public:
		/*!
		 * @brief Connect with input node
		 * @param in_node_ptr: The pointer of the input node
		 */
		void input_connect(DataNode *in_node_ptr)   { input.push_back(in_node_ptr);    }

		/*!
		 * @brief Connect with output node
		 * @param out_node_ptr: The pointer of the output node
		 */
		void output_connect(DataNode *out_node_ptr) { output.push_back(out_node_ptr);  }

		/*!
		 * @brief Get all input nodes
		 */
		const NodeArray &get_input() 	const { return input;  }

		/*!
		 * @brief Get a input node
		 */
		template<class T>
		T &get_input(size_t idx) 					const { return *static_cast<T *>(input[idx]);  }

		/*!
		 * @brief Get all output nodes
		 */
		const NodeArray &get_output() const { return output; }

		/*!
		 * @brief Get a output node
		 */
		template<class T>
		T &get_output(size_t idx) 					const { return *static_cast<T *>(output[idx]);  }

	public: /* Utilities */
		/*!
		 * @brief Get the credit of this node, which denote the position of the node in the graph
		 */
		NodeCredit  get_credit() const { return credit;  }

		/*!
		 * @brief Get the name of this node.
		 */
		std::string get_name()   const { return name; }
	};

	class Graph {
	public:
		static constexpr NodeCredit INVALID_NODE_CREDIT { true, std::numeric_limits<uint32_t>::max() };
		static constexpr NodeCredit OUTPUT_NODE_CREDIT { false, 0 };

		using DataNodeElement     = std::unique_ptr<DataNode>;
		using OperatorNodeElement = std::unique_ptr<OperatorNode>;

	protected:
		std::vector<DataNodeElement> 		data_nodes_;

		std::vector<OperatorNodeElement>	op_nodes_;

		std::string 						name_;

	public:
		Graph(const std::string_view name): name_(name) {
			const NodeCredit credit = alloc_node<OperatorNode>("output", OperatorType::Nop);
			spy_assert(credit == OUTPUT_NODE_CREDIT, "The first operator node should be output node");
		}

		Graph(Graph &&other) = delete;

		~Graph() noexcept = default;

	public:
		/*!
		 * @brief Allocate a new node in graph
		 */
		template<class T_Node, class ...Args>
		NodeCredit alloc_node(const std::string_view name, Args &&...args) { 
			if constexpr (std::is_same_v<T_Node, DataNode>) {
				const uint32_t new_node_id = data_nodes_.size();
				const NodeCredit new_node_credit{ true, new_node_id };

				data_nodes_.emplace_back(std::make_unique<T_Node>(new_node_credit, name, std::forward<Args>(args)...));
				return new_node_credit;				
			} else {
				const uint32_t new_node_id = op_nodes_.size();
				const NodeCredit new_node_credit{ false, new_node_id };

				op_nodes_.emplace_back(std::make_unique<T_Node>(new_node_credit, name, std::forward<Args>(args)...));
				return new_node_credit;				
			}
		}

		/*!
		 * @brief Connect two nodes
		 */
		void connect(NodeCredit from, NodeCredit to) {
			spy_assert(from != INVALID_NODE_CREDIT, "connect from invalid node");
			spy_assert(to   != INVALID_NODE_CREDIT, "connect to invalid node");
			spy_assert(from != to, "Do not build ring");

			if (from.is_data) {
				spy_assert(!to.is_data, "connect two data nodes");

				auto &from_node_ptr = data_nodes_[from.node_id];
				auto &to_node_ptr   = op_nodes_[to.node_id];

				from_node_ptr->output_connect(to_node_ptr.get());
				to_node_ptr->input_connect(from_node_ptr.get());
			} else {
				spy_assert(to.is_data, "connect two operator nodes");

				auto &from_node_ptr = op_nodes_[from.node_id];
				auto &to_node_ptr   = data_nodes_[to.node_id];

				from_node_ptr->output_connect(to_node_ptr.get());
			}
		}

		void set_end(NodeCredit credit) {
			connect(credit, OUTPUT_NODE_CREDIT);
		}

	public: /* Basic information */
		size_t 	num_data_node() 						const { return data_nodes_.size(); }

		size_t  num_op_node()							const { return op_nodes_.size(); }

		const DataNodeElement &get_data_node(NodeCredit node_credit) const { return data_nodes_[node_credit.node_id]; }

		const DataNodeElement &get_data_node(size_t idx) 			 const { return data_nodes_[idx]; }

		const DataNodeElement &get_op_node(NodeCredit node_credit) 	 const { return data_nodes_[node_credit.node_id]; }

		const DataNodeElement &get_op_node(size_t idx) 				 const { return data_nodes_[idx]; }

		template<class T_Container = std::vector<size_t>>
		T_Container get_data_send_count()	const {
			T_Container dep_count;
			dep_count.reserve(data_nodes_.size());

			for (const auto &data_node: data_nodes_) { dep_count.emplace_back(data_node->output.size()); }
			return dep_count;
		}

		template<class T_Container = std::vector<size_t>>
		T_Container get_data_recv_count()	const {
			T_Container dep_count(data_nodes_.size(), 0);

			for (const auto &op_node: op_nodes_) {
				for (const auto &data_node: op_node->output) {
					const auto data_node_id = data_node->credit.node_id;
					dep_count[data_node_id]++;
				}
			}
			return dep_count;
		}

		template<class T_Container = std::vector<size_t>>
		T_Container get_op_send_count()	const {
			T_Container dep_count;
			dep_count.reserve(op_nodes_.size());

			for (const auto &op_node: op_nodes_) { dep_count.emplace_back(op_node->output.size()); }
			return dep_count;
		}

		template<class T_Container = std::vector<size_t>>
		T_Container get_op_recv_count()	const {
			T_Container dep_count;
			dep_count.reserve(op_nodes_.size());

			for (const auto &op_node: op_nodes_) { dep_count.emplace_back(op_node->input.size()); }
			return dep_count;
		}

		template<class T_Node>
		auto *get_node_content(NodeCredit node_credit) const { 
			if constexpr (std::is_same_v<T_Node, DataNode>) {
				spy_assert_debug(node_credit.is_data, "Visit data node with credit `is_data` field as false.");
				return data_nodes_[node_credit.node_id].get();
			} else {
				spy_assert_debug(!node_credit.is_data, "Visit operator node with credit `is_data` field as true.");
				return static_cast<T_Node *>(op_nodes_[node_credit.node_id].get());
			}
		}
	};

}  // namespace spy