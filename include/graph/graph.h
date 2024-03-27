/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstddef>
#include <memory>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <fmt/format.h>
#include <magic_enum.hpp>

#include "util/logger.h"
#include "number/tensor.h"
#include "operator/type.h"
#include "graph/type.h"

namespace spy {

	using NodeCredit = uint32_t;

	struct DataNode;
	struct OperatorNode;
	class Graph;

	struct BaseNode {
		friend class Graph;
	protected:
		NodeCredit  credit;
		/// The name of the node
		std::string name;
		/// Input nodes
		std::vector<BaseNode *> input;
		/// Output nodes
		std::vector<BaseNode *> output;

	public:
		BaseNode(): credit(-1), name("unnamed") {}

		BaseNode(NodeCredit credit, const std::string_view name): credit(credit), name(name) {}

		BaseNode(const BaseNode &other) = default;

		virtual ~BaseNode() noexcept 	= default;

		BaseNode &operator=(const BaseNode &other) = default;

	public:
		/*!
		 * @brief Connect with input node
		 * @param in_node_ptr: The pointer of the input node
		 */
		template<class T>
		void input_connect(T &in_node_ptr)   { input.push_back(std::to_address(in_node_ptr));    }

		/*!
		 * @brief Connect with output node
		 * @param out_node_ptr: The pointer of the output node
		 */
		template<class T>
		void output_connect(T &out_node_ptr) { output.push_back(std::to_address(out_node_ptr));  }

		/*!
		 * @brief Get all input nodes
		 */
		const std::vector<BaseNode *> &get_input() 	const { return input;  }

		/*!
		 * @brief Get a input node
		 */
		template<class T>
		T &get_input(size_t idx) 					const { return *static_cast<T *>(input[idx]);  }

		/*!
		 * @brief Get all output nodes
		 */
		const std::vector<BaseNode *> &get_output() const { return output; }

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

	struct DataNode final: BaseNode {
	public: /* Content */
		/// The metadata of tensor
		Tensor tensor;
		/// The source of view
		DataNode *view_src 		= nullptr;
		/// The type of data
		DataNodeType data_type	= DataNodeType::Variable;

	public:
		DataNode() = default;

		template<class ...Args>
		DataNode(NodeCredit credit, std::string_view name, DataNodeType data_type, Args &&...args) : 
			BaseNode(credit, name), tensor(std::forward<Args>(args)...), data_type(data_type) {}

		DataNode(const DataNode &other) = default;

		~DataNode() noexcept override = default;
	};

	struct OperatorNode: BaseNode {
	public: /* Content */
		/// The type of operation
		OperatorType op_type;

	public:
		OperatorNode() : op_type(OperatorType::Nop) {}

		OperatorNode(NodeCredit credit, std::string_view name, OperatorType op_type) : BaseNode(credit, name), op_type(op_type) {}

		OperatorNode(const OperatorNode &other) = default;

		~OperatorNode() noexcept override = default;
	};

	class Graph: public OperatorNode {
	public:
		static constexpr NodeCredit INVALID_NODE_CREDIT = std::numeric_limits<NodeCredit>::max();
		static constexpr NodeCredit INPUT_NODE_CREDIT	= 0;
		static constexpr NodeCredit OUTPUT_NODE_CREDIT  = 1;

		using NodeElement     = std::unique_ptr<BaseNode>;

	protected:
		std::vector<NodeElement> node_storage_;
		/// The number of nodes connected to this node.
		std::vector<size_t>      dep_count_;
		/// The number of nodes connected from this node.
		std::vector<size_t>		 back_dep_count_;

	public:
		Graph(const std::string_view name, const NodeCredit credit = INVALID_NODE_CREDIT) : OperatorNode(credit, name, OperatorType::Nop) { 
			node_storage_.emplace_back(std::make_unique<OperatorNode>(INPUT_NODE_CREDIT, "input", OperatorType::Nop));
			node_storage_.emplace_back(std::make_unique<OperatorNode>(OUTPUT_NODE_CREDIT, "output", OperatorType::Nop));
			dep_count_ = { 0, 0 };
			back_dep_count_ = { 0, 0 };
		}

		~Graph() noexcept override = default;

	public:
		/*!
		 * @brief Allocate a new node in graph
		 */
		template<class T_Node, class ...Args>
		NodeCredit alloc_node(const std::string_view name, Args &&...args) {
			const NodeCredit new_node_credit = node_storage_.size();

			node_storage_.emplace_back(std::make_unique<T_Node>(new_node_credit, name, std::forward<Args>(args)...));
			dep_count_.emplace_back(0);
			back_dep_count_.emplace_back(0);

			return new_node_credit;
		}

		/*!
		 * @brief Connect two nodes
		 */
		void connect(NodeCredit from, NodeCredit to) {
			SPY_ASSERT(from != INVALID_NODE_CREDIT, "connect from invalid node");
			SPY_ASSERT(to   != INVALID_NODE_CREDIT, "connect to invalid node");
			SPY_ASSERT(from != to, "Do not build ring");

			auto &from_node_ptr = node_storage_[from];
			auto &to_node_ptr   = node_storage_[to];

			from_node_ptr->output_connect(to_node_ptr);
			to_node_ptr->input_connect(from_node_ptr);

			dep_count_[to]++;
			back_dep_count_[from]++;
		}

		/*!
		 * @brief Note the tensor as prepared at start
		 */
		void set_start(NodeCredit node_credit) {
			SPY_ASSERT(dep_count_[node_credit] == 0, "Expect the start tensor not to be dependent on others");
			connect(INPUT_NODE_CREDIT, node_credit);
		}

		/*!
		 * @brief Note the tensor as prepared at start
		 */
		void set_end(NodeCredit node_credit) {
			connect(node_credit, OUTPUT_NODE_CREDIT);
		}

	public: /* Basic information */
		size_t 	num_node() 								const { return node_storage_.size(); }

		size_t 	dep_count(NodeCredit node_credit) 		const { return dep_count_[node_credit]; }

		size_t 	back_dep_count(NodeCredit node_credit) 	const { return back_dep_count_[node_credit]; }

		const std::vector<size_t> &get_dep_count() 		const { return dep_count_; }

		const std::vector<size_t> &get_back_dep_count() const { return back_dep_count_; }

		const NodeElement &get_node(NodeCredit node_credit) const {
			return node_storage_[node_credit];
		}

		template<class T_Node = BaseNode>
		T_Node *get_node_content(NodeCredit node_credit) const { 
			return dynamic_cast<T_Node *>(std::to_address(node_storage_[node_credit]));
		}
	};


	/* -------------- Utilities --------------- */

	inline Tensor &get_tensor_from_node(BaseNode *node_ptr) {
		return static_cast<DataNode *>(node_ptr)->tensor;
	}

}  // namespace spy