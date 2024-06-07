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
#include <ranges>
#include <magic_enum.hpp>

#include "util/shell/logger.h"
#include "util/wrapper/atomic.h"
#include "number/tensor.h"
#include "operator/type.h"
#include "graph/type.h"
#include "backend/type.h"

namespace spy {

	using NodeID = uint32_t;
	static constexpr NodeID INVALID_NODE_ID = std::numeric_limits<NodeID>::max();

	struct DataNode;
	struct OperatorNode;
	class Graph;

	struct DataNode {
		friend class Graph;
		friend class GraphControlHeader;
	public:
		using NodeArray = std::vector<OperatorNode *>;

	public: /* Content */
		/// The name of the node
		DataNodeProperty 	property;
		/// The metadata of tensor
		Tensor 				tensor;

	public: /* Information about data */
		/// The source of view
		DataNode *			view_src 		= nullptr;
		/// The type of backend
		BackendType			backend_type	= BackendType::Unknown;

	protected: /* Information about graph */
		struct NodeInfo {
			NodeID          id              = INVALID_NODE_ID;
			NodeArray       output;
		} info;

	public:
		DataNode() = default;

		template<class ...Args>
		DataNode(DataNodeProperty property, Args &&...args) :
			property(property), tensor(std::forward<Args>(args)...) {}

		DataNode(const DataNode &other) = default;

		~DataNode() noexcept = default;

	protected:
		/*!
		 * @brief Connect with output node
		 * @param out_node_ptr: The pointer of the output node
		 */
		void output_connect(OperatorNode *out_node_ptr) { info.output.push_back(out_node_ptr);  }

	public:
		NodeID id()                         const { return info.id;            }

		size_t num_output()                 const { return info.output.size(); }
		/*!
		 * @brief Get all output nodes
		 */
		const NodeArray &output()           const { return info.output; }

		/*!
		 * @brief Get a output node
		 */
		OperatorNode &output(size_t idx)    const { return *info.output[idx];  }
	};

	struct OperatorNode {
		friend class Graph;
		friend class GraphControlHeader;
	public:
		using NodeArray = std::vector<DataNode *>;

	public: /* Content */
		/// The type of operation
		OperatorType 	op_type;

	public: /* Information about data */
		/// The type of backend
		BackendType		backend_type	= BackendType::Unknown;

	protected: /* Information about graph */
		struct NodeInfo {
			NodeID          id          = INVALID_NODE_ID;
			/// Input nodes
			NodeArray   	input;
			/// Output nodes
			NodeArray  		output;
		} info;


	public:
		OperatorNode() : op_type(OperatorType::Nop) {}

		OperatorNode(OperatorType op_type): op_type(op_type) {}

		OperatorNode(const OperatorNode &other) = default;

		~OperatorNode() noexcept = default;

	protected:
		/*!
		 * @brief Connect with input node
		 * @param in_node_ptr: The pointer of the input node
		 */
		void input_connect(DataNode *in_node_ptr)   { info.input.push_back(in_node_ptr);    }

		/*!
		 * @brief Connect with output node
		 * @param out_node_ptr: The pointer of the output node
		 */
		void output_connect(DataNode *out_node_ptr) { info.output.push_back(out_node_ptr);  }

	public:
		NodeID id()                     const { return info.id;            }

		size_t num_input()              const { return info.input.size();  }

		size_t num_output()             const { return info.output.size(); }

		/*!
		 * @brief Get all input nodes
		 */
		const NodeArray &input() 	    const { return info.input;  }

		/*!
		 * @brief Get a input node
		 */
		DataNode &input(size_t idx)     const { return *info.input[idx];  }

		/*!
		 * @brief Get all output nodes
		 */
		const NodeArray &output()       const { return info.output; }

		/*!
		 * @brief Get a output node
		 */
		DataNode &output(size_t idx)    const { return *info.output[idx];  }
	};

	class Graph {
		friend class GraphView;
	public:
		static constexpr NodeID OUTPUT_NODE_ID = 0;

		using DataNodeElement     = std::unique_ptr<DataNode>;

		using OperatorNodeElement = std::unique_ptr<OperatorNode>;

	protected:
		std::vector<DataNodeElement> 		data_nodes_;

		std::vector<OperatorNodeElement>	op_nodes_;

		std::string 						name_;

	public:
		Graph(const std::string_view name): name_(name) {
			OperatorNode *node_ptr = alloc_node<OperatorNode>(OperatorType::Nop);
			node_ptr->info.id = OUTPUT_NODE_ID;
		}

		Graph(Graph &&other) = delete;

		~Graph() noexcept = default;

	public:
		/*!
		 * @brief Allocate a new node in graph
		 */
		template<class T_Node, class ...Args>
		T_Node *alloc_node(Args &&...args) {
			if constexpr (std::is_same_v<T_Node, DataNode>) {
				auto &iter = data_nodes_.emplace_back(std::make_unique<T_Node>(std::forward<Args>(args)...));
				return iter.get();
			} else {
				auto &iter = op_nodes_.emplace_back(std::make_unique<T_Node>(std::forward<Args>(args)...));
				return static_cast<T_Node *>(iter.get());
			}
		}

		void connect(OperatorNode *op_node_ptr, DataNode *data_node_ptr) const {
			op_node_ptr->output_connect(data_node_ptr);
		}

		void connect(DataNode *data_node_ptr, OperatorNode *op_node_ptr) const {
			op_node_ptr->input_connect(data_node_ptr);
		}

		void set_end(DataNode *data_node_ptr) {
			OperatorNodeElement &output_node_ptr = op_nodes_[OUTPUT_NODE_ID];
			output_node_ptr->input_connect(data_node_ptr);
		}

	public: /* Basic information */
		size_t 	num_data_node() 						const { return data_nodes_.size(); }

		size_t  num_op_node()							const { return op_nodes_.size(); }
	};

	struct GraphControlHeader {
		friend class GraphView;
	protected:
		std::vector<DataNode *>                     data_node_ptr_array_;
		std::vector<OperatorNode *>                 op_node_ptr_array_;

		std::vector<RelaxedAtomWrapper<int>>            data_recv_count_array_;
		std::vector<RelaxedAtomWrapper<int>>            data_send_count_array_;
		std::vector<RelaxedAtomWrapper<int>>            op_recv_count_array_;

	public:
		void init_node_id() const {
			for (size_t id = 0; DataNode *data_node_ptr: data_node_ptr_array_) { data_node_ptr->info.id = id++; }
			for (size_t id = 0; OperatorNode *op_node_ptr: op_node_ptr_array_) { op_node_ptr->info.id = id++; }
		}

		void init_dep_count() {
			data_recv_count_array_.clear();
			data_send_count_array_.clear();
			op_recv_count_array_.clear();

			data_recv_count_array_.resize(data_node_ptr_array_.size(), 0);
			data_send_count_array_.resize(data_node_ptr_array_.size(), 0);
			op_recv_count_array_.resize(op_node_ptr_array_.size(), 0);

			for (OperatorNode *op_node_ptr: op_node_ptr_array_) {
				op_recv_count_array_[op_node_ptr->info.id].store(op_node_ptr->num_input(), std::memory_order_relaxed);

				for (DataNode *data_node_ptr: op_node_ptr->info.input) {
					data_node_ptr->output_connect(op_node_ptr);
				}
				for (DataNode *data_node_ptr: op_node_ptr->info.output) {
					data_recv_count_array_[data_node_ptr->info.id].value.fetch_add(1, std::memory_order_relaxed);
				}
			}
			for (DataNode *data_node_ptr: data_node_ptr_array_) {
				data_send_count_array_[data_node_ptr->info.id].store(data_node_ptr->num_output(), std::memory_order_relaxed);
			}
		}

	public:
		size_t num_data_node() const { return data_node_ptr_array_.size();  }
		size_t num_op_node()   const { return op_node_ptr_array_.size();    }

		DataNode *    data_node(NodeID id) const { return data_node_ptr_array_[id]; }
		OperatorNode *  op_node(NodeID id) const { return op_node_ptr_array_[id];   }

		RelaxedAtomWrapper<int> &data_recv(NodeID id) { return data_recv_count_array_[id]; }
		RelaxedAtomWrapper<int> &data_send(NodeID id) { return data_send_count_array_[id]; }
		RelaxedAtomWrapper<int> &  op_recv(NodeID id) { return op_recv_count_array_[id];   }

		RelaxedAtomWrapper<int> &data_recv(const DataNode *node_ptr)      { return data_recv_count_array_[node_ptr->info.id]; }
		RelaxedAtomWrapper<int> &data_send(const DataNode *node_ptr)      { return data_send_count_array_[node_ptr->info.id]; }
		RelaxedAtomWrapper<int> &  op_recv(const OperatorNode *node_ptr)  { return op_recv_count_array_[node_ptr->info.id];   }
	};

	class GraphView {
	private:
		std::vector<Graph *> graph_ptr_array_;

	public:
		GraphView() = default;

		GraphView(Graph *graph_ptr): GraphView({graph_ptr}) {}

		GraphView(std::initializer_list<Graph *> graph_view_list): graph_ptr_array_(graph_view_list) {}

		~GraphView() = default;

	public:
		GraphView &concat(GraphView &other_view) {
			graph_ptr_array_.insert(graph_ptr_array_.end(),
									other_view.graph_ptr_array_.begin(),
									other_view.graph_ptr_array_.end());
			return *this;
		}

		GraphControlHeader get_control_header() const {
			GraphControlHeader control_header;

			for (Graph *graph_ptr: graph_ptr_array_) {
				auto &array = control_header.data_node_ptr_array_;
				array.reserve(array.size() + graph_ptr->data_nodes_.size());
				for (auto &node_ptr: graph_ptr->data_nodes_) {
					array.emplace_back(node_ptr.get());
				}
			}
			for (Graph *graph_ptr: graph_ptr_array_) {
				auto &array = control_header.op_node_ptr_array_;
				array.reserve(array.size() + graph_ptr->op_nodes_.size());
				for (auto &node_ptr: graph_ptr->op_nodes_) {
					array.emplace_back(node_ptr.get());
				}
			}
			control_header.init_node_id();
			control_header.init_dep_count();
			return control_header;
		}
	};

}  // namespace spy