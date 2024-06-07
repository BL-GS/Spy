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
		friend class GraphView;
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

		void clear_output() { info.output.clear(); }

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
		friend class GraphView;
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

		void clear_input()  { info.input.clear(); }

		void clear_output() { info.output.clear(); }

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
		using DataNodeElement     = std::unique_ptr<DataNode>;

		using OperatorNodeElement = std::unique_ptr<OperatorNode>;

	protected:
		std::vector<DataNodeElement> 		data_nodes_;

		std::vector<OperatorNodeElement>	op_nodes_;

		std::string 						name_;

	public:
		Graph(const std::string_view name): name_(name) { }

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
			data_node_ptr->output_connect(op_node_ptr);
		}

	public:
		void clear_data_node()          { data_nodes_.clear(); }
		void clear_op_node()            { op_nodes_.clear(); }
		void clear_data_connection()    { for (auto &data_node: data_nodes_) { data_node->clear_output(); } }
		void clear_op_connection()      { for (auto &op_node: op_nodes_) { op_node->clear_output(); op_node->clear_input(); }}

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
		void reserve(size_t num_data_node, size_t num_op_node) {
			data_node_ptr_array_.resize(num_data_node, nullptr);
			op_node_ptr_array_.resize(num_op_node, nullptr);
			data_recv_count_array_.resize(num_data_node, 0);
			data_send_count_array_.resize(num_data_node, 0);
			op_recv_count_array_.resize(num_op_node, 0);
		}

		void init_node_id() const {
			for (size_t id = 0; DataNode *data_node_ptr: data_node_ptr_array_) { data_node_ptr->info.id = id++; }
			for (size_t id = 0; OperatorNode *op_node_ptr: op_node_ptr_array_) { op_node_ptr->info.id = id++; }
		}

		void init_dep_count() {
			data_recv_count_array_.resize(data_node_ptr_array_.size(), 0);
			data_send_count_array_.resize(data_node_ptr_array_.size(), 0);
			op_recv_count_array_.resize(op_node_ptr_array_.size(), 0);

			for (const OperatorNode *op_node_ptr: op_node_ptr_array_) {
				op_recv_count_array_[op_node_ptr->info.id].store(op_node_ptr->num_input(), std::memory_order_relaxed);

				for (const DataNode *data_node_ptr: op_node_ptr->info.output) {
					data_recv_count_array_[data_node_ptr->info.id].value.fetch_add(1, std::memory_order_relaxed);
				}
			}
			for (const DataNode *data_node_ptr: data_node_ptr_array_) {
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
		std::vector<const Graph *> graph_ptr_array_;

	public:
		GraphView() = default;

		GraphView(const Graph *graph_ptr): GraphView({graph_ptr}) {}

		GraphView(std::initializer_list<const Graph *> &&graph_view_list): graph_ptr_array_(graph_view_list) {}

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

			/* Reserve enough space */
			size_t total_data_node_num = 0;
			size_t total_op_node_num   = 0;
			for (const Graph *graph_ptr: graph_ptr_array_) { total_data_node_num += graph_ptr->num_data_node(); }
			for (const Graph *graph_ptr: graph_ptr_array_) { total_op_node_num   += graph_ptr->num_op_node();   }
			control_header.reserve(total_data_node_num, total_op_node_num);

			/* Init header */
			for (size_t cur_node_num = 0; const Graph *graph_ptr: graph_ptr_array_) {
				auto &dst_array = control_header.data_node_ptr_array_;
				auto &src_array = graph_ptr->data_nodes_;
				std::transform(src_array.begin(), src_array.end(), dst_array.begin() + cur_node_num,
							   [](auto &node_ptr){ return std::to_address(node_ptr); });
				cur_node_num += src_array.size();
			}
			// The end node of graph should be merged as an identity
			for (size_t cur_node_num = 0; const Graph *graph_ptr: graph_ptr_array_) {
				auto &dst_array = control_header.op_node_ptr_array_;
				auto &src_array = graph_ptr->op_nodes_;
				std::transform(src_array.begin(), src_array.end(), dst_array.begin() + cur_node_num,
				               [](auto &node_ptr){ return std::to_address(node_ptr); });
				cur_node_num += src_array.size();
			}

			control_header.init_node_id();
			control_header.init_dep_count();
			return control_header;
		}
	};

}  // namespace spy