#pragma once

#include <span>

#include "util/wrapper/atomic.h"
#include "graph/basic_node.h"
#include "graph/data_node.h"
#include "graph/op_node.h"
#include "graph/graph.h"

namespace spy {

	class ModelLoader;

	struct GraphControlHeader {
		using Count		= RelaxedAtomWrapper<int>;
		using CountSpan = std::span<Count>;

	public:
		CountSpan data_input_array_;
		CountSpan data_output_array_;
		CountSpan op_input_array_;

		ModelLoader *loader_ptr_ = nullptr;

	public:
		size_t num_data_node() const { return data_input_array_.size(); }

		size_t num_op_node() const { return op_input_array_.size(); }

	public:
		Count &data_input(NodeID id)  { return data_input_array_[id]; }

		Count &data_output(NodeID id) { return data_output_array_[id]; }

		Count &  op_input(NodeID id)  { return op_input_array_[id];   }

		Count &data_input(const DataNode *node_ptr)      { return data_input_array_[GraphStorage::id2idx(node_ptr->id)]; }

		Count &data_output(const DataNode *node_ptr)     { return data_output_array_[GraphStorage::id2idx(node_ptr->id)]; }

		Count &  op_input(const OperatorNode *node_ptr)  { return op_input_array_[GraphStorage::id2idx(node_ptr->id)];   }
	};

} // namespace spt