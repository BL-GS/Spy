#pragma once

#include <magic_enum.hpp>

#include "number/tensor.h"
#include "graph/basic_node.h"

namespace spy {

    /// The type of data node. Which assist the allocation and schedule of Graph Scheduler.
    /// It depends on the relevant operator 
    enum class DataNodeType: int {
        /// The data and the metadata remains constant during the whole execution
        Constant        = 1 << 0, 
        /// The metadata remains static during the inference
        Static          = 1 << 1,
		/// The metadata of the node depends on the forward input during the inference.
		/// The data remains static
        /// For example: add
        ShapeDynamic    = 1 << 2,
        /// The data of the node depends on the forward input during the inference
		/// The metadata remains static
        /// For example: shape
        DataDynamic     = 1 << 3,
        /// The metadata and the data of the node will be changed during the inference
        /// For example: top-k
        Dynamic         = 1 << 4,

        Default         = Dynamic
    };

    struct DataNodeProperty {
        DataNodeType	node_type	    = DataNodeType::Default;
        int             layer_id        = -1;
        int             expert_id       = -1;
    }; 

	struct DataNode final: BasicNode {
	public:
		/// The source of view
		DataNode *			view_src 		= nullptr;
		/// The metadata of tensor
		Tensor 				tensor;
        /// Some properties gained from graph construction
        DataNodeProperty    node_prop;

	public:
		DataNode() = default;

		template<class ...Args>
		DataNode(const DataNodeProperty &prop, Args &&...args) :
            tensor(std::forward<Args>(args)...), node_prop(prop) {}

		DataNode(const DataNode &other) = default;

		~DataNode() noexcept = default;

    public:
        void set_prop(const DataNodeProperty &new_prop) { node_prop = new_prop; }

        void set_tensor(const Tensor &new_tensor) { tensor = new_tensor; }

        void set_view_src(DataNode *node_ptr) { view_src = node_ptr; }

    public:
        bool is_view() const { return view_src != nullptr; }

        std::string get_tensor_name() const;

        std::map<std::string_view, std::string> property() const override;
	};

} // namespace spy