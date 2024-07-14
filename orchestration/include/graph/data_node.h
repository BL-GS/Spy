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
        /// The property of the node depends on the view source
        View            = 1 << 5,

        Default         = Dynamic
    };

    enum class TensorType: uint32_t {
        Unknown,
        /* Model parameter */
        // embedding
		TokenEmbedding,
        // output
		Output, OutputNorm, RopeFrequency,
        // attention
		AttentionNorm, AttentionQ, AttentionK, AttentionV, AttentionOutput, 
        // ffn
		FFNGateInp, FFNNorm, FFNDown, FFNGate, FFNUp,

        /* Buffer tensor */
        KCache, VCache,

        /* Variable tensor */
        InputTokenId, InputTokenEmbedding, InputPosition, InputKQMask,

        OutputLogits,

        V_AttentionNorm, V_AttentionNormWeighted, 
        V_QWeighted, V_KWeighted, V_VWeighted, V_QWeightedBiased, V_KWeightedBiased, V_VWeightedBiased,
        V_QRope, V_KRope,
        V_AttentionScore, V_AttentionContext,
        V_KQV, V_KQVMerged, V_KQVWeighted,
        V_AttentionOutput,
        // KVCache

        V_FFNInput, V_FFNOutput,
        V_FFNNorm, V_FFNNormWeighted, V_FFNUp, V_FFNUpUnary, V_FFNGate, V_FFNGateUnary, 
        V_FFNPar, V_FFNDown,

        V_ResultNorm, V_ResultNormWeighted, V_ResultOutput,
        
    };

    enum class WeightType: int {
        None,
        // Constant
        Weight,
        Bias,
        Input,
        Output,
        // Variable
        View
    };

    struct DataNodeProperty {
        DataNodeType	node_type	    = DataNodeType::Default;
        TensorType      tensor_type     = TensorType::Unknown;
        WeightType      weight_type     = WeightType::None;
        int             layer_id        = -1;
        int             expert_id       = -1;
    }; 

	struct DataNode final: DataNodeProperty, BasicNode {
	public:
		/// The source of view
		DataNode *			view_src 		= nullptr;
		/// The metadata of tensor
		Tensor 				tensor;

	public:
		DataNode() = default;

		template<class ...Args>
		DataNode(DataNodeProperty property, Args &&...args) :
            DataNodeProperty(property), tensor(std::forward<Args>(args)...) {}

		DataNode(const DataNode &other) = default;

		~DataNode() noexcept = default;

    public:
        std::string get_tensor_name() const;

        std::map<std::string_view, std::string> property() const override;
	};

} // namespace spy