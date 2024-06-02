#pragma once

#include <string>
#include <fmt/format.h>

#include "util/shell/logger.h"

namespace spy {

    /// The type of data node. Which assist the allocation and schedule of Graph Scheduler.
    /// TODO: It should be determined by the schedule policy.
    enum class DataNodeType: int {
        /// The location and the size remains fixed during the runtime.
        /// It cannot be the output node.
        /// The scheduler SHOULD NOT allocate or deallocate it.
        Constant = 0, 
        /// The content remain fixed, but it wasn't built as a constant node.
        /// Compared to Constant node, it can be the output node, which means its content can be changed during runtime.
        /// The scheduler SHOULD NOT allocate or deallocate it.
        Buffered = 1,
        /// The location and the size may change during the runtime.
        /// The scheduler CAN determine the allocation and deallocation.
        Variable = 2,
        /// The data is owned by another tensor.
        /// The scheduler SHOULD NOT allocate or deallocate it+
        View     = 3,
        /// By default, we set node as variable, scheduler conventionally allocate it every time.
        Default  = Variable,

        Unknwon
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

    inline std::string get_tensor_base_name(TensorType type, int layer_id = -1, int expert_id = -1) {
        const auto layer_format = [layer_id](std::string_view base){ 
            return fmt::vformat(base, fmt::make_format_args(layer_id)); 
        };
        [[maybe_unused]] const auto expert_format = [expert_id](std::string_view base){ 
            return fmt::vformat(base, fmt::make_format_args(expert_id)); 
        };
        [[maybe_unused]] const auto layer_expert_format = [layer_id, expert_id](std::string_view base){
            return fmt::vformat(base, fmt::make_format_args(layer_id, expert_id)); 
        };

        switch (type) {
        /* Constant */
        case TensorType::TokenEmbedding:            return "token_embd";

        case TensorType::Output:                    return "output";
        case TensorType::OutputNorm:                return "output_norm";
        case TensorType::RopeFrequency:             return "rope_freqs";

        case TensorType::AttentionNorm:             return layer_format("blk.{}.attn_norm");
        case TensorType::AttentionQ:                return layer_format("blk.{}.attn_q");
        case TensorType::AttentionK:                return layer_format("blk.{}.attn_k");
        case TensorType::AttentionV:                return layer_format("blk.{}.attn_v");
        case TensorType::AttentionOutput:           return layer_format("blk.{}.attn_output");

        case TensorType::FFNGateInp:                return layer_format("blk.{}.ffn_gate_inp");
        case TensorType::FFNNorm:                   return layer_format("blk.{}.ffn_norm");
        case TensorType::FFNDown:                   return layer_format("blk.{}.ffn_down");
        case TensorType::FFNGate:                   return layer_format("blk.{}.ffn_gate");
        case TensorType::FFNUp:                     return layer_format("blk.{}.ffn_up");

        /* Buffer */
        case TensorType::KCache:                    return layer_format("k-cache.{}");
        case TensorType::VCache:                    return layer_format("v-cache.{}");

        /* Variable */

        case TensorType::InputTokenId:              return "input_token_id";
        case TensorType::InputTokenEmbedding:       return "input_token_embedding";
        case TensorType::InputPosition:             return "input_position";
        case TensorType::InputKQMask:               return "input_KQmask";

        case TensorType::OutputLogits:              return "output_logits";

        case TensorType::V_AttentionNorm:           return "attention_norm";
        case TensorType::V_AttentionNormWeighted:   return "attention_norm_weighted";
        case TensorType::V_QWeighted:               return "Q_weighted";
        case TensorType::V_KWeighted:               return "K_weighted";
        case TensorType::V_VWeighted:               return "V_weighted";
        case TensorType::V_QWeightedBiased:         return "Q_weighted_biased";
        case TensorType::V_KWeightedBiased:         return "K_weighted_biased";
        case TensorType::V_VWeightedBiased:         return "V_weighted_biased";
        case TensorType::V_QRope:                   return "Q_rope";
        case TensorType::V_KRope:                   return "K_rope";
        case TensorType::V_AttentionScore:          return "attention_score";
        case TensorType::V_AttentionContext:        return "attention_context";
        case TensorType::V_KQV:                     return "KQV";
        case TensorType::V_KQVMerged:               return "KQV_merged";
        case TensorType::V_KQVWeighted:             return "KQV_weighted";
        case TensorType::V_AttentionOutput:         return "attention_output";

        case TensorType::V_FFNInput:                return "ffn_input";
        case TensorType::V_FFNOutput:               return "ffn_output";
        case TensorType::V_FFNNorm:                 return "ffn_norm";
        case TensorType::V_FFNNormWeighted:         return "ffn_norm_weighted";
        case TensorType::V_FFNUp:                   return "ffn_up";
        case TensorType::V_FFNUpUnary:              return "ffn_up_unary";
        case TensorType::V_FFNGate:                 return "ffn_gate";
        case TensorType::V_FFNGateUnary:            return "ffn_gate_unary";
        case TensorType::V_FFNPar:                  return "ffn_par";
        case TensorType::V_FFNDown:                 return "ffn_down";

        case TensorType::V_ResultNorm:              return "result_norm";
        case TensorType::V_ResultNormWeighted:      return "result_norm_weighted";
        case TensorType::V_ResultOutput:            return "result_output";
        case TensorType::Unknown:                   return "unknown";
        }
        spy_unreachable();
    }

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
    public:
        DataNodeType    node_type       = DataNodeType::Unknwon;
        TensorType      tensor_type     = TensorType::Unknown;
        WeightType      weight_type     = WeightType::None;
        int             layer_id        = -1;
        int             expert_id       = -1;

    public:
        std::string to_string() const {
            std::string res = get_tensor_base_name(tensor_type, layer_id, expert_id);

            switch (weight_type) {
            case WeightType::Weight:    res += ".weight";   break;
            case WeightType::Bias:      res += ".bias";     break;
            case WeightType::Input:     res += ".in";       break;
            case WeightType::Output:    res += ".out";      break;
            case WeightType::View:      res += ".view";     break;
            case WeightType::None:                          break;
            }
            return res;
        }
    };

    enum class SchedulerType {
        Default
    };

} // namespace spy