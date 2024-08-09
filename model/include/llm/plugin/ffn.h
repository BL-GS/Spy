#pragma once

#include "operator/type.h"
#include "operator/operator.h"
#include "llm/plugin/graph_builder.h"

namespace spy {

    struct FFNWeight {
        DataNode *ffn_norm  = nullptr; 
        DataNode *ffn_up    = nullptr; 
        DataNode *ffn_gate  = nullptr; 
        DataNode *ffn_down  = nullptr;
    };

    struct FFNBlockData {
        /* Hyper param */
        NormRMSParam     norm_rms_param;

        /* Weight */
        FFNWeight        weight;

        /* Input */
        DataNode *       ffn_input;    
    };

    struct FFNBlock final: public GraphBuilder, FFNBlockData {

        FFNBlock() = default;

        FFNBlock(const FFNBlockData &data): FFNBlockData(data) {}

        FFNBlock(FFNBlock &&other) = default;

        FFNBlock &operator= (FFNBlock &&other) = default;

        DataNode *connect_ffn(Graph &graph, int layer_id = -1, int expert_id = -1) const;
    };

} // namespace spy