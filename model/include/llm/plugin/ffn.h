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

    struct FFNBlock: public GraphBuilder {
        /* Hyper param */
        NormRMSParam     ffn_norm_rms_eps;

        /* Weight */
        FFNWeight        weight;

        /* Input */
        DataNode *ffn_input;    

        DataNode *connect_ffn(Graph &graph, int layer_id = -1, int expert_id = -1) const;
    };

} // namespace spy