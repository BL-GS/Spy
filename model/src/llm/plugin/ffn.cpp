#include "llm/plugin/ffn.h"

namespace spy {

    DataNode *FFNBlock::connect_ffn(Graph &graph, int layer_id, int expert_id) const {
        const DataNodeProperty default_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = layer_id,
            .expert_id = expert_id
        };

        DataNode *ffn_norm          = make_augmented_stream<OperatorType::NormRMS>(graph, "ffn_norm",
            default_prop, norm_rms_param,
            ffn_input
        );
        DataNode *ffn_norm_weighted = make_stream<OperatorType::Mul>(graph, "ffn_norm_linear",
            default_prop,
            ffn_norm, weight.ffn_norm
        );
        DataNode *ffn_up            = make_stream<OperatorType::MatMul>(graph, "ffn_up",
            default_prop,
            weight.ffn_up, ffn_norm_weighted
        );
        DataNode *ffn_gate          = make_stream<OperatorType::MatMul>(graph, "ffn_gate",
            default_prop,
            weight.ffn_gate, ffn_norm_weighted
        );
        DataNode *ffn_gate_silu     = make_stream<OperatorType::Silu>(graph, "ffn_gate_activate",
            default_prop,
            ffn_gate
        );
        DataNode *ffn_par           = make_stream<OperatorType::Mul>(graph, "ffn_par",
            default_prop,
            ffn_up, ffn_gate_silu
        );
        DataNode *ffn_down          = make_stream<OperatorType::MatMul>(graph, "ffn_down",
            default_prop,
            weight.ffn_down, ffn_par
        );

        return ffn_down;
    }

} // namespace spy