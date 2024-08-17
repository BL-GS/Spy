#include "llm/plugin/ffn.h"

namespace spy {

    DataNode *FFNBlock::connect_ffn(Graph &graph, int layer_id, int expert_id) const {
        const DataNodeProperty default_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = layer_id,
            .expert_id = expert_id
        };

        DataNode *ffn_norm          = make_augmented_stream<OperatorType::NormRMS>(graph, norm_rms_param)
            .set_name("ffn_norm")
            .set_input(ffn_input)
			.deduce(graph, default_prop);
        DataNode *ffn_norm_weighted = make_stream<OperatorType::Mul>(graph)
            .set_name("ffn_norm_linear")
            .set_input(ffn_norm, weight.ffn_norm)
			.deduce(graph, default_prop);
        DataNode *ffn_up            = make_stream<OperatorType::MatMul>(graph)
            .set_name("ffn_up")
            .set_input(weight.ffn_up, ffn_norm_weighted)
			.deduce(graph, default_prop);
        DataNode *ffn_gate          = make_stream<OperatorType::MatMul>(graph)
            .set_name("ffn_gate")
            .set_input(weight.ffn_gate, ffn_norm_weighted)
			.deduce(graph, default_prop);
        DataNode *ffn_gate_silu     = make_stream<OperatorType::Silu>(graph)
            .set_name("ffn_gate_activate")
            .set_input(ffn_gate)
			.deduce(graph, default_prop);
        DataNode *ffn_par           = make_stream<OperatorType::Mul>(graph)
            .set_name("ffn_par")
            .set_input(ffn_up, ffn_gate_silu)
			.deduce(graph, default_prop);
        DataNode *ffn_down          = make_stream<OperatorType::MatMul>(graph)
            .set_name("ffn_down")
            .set_input(weight.ffn_down, ffn_par)
			.deduce(graph, default_prop);

        return ffn_down;
    }

} // namespace spy