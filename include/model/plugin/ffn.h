#pragma once

#include "operator/type.h"
#include "operator/operator.h"
#include "model/plugin/graph_builder.h"

namespace spy {

    struct FFNWeight {
        NodeCredit ffn_norm  = Graph::INVALID_NODE_CREDIT; 
        NodeCredit ffn_up    = Graph::INVALID_NODE_CREDIT; 
        NodeCredit ffn_gate  = Graph::INVALID_NODE_CREDIT; 
        NodeCredit ffn_down  = Graph::INVALID_NODE_CREDIT;
    };

    struct FFNBlock: public GraphBuilder {
        /* Hyper param */
        const float      ffn_norm_rms_eps;

        /* Weight */
        FFNWeight        weight;

        /* Input */
        NodeCredit ffn_input;    

        NodeCredit connect_ffn(Graph &graph, const std::string &layer_suffix) const {

			const NodeCredit ffn_norm_buffer = make_stream<OperatorType::NormRMS>(graph, "FFN - norm" + layer_suffix, { ffn_input }, ffn_norm_rms_eps);
			const NodeCredit ffn_norm        = make_stream<OperatorType::Mul>(graph, "FFN - norm" + layer_suffix, { ffn_norm_buffer, weight.ffn_norm });
			const NodeCredit ffn_up          = make_stream<OperatorType::MatMul>(graph, "FFN up" + layer_suffix, { weight.ffn_up, ffn_norm });
			const NodeCredit ffn_gate        = make_stream<OperatorType::MatMul>(graph, "FFN gate" + layer_suffix, { weight.ffn_gate, ffn_norm });
			const NodeCredit ffn_gate_silu   = make_stream<OperatorType::Silu>(graph, "FFN gate silu" + layer_suffix, { ffn_gate });
			const NodeCredit ffn_par         = make_stream<OperatorType::Mul>(graph, "FFN par" + layer_suffix, { ffn_up, ffn_gate_silu });
			const NodeCredit ffn_down        = make_stream<OperatorType::MatMul>(graph, "FFN down" + layer_suffix, { weight.ffn_down, ffn_par });

			return ffn_down;
		}
    };

} // namespace spy