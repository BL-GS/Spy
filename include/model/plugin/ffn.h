#pragma once

#include "operator/type.h"
#include "operator/operator.h"
#include "model/plugin/graph_builder.h"

namespace spy {

    struct FFNWeight {
        DataNode *ffn_norm  = nullptr; 
        DataNode *ffn_up    = nullptr; 
        DataNode *ffn_gate  = nullptr; 
        DataNode *ffn_down  = nullptr;
    };

    struct FFNBlock: public GraphBuilder {
        /* Hyper param */
        const float      ffn_norm_rms_eps;

        /* Weight */
        FFNWeight        weight;

        /* Input */
        DataNode *ffn_input;    

        DataNode *connect_ffn(Graph &graph, int layer_id = -1, int expert_id = -1) const {

			DataNode *ffn_norm          = make_stream<OperatorType::NormRMS>(graph, 
                TensorType::V_FFNNorm, layer_id, expert_id,
                { ffn_input }, ffn_norm_rms_eps
            );
			DataNode *ffn_norm_weighted = make_stream<OperatorType::Mul>(graph, 
                TensorType::V_FFNNormWeighted, layer_id, expert_id, 
                { ffn_norm, weight.ffn_norm }
            );
			DataNode *ffn_up            = make_stream<OperatorType::MatMul>(graph, 
                TensorType::V_FFNUp, layer_id, expert_id, 
                { weight.ffn_up, ffn_norm_weighted }
            );
			DataNode *ffn_gate          = make_stream<OperatorType::MatMul>(graph, 
                TensorType::V_FFNGate, layer_id, expert_id, 
                { weight.ffn_gate, ffn_norm_weighted }
            );
			DataNode *ffn_gate_silu     = make_stream<OperatorType::Silu>(graph, 
                TensorType::V_FFNGateUnary, layer_id, expert_id, 
                { ffn_gate }
            );
			DataNode *ffn_par           = make_stream<OperatorType::Mul>(graph, 
                TensorType::V_FFNPar, layer_id, expert_id, 
                { ffn_up, ffn_gate_silu }
            );
			DataNode *ffn_down          = make_stream<OperatorType::MatMul>(graph, 
                TensorType::V_FFNDown, layer_id, expert_id, 
                { weight.ffn_down, ffn_par }
            );

			return ffn_down;
		}
    };

} // namespace spy