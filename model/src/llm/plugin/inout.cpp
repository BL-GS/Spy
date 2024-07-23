#include "llm/plugin/inout.h"

namespace spy {

    InputBlockResult InputBlock::connect_input(Graph &graph) {
        InputBlockResult res;

        const DataNodeProperty input_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = -1,
            .expert_id = -1
        };

        res.input_token_id = make_dynamic_stream<OperatorType::Input>(graph, "input_token_id", input_prop, 
            [this]{  
                return InputParam{ Shape(1, {num_token}, NumberType::INT32) };
            }
        );
        res.input_embedding = make_stream<OperatorType::GetRow>(graph, "input_embedding",
            input_prop, 
            weight.token_embedding, res.input_token_id
        );

        res.input_pos = make_dynamic_stream<OperatorType::Input>(graph, "input_position", input_prop,
            [this]{ return InputParam{
                .shape = Shape(1, { num_token }, NumberType::INT32 )
            }; }
        );

        res.KQ_mask = make_dynamic_stream<OperatorType::Input>(graph, "KQ_mask", input_prop,
            [this]{ return InputParam{
                .shape = Shape(2, { num_context, num_token }, NumberType::INT32 )
            }; }
        );

        graph.entry_point_array.emplace_back(dynamic_cast<OperatorNode *>(res.input_token_id->input(0)));
        graph.entry_point_array.emplace_back(dynamic_cast<OperatorNode *>(res.input_pos->input(0)));
        graph.entry_point_array.emplace_back(dynamic_cast<OperatorNode *>(res.KQ_mask->input(0)));
        return res;
    }

    DataNode *OutputBlock::connect_output(Graph &graph) const {
        DataNodeProperty default_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = -1,
            .expert_id = -1
        };

        DataNode *result_norm = make_augmented_stream<OperatorType::NormRMS>(graph, "result_norm",
            default_prop, result_norm_param,
            logit_out
        );

        DataNode *result_norm_weighted = make_stream<OperatorType::Mul>(graph, "result_linear",
            default_prop,
            result_norm, weight.output_norm
        );
        
        // Final output
        DataNode *output = make_stream<OperatorType::MatMul>(graph, "output",
            default_prop, 
            weight.output_weight, result_norm_weighted
        );

        return output;
    }

} // namespace spy