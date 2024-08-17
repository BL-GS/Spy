#include "llm/plugin/inout.h"

namespace spy {

    InputBlockResult InputBlock::connect_input(Graph &graph) {
        InputBlockResult res;

        DataNodeProperty default_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = -1,
            .expert_id = -1
        };

        const DataNodeProperty input_prop {
            .node_type = DataNodeType::IO,
            .layer_id  = -1,
            .expert_id = -1
        };

        res.input_token_id = make_dynamic_stream<OperatorType::Input>(graph,
            [this]{  
                return InputParam{ Shape(1, {num_token}, NumberType::INT32) };
            })
			.set_name("input_token_id")
			.deduce(graph, input_prop);
        res.input_embedding = make_stream<OperatorType::GetRow>(graph)
            .set_name("input_embedding")
            .set_input(weight.token_embedding, res.input_token_id)
			.deduce(graph, default_prop);

        res.input_pos = make_dynamic_stream<OperatorType::Input>(graph,
            [this]{
				 return InputParam{ .shape = Shape(1, { num_token }, NumberType::INT32 ) };
			})
			.set_name("input_position")
			.deduce(graph, input_prop);

        res.KQ_mask = make_dynamic_stream<OperatorType::Input>(graph,
            [this]{
			   return InputParam{ .shape = Shape(2, { num_context, num_token }, NumberType::INT32 ) };
		    })
			.set_name("KQ_mask")
			.deduce(graph, input_prop);

        return res;
    }

    OutputBlockResult OutputBlock::connect_output(Graph &graph) const {
        DataNodeProperty default_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = -1,
            .expert_id = -1
        };

        DataNodeProperty output_prop {
            .node_type = DataNodeType::IO,
            .layer_id  = -1,
            .expert_id = -1
        };

        DataNode *result_norm = make_augmented_stream<OperatorType::NormRMS>(graph, result_norm_param)
            .set_name("result_norm")
			.set_input(logit_out)
			.deduce(graph, default_prop);

        DataNode *result_norm_weighted = make_stream<OperatorType::Mul>(graph)
            .set_name("result_linear")
            .set_input(result_norm, weight.output_norm)
			.deduce(graph, default_prop);
        
        // Final output
        DataNode *output = make_stream<OperatorType::MatMul>(graph)
            .set_name("output")
            .set_input(weight.output_weight, result_norm_weighted)
			.deduce(graph, output_prop);

        return { output };
    }

} // namespace spy