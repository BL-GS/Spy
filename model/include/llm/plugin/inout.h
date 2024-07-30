#pragma once

#include "operator/type.h"
#include "operator/operator.h"
#include "llm/plugin/graph_builder.h"

namespace spy {

    struct InputWeight {
        DataNode *token_embedding	= nullptr;	
    };

	struct OutputWeight {
        DataNode *output_norm		= nullptr;
        DataNode *output_weight     = nullptr;	
	};
    
    struct InputBlockResult {
        DataNode *input_token_id  = nullptr;
        DataNode *input_embedding = nullptr;
        DataNode *input_pos       = nullptr;
        DataNode *KQ_mask         = nullptr;
    };

    struct OutputBlockResult {
        DataNode *output_logits = nullptr;
    };

    struct InputBlock final: public GraphBuilder {
        /* Hyper param */
        int64_t num_token;
        int64_t num_context;

        InputWeight weight;

        InputBlockResult connect_input(Graph &graph);
    };

	struct OutputBlock final: public GraphBuilder {
		NormRMSParam result_norm_param;

        OutputWeight weight;

		DataNode *logit_out;

		OutputBlockResult connect_output(Graph &graph) const;
	};

} // namespace spy