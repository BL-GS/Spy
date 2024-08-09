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

    struct InputBlockData {
        /* Hyper param */
        int64_t num_token;
        int64_t num_context;

        InputWeight weight;        
    };

    struct InputBlock final: public GraphBuilder, InputBlockData {
        InputBlock() = default;

        InputBlock(const InputBlockData &data): InputBlockData(data) {}

        InputBlock(InputBlock &&other) = default;

        InputBlock &operator= (InputBlock &&other) = default;

        InputBlockResult connect_input(Graph &graph);
    };

    struct OutputBlockData {
		NormRMSParam result_norm_param;

        OutputWeight weight;

		DataNode *logit_out;
    };

	struct OutputBlock final: public GraphBuilder, OutputBlockData {
        OutputBlock() = default;

        OutputBlock(const OutputBlockData &data): OutputBlockData(data) {}

        OutputBlock(OutputBlock &&other) = default;

        OutputBlock &operator= (OutputBlock &&other) = default;

		OutputBlockResult connect_output(Graph &graph) const;
	};

} // namespace spy