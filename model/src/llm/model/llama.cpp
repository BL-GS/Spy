#include "llm/model/llama.h"

namespace spy {

    constexpr NumberType KVCacheType = NumberType::FP16;

    void LLAMAModel::build_input(ModelMetaContext &context, Graph &graph) {
        pre_train.token_embedding = create_constant_tensor(context, graph,
            "token_embd", DataNodeProperty {
                .node_type = DataNodeType::Constant,
                .layer_id  = -1,
                .expert_id = -1
            }
        );
    }

    void LLAMAModel::build_output(ModelMetaContext &context, Graph &graph) {
        DataNodeProperty output_prop {
            .node_type = DataNodeType::Constant,
            .layer_id  = -1,
            .expert_id = -1				
        };

        pre_train.output_norm = create_constant_tensor(context, graph,
                "output_norm", output_prop
        );
        pre_train.output_weight = create_constant_tensor(context, graph,
                "output", output_prop
        );
        if (pre_train.output_weight == nullptr) {
            pre_train.output_weight = create_constant_tensor(context, graph,
                "token_embd", output_prop
            );
        }
    }

    void LLAMAModel::build_attention(ModelMetaContext &context, Graph &graph, int layer_id) {
        auto  &layer      = pre_train.layers[layer_id];

        DataNodeProperty layer_prop {
            .node_type = DataNodeType::Constant,
            .layer_id  = layer_id,
            .expert_id = -1	
        };

        layer.attention_norm = create_constant_tensor(context, graph, 
                "attn_norm", layer_prop);
        layer.weight_q = create_constant_tensor(context, graph, 
                "attn_q", layer_prop);
        layer.weight_k = create_constant_tensor(context, graph,
                "attn_k", layer_prop);
        layer.weight_v = create_constant_tensor(context, graph, 
                "attn_v", layer_prop);
        layer.weight_o = create_constant_tensor(context, graph, 
                "attn_output", layer_prop);

        layer.bias_q = create_constant_tensor(context, graph, 
                "attn_q", layer_prop, "bias");
        layer.bias_k = create_constant_tensor(context, graph, 
                "attn_k", layer_prop, "bias");
        layer.bias_v = create_constant_tensor(context, graph, 
                "attn_v", layer_prop, "bias");
        layer.bias_o = create_constant_tensor(context, graph, 
                "attn_output", layer_prop, "bias");
    }

    void LLAMAModel::build_ffn(ModelMetaContext &context, Graph &graph, int layer_id) {
        auto  &layer      = pre_train.layers[layer_id];

        DataNodeProperty layer_prop {
            .node_type = DataNodeType::Constant,
            .layer_id  = layer_id,
            .expert_id = -1	
        };

        layer.ffn_norm = create_constant_tensor(context, graph, 
                "ffn_norm", layer_prop);
        layer.ffn_up   = create_constant_tensor(context, graph, 
                "ffn_up", layer_prop);
        layer.ffn_gate = create_constant_tensor(context, graph, 
                "ffn_gate", layer_prop);
        layer.ffn_down = create_constant_tensor(context, graph, 
                "ffn_down", layer_prop);
    }

    void LLAMAModel::build_graph(ModelMetaContext &context, Graph &graph, ModelIO &model_io) {
        const int32_t num_layer 		  = metadata_.num_layer;

        const int64_t num_token           = model_io.token_id_array.size();
        const int64_t num_past_token      = model_io.acc_token - num_token;
        const int64_t num_head            = metadata_.num_head;
        const int64_t num_head_kv         = metadata_.num_head_kv;
        const int64_t num_embedding_head  = metadata_.num_embedding_head_v;
        const int64_t num_embedding_k_gqa = metadata_.num_embedding_k_gqa;
        const int64_t num_embedding_v_gqa = metadata_.num_embedding_v_gqa;
        const int64_t num_context         = hyper_param_.num_context;
        spy_assert(num_embedding_head == metadata_.num_embedding_head_k);
        spy_assert(num_embedding_head == metadata_.num_rot);

        const RopeParam rope_context = {
            .mode               = hyper_param_.rope_type,
            .num_past           = static_cast<int32_t>(num_past_token),
            .num_dim            = static_cast<int32_t>(metadata_.num_rot),
            .num_context        = static_cast<int32_t>(num_context),
            .num_origin_context = static_cast<int32_t>(hyper_param_.yarn_orig_ctx),

            .freq_base        = hyper_param_.rope_freq_base,
            .freq_scale       = hyper_param_.rope_freq_scale,
            .extend_factor    = hyper_param_.yarn_ext_factor,
            .attention_factor = hyper_param_.yarn_attn_factor,
            .beta_fast        = hyper_param_.yarn_beta_fast,
            .beta_slow        = hyper_param_.yarn_beta_slow,
            .xpos_base        = 0.0F,
            .xpos_down        = false
        };

        /*
         * Build up graph from pre-trained parameter
         */
        pre_train.layers.resize(num_layer);

        // Set constant tensor
        build_input(context, graph);
        build_output(context, graph);
        for (int32_t layer_id = 0; layer_id < num_layer; ++layer_id) {
            build_attention(context, graph, layer_id);
            build_ffn(context, graph, layer_id);
        }			

        /*
         * Set up initialized input
         */

        /* Set input */
        input_block = InputBlock{ {
            .num_token   = num_token,
            .num_context = num_context,
            .weight      = { .token_embedding = pre_train.token_embedding }
        } };
        auto [input_token_id, input_embedding, input_pos, KQ_mask] = input_block.connect_input(graph);

        /* Connection */
        attention_block_array.reserve(num_layer);
        ffn_block_array.reserve(num_layer);

        for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
            LLAMALayer &layer = pre_train.layers[layer_id];

            DataNodeProperty layer_prop {
                .node_type = DataNodeType::Dynamic,
                .layer_id  = layer_id,
                .expert_id = -1
            };

            KVCache &kv_cache_block = kv_cache_array.emplace_back(KVCache {{
                .num_embedding_k_gqa = num_embedding_k_gqa,
                .num_embedding_v_gqa = num_embedding_v_gqa,
                .num_context         = num_context,
                .k_cache_type        = KVCacheType,
                .v_cache_type        = KVCacheType,
                .num_past_token      = 0
            }});
            kv_cache_block.connect_KVCache(graph, layer_id);

            MultiHeadAttentionBlock &cur_attention_block = attention_block_array.emplace_back(MultiHeadAttentionBlock{{
                /* Hyper param */
                .norm_rms_param		 = NormRMSParam{ .eps = metadata_.ffn_norm_rms_eps },
                .rope_param_draft	 = rope_context,

                .num_embedding_head  = num_embedding_head,
                .num_embedding_k_gqa = num_embedding_k_gqa,
                .num_embedding_v_gqa = num_embedding_v_gqa,
                .num_head_kv         = num_head_kv,
                .num_head            = num_head,
                /* Dynamic params */
                .num_context         = num_context,
                .num_token           = num_token,
                .num_past_token      = 0,
                /* Weights */
                .weight = layer,
                /* Buffer */
                .KQ_mask      = KQ_mask,
                .k_cache      = kv_cache_block.k_cache,
                .v_cache      = kv_cache_block.v_cache,
                .k_cache_type = KVCacheType,
                .v_cache_type = KVCacheType,
                /* Input */
                .input_embedding = input_embedding,
                .input_pos		 = input_pos
            }});
            DataNode *attn_out = cur_attention_block.connect_attention(graph, layer_id, -1, true);

            /* Feed-forward network */
            DataNode *ffn_inp  = make_stream<OperatorType::Add>(graph)
                .set_name("ffn_input")
                .set_input(attn_out, input_embedding)
	            .deduce(graph, layer_prop);
            FFNBlock &cur_ffn_block = ffn_block_array.emplace_back(FFNBlock{{
                .norm_rms_param	  = NormRMSParam{ .eps = metadata_.ffn_norm_rms_eps },
                .weight			  = layer,
                .ffn_input 		  = ffn_inp					
            }});
            DataNode *ffn_out  = cur_ffn_block.connect_ffn(graph, layer_id);

            /* Output */
            DataNode *logit_out = make_stream<OperatorType::Add>(graph)
                .set_name("logit_output")
                .set_input(ffn_inp, ffn_out)
	            .deduce(graph, layer_prop);

            input_embedding = logit_out;
        }

        output_block = OutputBlock{{
            .result_norm_param = NormRMSParam{ .eps = metadata_.ffn_norm_rms_eps },
            .weight = { 
                .output_norm = pre_train.output_norm,
                .output_weight = pre_train.output_weight
            },
            .logit_out = input_embedding
        }};
        auto [output_logits] = output_block.connect_output(graph);

        /* Connect IO */
        input = {
            .input_token_id = input_token_id,
            .input_pos = input_pos,
            .KQ_mask = KQ_mask
        };
        output = {
            .output_logits = output_logits
        };

        /* Notify all listeners on parameters */
        notify_listeners();
        input_block.notify_listeners();
        for (auto &kv_cache_block: kv_cache_array) { kv_cache_block.notify_listeners(); }
        for (auto &attention_block: attention_block_array) { attention_block.notify_listeners(); }
        for (auto &ffn_block: ffn_block_array) { ffn_block.notify_listeners(); }
        output_block.notify_listeners();

        /* Propagate parameters */
        graph.storage_ptr->propagate();

        /* Allocate and assign input/output */
        model_io.logits.resize(output.output_logits->tensor.total_element());

        {  // Build KQ Mask
            spy_assert(num_past_token + num_token <= num_context, "the number of tokens excesses the length of context");
            kq_mask_.resize(num_token * num_context, -INFINITY);
            kq_mask_.assign(num_token * num_context, -INFINITY);
            for (int64_t i_token = 0; i_token < num_token; ++i_token) {
                for (int64_t j_token = 0; j_token <= i_token + num_past_token; ++j_token) {
                    kq_mask_[num_context * i_token + j_token] = 0.0F;
                }
            }
        }

        // We don't need to translate input token into the embedding
        input.input_token_id->tensor.set_data_ptr(model_io.token_id_array.data());
        input.input_pos->tensor.set_data_ptr(model_io.positions.data());
        input.KQ_mask->tensor.set_data_ptr(kq_mask_.data());
        output.output_logits->tensor.set_data_ptr(model_io.logits.data());
    }

    void LLAMAModel::propagate(Graph &graph, ModelIO &model_io) {
        /* Get the dynamic parameter */
        const int64_t num_token 	 = model_io.token_id_array.size();
        const int64_t acc_token      = model_io.acc_token;
        const int64_t num_past_token = acc_token - num_token;
        const int64_t num_context    = hyper_param_.num_context;

        /* Set the dynamic parameter */
        input_block.num_token   = num_token;
        for (auto &attention_block: attention_block_array) { 
            attention_block.num_token      = num_token;
            attention_block.num_past_token = num_past_token;
        }
        for (auto &kv_cache_block: kv_cache_array) {
            kv_cache_block.num_past_token = num_past_token;
        }

        /* Notify all listeners on parameters */
        notify_listeners();
        input_block.notify_listeners();
        for (auto &kv_cache_block: kv_cache_array) { kv_cache_block.notify_listeners(); }
        for (auto &attention_block: attention_block_array) { attention_block.notify_listeners(); }
        for (auto &ffn_block: ffn_block_array) { ffn_block.notify_listeners(); }
        output_block.notify_listeners();

        /* Propagate parameters */
        graph.storage_ptr->propagate();

        /* Connect IO */

        {  // Build KQ Mask
            spy_assert(num_past_token + num_token <= num_context, "the number of tokens excesses the length of context");
            kq_mask_.resize(num_token * num_context, -INFINITY);
            kq_mask_.assign(num_token * num_context, -INFINITY);
            for (int64_t i_token = 0; i_token < num_token; ++i_token) {
                for (int64_t j_token = 0; j_token <= i_token + num_past_token; ++j_token) {
                    kq_mask_[num_context * i_token + j_token] = 0.0F;
                }
            }
        }

        model_io.logits.resize(output.output_logits->tensor.total_element());

        input.input_token_id->tensor.set_data_ptr(model_io.token_id_array.data());
        input.input_pos->tensor.set_data_ptr(model_io.positions.data());
        input.KQ_mask->tensor.set_data_ptr(kq_mask_.data());
        output.output_logits->tensor.set_data_ptr(model_io.logits.data());
    }

} // namespace spy