#include "llm/model/llama.h"

namespace spy {

    void LLAMAModel::build_input(Graph &graph) {
        const ModelMetaContext &context = *context_ptr_;
        pre_train.token_embedding = create_constant_tensor(context, graph,
            "token_embd.weight", DataNodeProperty {
                .node_type = DataNodeType::Constant,
                .layer_id  = -1,
                .expert_id = -1
            }
        );
    }

    void LLAMAModel::build_output(Graph &graph) {
        const ModelMetaContext &context = *context_ptr_;

        DataNodeProperty output_prop {
            .node_type = DataNodeType::Constant,
            .layer_id  = -1,
            .expert_id = -1				
        };

        pre_train.output_norm = create_constant_tensor(context, graph,
                "output_norm.weight", output_prop
        );
        pre_train.output_weight = create_constant_tensor(context, graph,
                "output.weight", output_prop
        );
        if (pre_train.output_weight == nullptr) {
            pre_train.output_weight = create_constant_tensor(context, graph,
                "token_embd.weight", output_prop
            );
        }
    }

    void LLAMAModel::build_attention(Graph &graph, int layer_id) {
        const ModelMetaContext &context = *context_ptr_;
        auto  &layer      = pre_train.layers[layer_id];

        DataNodeProperty layer_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = layer_id,
            .expert_id = -1	
        };

        layer.attention_norm = create_constant_tensor(context, graph, 
                "attn_norm.weight", layer_prop);
        layer.weight_q = create_constant_tensor(context, graph, 
                "attn_q.weight", layer_prop);
        layer.weight_k = create_constant_tensor(context, graph,
                "attn_k.weight", layer_prop);
        layer.weight_v = create_constant_tensor(context, graph, 
                "attn_v.weight", layer_prop);
        layer.weight_o = create_constant_tensor(context, graph, 
                "attn_o.weight", layer_prop);

        layer.bias_q = create_constant_tensor(context, graph, 
                "attn_q.bias", layer_prop);
        layer.bias_k = create_constant_tensor(context, graph, 
                "attn_k.bias", layer_prop);
        layer.bias_v = create_constant_tensor(context, graph, 
                "attn_v.bias", layer_prop);
        layer.bias_o = create_constant_tensor(context, graph, 
                "attn_o.bias", layer_prop);
    }

    void LLAMAModel::build_ffn(Graph &graph, int layer_id) {
        const ModelMetaContext &context = *context_ptr_;
        auto  &layer      = pre_train.layers[layer_id];

        DataNodeProperty layer_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = layer_id,
            .expert_id = -1	
        };

        layer.ffn_norm = create_constant_tensor(context, graph, 
                "ffn_norm.weight", layer_prop);
        layer.ffn_up   = create_constant_tensor(context, graph, 
                "ffn_up.weight", layer_prop);
        layer.ffn_gate = create_constant_tensor(context, graph, 
                "ffn_gate.weight", layer_prop);
        layer.ffn_down = create_constant_tensor(context, graph, 
                "ffn_down.weight", layer_prop);
    }

    void LLAMAModel::build_kv_cache(Graph &graph, int layer_id) {
        auto  &layer      = pre_train.layers[layer_id];

        DataNodeProperty layer_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = layer_id,
            .expert_id = -1	
        };
        const int64_t num_embedding_k_gqa = metadata_.num_embedding_k_gqa;
        const int64_t num_embedding_v_gqa = metadata_.num_embedding_v_gqa;
        const int64_t num_context		 = hyper_param_.num_context;

        layer.k_cache = create_tensor(graph, "KCache",
            Shape(1, { num_embedding_k_gqa * num_context }, NumberType::FP16),
            layer_prop, kv_cache_ptr_->k_cache[layer_id].get()
        );
        layer.v_cache = create_tensor(graph, "VCache",
            Shape(1, { num_embedding_v_gqa * num_context }, NumberType::FP16), 
            layer_prop, kv_cache_ptr_->v_cache[layer_id].get()
        );
    }

    void LLAMAModel::build_graph(Graph &graph, ModelIO &model_io) {
        const uint32_t num_layer 		 = metadata_.num_layer;

        const int64_t num_token 			 = model_io.num_token();
        const int64_t num_vocab			 = metadata_.num_vocab;
        const int64_t num_head		 	 = metadata_.num_head;
        const int64_t num_head_kv		 = metadata_.num_head_kv;
        const int64_t num_embedding_head  = metadata_.num_embedding_head_v;
        const int64_t num_embedding_k_gqa = metadata_.num_embedding_k_gqa;
        const int64_t num_embedding_v_gqa = metadata_.num_embedding_v_gqa;
        spy_assert(num_embedding_head == metadata_.num_embedding_head_k);
        spy_assert(num_embedding_head == metadata_.num_rot);
        const int64_t num_context		 = (USE_KV_CACHE) ? hyper_param_.num_context: num_token;

        const RopeParam rope_context = {
            .mode               = hyper_param_.rope_type,
            .num_past           = 0,
            .num_dim            = static_cast<int32_t>(metadata_.num_rot),
            .num_context        = 0,
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

        // Set KV Cache
        if constexpr (USE_KV_CACHE) {
            if (kv_cache_ptr_ == nullptr) { kv_cache_ptr_ = std::make_unique<KVCache>(); }
            kv_cache_ptr_->reserve(num_embedding_k_gqa, num_embedding_v_gqa,
                                    num_context, num_layer);
            for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
                build_kv_cache(graph, layer_id);
            }
        }
        // Set constant tensor
        build_input(graph);
        build_output(graph);
        for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
            build_attention(graph, layer_id);
            build_ffn(graph, layer_id);
        }			

        /*
         * Set up initialized input
         */

        /* Build KQ Mask */
        {
            const size_t past_kv = (kv_cache_ptr_ == nullptr) ? 0 : kv_cache_ptr_->head;

            kq_mask_.resize(num_token * num_context, -INFINITY);
            kq_mask_.assign(num_token * num_context, -INFINITY);
            for (size_t i_token = 0; i_token < num_token; ++i_token) {
                for (size_t j_token = 0; j_token <= i_token + past_kv; ++j_token) {
                    kq_mask_[num_context * i_token + j_token] = 0.0F;
                }
            }
        }

        const DataNodeProperty inout_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = -1,
            .expert_id = -1
        };

        const DataNodeProperty default_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = -1,
            .expert_id = -1
        };

        /* Set input */
        input_block = {
            .num_token   = num_token,
            .num_context = num_context,
            .weight      = { .token_embedding = pre_train.token_embedding }
        };
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

            MultiHeadAttentionBlock &cur_attention_block = attention_block_array.emplace_back(MultiHeadAttentionBlock{
                /* Hyper param */
                .norm_rms_param		 = NormRMSParam{ .eps = metadata_.ffn_norm_rms_eps },
                .rope_param			 = rope_context,

                .num_embedding_head  = num_embedding_head,
                .num_embedding_k_gqa = num_embedding_k_gqa,
                .num_embedding_v_gqa = num_embedding_v_gqa,
                .num_head_kv         = num_head_kv,
                .num_head            = num_head,
                /* Dynamic params */
                .num_context         = num_context,
                .num_token           = num_token,
                .num_past_token      = kv_cache_ptr_->head,
                /* Weights */
                .weight = layer,
                /* Buffer */
                .KQ_mask = KQ_mask,
                .k_cache = layer.k_cache,
                .v_cache = layer.v_cache,
                /* Input */
                .input_embedding = input_embedding,
                .input_pos		 = input_pos
            });
            DataNode *attn_out = cur_attention_block.connect_attention(graph, layer_id);

            /* Feed-forward network */
            DataNode *ffn_inp  = make_stream<OperatorType::Add>(graph, "ffn_input",
                layer_prop,
                attn_out, input_embedding
            );
            FFNBlock &cur_ffn_block = ffn_block_array.emplace_back(FFNBlock{
                .norm_rms_param	  = NormRMSParam{ .eps = metadata_.ffn_norm_rms_eps },
                .weight			  = layer,
                .ffn_input 		  = ffn_inp					
            });
            DataNode *ffn_out  = cur_ffn_block.connect_ffn(graph, layer_id);

            /* Output */
            DataNode *logit_out = make_stream<OperatorType::Add>(graph, "logit_output",
                layer_prop,
                ffn_inp, ffn_out
            );

            input_embedding = logit_out;
        }

        output_block = {
            .result_norm_param = NormRMSParam{ .eps = metadata_.ffn_norm_rms_eps },
            .weight = { 
                .output_norm = pre_train.output_norm,
                .output_weight = pre_train.output_weight
            },
            .logit_out = input_embedding
        };
        DataNode *output = output_block.connect_output(graph);

        if constexpr (USE_KV_CACHE) {
            kv_cache_ptr_->step(num_token);
        }
    }

    void LLAMAModel::propagate(ModelIO &model_io) {
        /* Get the dynamic parameter */
        const int64_t num_token 	= model_io.num_token();
        const int64_t num_context	= (USE_KV_CACHE) ? hyper_param_.num_context: num_token;

        /* Set the dynamic parameter */
        input_block.num_token   = num_token;
        input_block.num_context = num_context;
        for (auto &attention_block: attention_block_array) { 
            attention_block.num_token   = num_token;
            attention_block.num_context = num_context;
            attention_block.num_past_token = kv_cache_ptr_->head;
        }
        if constexpr (USE_KV_CACHE) {
            kv_cache_ptr_->step(num_token);
        }

        /* Notify all listeners on parameters */
        input_block.notify_listeners();
        for (auto &attention_block: attention_block_array) { attention_block.notify_listeners(); }
        for (auto &ffn_block: ffn_block_array) { ffn_block.notify_listeners(); }
        output_block.notify_listeners();

        /* Remember to propagate in order to broadcast the effect to the whole graph */
    }

} // namespace spy