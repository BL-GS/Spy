#include "llm/plugin/attention.h"

namespace spy {

    void KVCache::connect_KVCache(Graph &graph, int layer_id) {
        const DataNodeProperty cache_prop {
            .node_type = DataNodeType::Cache,
            .layer_id  = layer_id,
            .expert_id = -1
        };

        const int64_t k_num  = num_embedding_k_gqa * num_context;
        const Shape k_shape{{k_num}, k_cache_type};
		k_cache = make_augmented_stream<OperatorType::Input>(graph, InputParam{ .shape = k_shape })
					.set_name("KCache")
					.deduce(graph, cache_prop);

        const int64_t v_num  = num_embedding_v_gqa * num_context;
        const Shape v_shape{{v_num}, v_cache_type};
        v_cache = make_augmented_stream<OperatorType::Input>(graph, InputParam{ .shape = v_shape })
					.set_name("VCache")
					.deduce(graph, cache_prop);
    }

    std::tuple<DataNode *, DataNode *, DataNode *> MultiHeadAttentionBlock::input_linear_map(Graph &graph,
            const DataNodeProperty &default_prop, DataNode *attn_norm) const {

        DataNode *Q_cur    = make_stream<OperatorType::MatMul>(graph)
                                .set_name("Q_project")
								.set_input(weight.weight_q, attn_norm)
								.deduce(graph, default_prop);
        DataNode *K_cur    = make_stream<OperatorType::MatMul>(graph)
							  .set_name("K_project")
							  .set_input(weight.weight_k, attn_norm)
							  .deduce(graph, default_prop);
        DataNode *V_cur    = make_stream<OperatorType::MatMul>(graph)
							  .set_name("V_project")
							  .set_input(weight.weight_v, attn_norm)
							  .deduce(graph, default_prop);

        DataNode *Q_cur_out = (weight.bias_q == nullptr) ? Q_cur :
            make_stream<OperatorType::Add>(graph)
                .set_name("Q_project_bias")
				.set_input(Q_cur, weight.bias_q)
                .deduce(graph, default_prop);
        DataNode *K_cur_out = (weight.bias_k == nullptr) ? K_cur :
			 make_stream<OperatorType::Add>(graph)
				 .set_name("K_project_bias")
				 .set_input(K_cur, weight.bias_k)
				 .deduce(graph, default_prop);
        DataNode *V_cur_out = (weight.bias_v == nullptr) ? V_cur :
			 make_stream<OperatorType::Add>(graph)
				 .set_name("V_project_bias")
				 .set_input(V_cur, weight.bias_v)
				 .deduce(graph, default_prop);

        return { Q_cur_out, K_cur_out, V_cur_out };
    }

    std::pair<DataNode *, DataNode *> MultiHeadAttentionBlock::position_encode(Graph &graph,
            const DataNodeProperty &default_prop, 
            DataNode *Q_cur, DataNode *K_cur) {

        DataNode *Q_reshaped_cur = make_dynamic_stream<OperatorType::Reshape>(graph,
			  [this]{ return ReshapeParam{
				  .new_shape = Shape({num_embedding_head, num_head, num_token}, NumberType::FP32)
			  }; })
			.set_name("Q_project")
            .set_input(Q_cur)
			.deduce(graph, default_prop);
        DataNode *K_reshaped_cur = make_dynamic_stream<OperatorType::Reshape>(graph,
			  [this]{ return ReshapeParam{
				  .new_shape = Shape({num_embedding_head, num_head, num_token}, NumberType::FP32)
			  }; })
			.set_name("K_project")
			.set_input(K_cur)
            .deduce(graph, default_prop);
        DataNode *Q_rope = make_dynamic_stream<OperatorType::Rope>(graph,
            [this]{
                RopeParam param = rope_param_draft;
                param.num_context = num_context;
                param.num_past = num_past_token;
                return param;
            })
			.set_name("Q_rope")
			.set_input(Q_reshaped_cur, input_pos)
		    .deduce(graph, default_prop);
        DataNode *K_rope = make_dynamic_stream<OperatorType::Rope>(graph,
            [this]{
                RopeParam param = rope_param_draft;
                param.num_context = num_context;
                param.num_past = num_past_token;
                return param;
            })
			.set_name("K_rope")
			.set_input(K_reshaped_cur, input_pos)
			.deduce(graph, default_prop);

        return { Q_rope, K_rope };
    }

    DataNode *MultiHeadAttentionBlock::attention_projection(Graph &graph,
            const DataNodeProperty &default_prop, 
            DataNode *query, DataNode *key, DataNode *value) {

        DataNode *KQ_out = make_stream<OperatorType::MatMul>(graph)
            .set_name("attn_score")
            .set_input(key, query)
	        .deduce(graph, default_prop);
        DataNode *KQ_softmax_out = make_augmented_stream<OperatorType::Softmax>(graph,
			SoftmaxParam{ .scale = 1.0F / std::sqrt(static_cast<float>(num_embedding_head)) })
			.set_name("attn_score_softmax")
            .set_input(KQ_out, KQ_mask)
			.deduce(graph, default_prop);

        DataNode *KQV       = make_stream<OperatorType::MatMul>(graph)
            .set_name("KQV")
            .set_input(value, KQ_softmax_out)
			.deduce(graph, default_prop);
        DataNode *KQV_merge = make_augmented_stream<OperatorType::Permute>(graph, PermuteParam{ .axis = {0, 2, 1, 3} })
			.set_name("KQV")
			.set_input(KQV)
            .deduce(graph, default_prop);
        DataNode *KQV_merged_cont = make_stream<OperatorType::Contiguous>(graph)
            .set_name("KQV")
			.set_input(KQV_merge)
			.deduce(graph, default_prop);
        KQV_merged_cont = make_dynamic_stream<OperatorType::Reshape>(graph,
			 [this]{ return ReshapeParam{ .new_shape = Shape({num_embedding_head * num_head, num_token}, NumberType::FP32) }; })
			.set_name("KQV")
			.set_input(KQV_merged_cont)
            .deduce(graph, default_prop);
        DataNode *KQV_w_out = make_stream<OperatorType::MatMul>(graph)
            .set_name("KQV_linear")
	        .set_input(weight.weight_o, KQV_merged_cont)
            .deduce(graph, default_prop);
        DataNode *KQV_out   = (weight.bias_o == nullptr) ?  KQV_w_out :
            make_stream<OperatorType::Add>(graph)
                .set_name("KQV_linear")
	            .set_input(KQV_w_out, weight.bias_o)
                .deduce(graph, default_prop);

        return KQV_out;
    }

    std::pair<DataNode *, DataNode *> MultiHeadAttentionBlock::reshape_KV(Graph &graph,
        const DataNodeProperty &default_prop,
        DataNode *key, DataNode *value) {
        DataNode *K_out = make_augmented_stream<OperatorType::View>(graph,
            ViewParam {
                .offset    = 0,
                .new_shape = Shape(
                        { num_embedding_head, num_token, num_head_kv }, // New dimensions
                        { get_type_size(NumberType::FP32), get_row_size(NumberType::FP32, num_embedding_k_gqa), get_row_size(NumberType::FP32, num_embedding_head) }, // New offsets
                        NumberType::FP32
                    )
            })
			.set_name("VCache_view")
			.set_input(key)
			.deduce(graph, default_prop);
        // If we don't use KV cache, the value is incontiguous, which may incur performance degradation consequentially.
        // Therefore, we reconstruct it.
        DataNode *V_out = make_stream<OperatorType::Contiguous>(graph)
            .set_name("VCache")
			.set_input(value)
            .deduce(graph, default_prop);

        V_out = make_dynamic_stream<OperatorType::Reshape>(graph,
	            [this]{ return ReshapeParam{
	                .new_shape = Shape({ num_token, num_embedding_head, num_head_kv }, NumberType::FP32)
	            };
			})
			.set_name("VCache")
			.set_input(V_out)
			.deduce(graph, default_prop);

        return {K_out, V_out}; 
    }

    std::pair<DataNode *, DataNode *> MultiHeadAttentionBlock::connect_KVCache(Graph &graph,
        const DataNodeProperty &default_prop,
        DataNode *key, DataNode *value) {

        // Update key cache
        spy_assert(k_cache != nullptr, "expect the k cache not to be null");
        spy_assert(v_cache != nullptr, "expect the v cache not to be null");

        DataNode *k_cache_view = make_dynamic_stream<OperatorType::View>(graph,
            [this]{
                const int64_t k_cache_update_offset = get_row_size(k_cache_type, num_embedding_k_gqa) * num_past_token;
                return ViewParam{
                    .offset = k_cache_update_offset,
                    .new_shape = Shape(
                        { num_token * num_embedding_k_gqa },
                        { get_type_size(k_cache_type) },
                        k_cache_type                        
                )}; 
            })
			.set_name("KCache_view")
            .set_input(k_cache)
			.deduce(graph, default_prop);
        DataNode *k_cache_sync = make_stream<OperatorType::Copy>(graph)
            .set_name("KCache_cpy")
	        .set_input(key, k_cache_view)
            .deduce(graph, default_prop);
        // Concat K cache and output
        DataNode *K_out = make_dynamic_stream<OperatorType::View>(graph,
            [this]{
                const int64_t num_kv = num_past_token + num_token;

                const int64_t k_cache_update_offset = get_row_size(k_cache_type, num_embedding_k_gqa) * num_past_token;
                return ViewParam{
                    .offset = -k_cache_update_offset,
                    .new_shape = Shape(
                        { num_embedding_head, num_kv, num_head_kv }, // New dimensions
                        { get_type_size(k_cache_type), get_row_size(k_cache_type, num_embedding_k_gqa), get_row_size(k_cache_type, num_embedding_head) }, // New offsets
                        k_cache_type                            
                    )
                };
            })
			.set_name("KCache_total")
            .set_input(k_cache_sync)
			.deduce(graph, default_prop);

        // Update value cache
        spy_assert(v_cache != nullptr, "Expect the v cache not to be invalid node");

        DataNode *v_cache_view = make_dynamic_stream<OperatorType::View>(graph,
            [this]{
                const int64_t v_cache_update_offset = get_type_size(v_cache_type) * num_past_token;
                return ViewParam {
                    .offset = v_cache_update_offset,
                    .new_shape = Shape(
                        { num_token, num_embedding_v_gqa },
                        { get_type_size(v_cache_type), num_context * get_type_size(v_cache_type) },
                        v_cache_type
                )};
            })
			.set_name("VCache_view")
            .set_input(v_cache)
			.deduce(graph, default_prop);
        // A fake output for synchronization
        DataNode *v_cache_sync = make_stream<OperatorType::Copy>(graph)
            .set_name("VCache_cpy")
            .set_input(value, v_cache_view)
			.deduce(graph, default_prop);
        // Concat value cache and output
        DataNode *V_out = make_dynamic_stream<OperatorType::View>(graph,
            [this]{
                const int64_t num_kv = num_past_token + num_token;
                const int64_t v_cache_update_offset = get_type_size(v_cache_type) * num_past_token;
                return ViewParam{
                .offset = -v_cache_update_offset,
                .new_shape = Shape(
                    { num_kv, num_embedding_head, num_head_kv }, // New dimensions
                    { get_type_size(v_cache_type), get_type_size(v_cache_type) * num_context, get_type_size(v_cache_type) * num_context * num_embedding_head }, // New offset
                    v_cache_type                        
                )};
            })
			.set_name("VCache_total")
            .set_input(v_cache_sync)
			.deduce(graph, default_prop);

        return {K_out, V_out};
    }

    DataNode * MultiHeadAttentionBlock::connect_attention(Graph &graph, int layer_id, int expert_id, bool enable_kvcache) {
        const DataNodeProperty default_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = layer_id,
            .expert_id = expert_id
        };

        DataNode *attn_norm = make_augmented_stream<OperatorType::NormRMS>(graph, norm_rms_param)
            .set_name("attn_norm")
			.set_input(input_embedding)
			.deduce(graph, default_prop);
        
        DataNode *attn_norm_weighted = make_stream<OperatorType::Mul>(graph)
            .set_name("attn_norm_linear")
	        .set_input(attn_norm, weight.attention_norm)
            .deduce(graph, default_prop);

        /* Self-attention */
        const auto [Q_cur_out, K_cur_out, V_cur_out] = input_linear_map(graph, default_prop, attn_norm_weighted);
        const auto [Q_rope, K_rope] = position_encode(graph, default_prop, Q_cur_out, K_cur_out);

        DataNode *Q_out = make_augmented_stream<OperatorType::Permute>(graph, PermuteParam{ .axis={0, 2, 1, 3} })
			.set_name("Q_out")
			.set_input(Q_rope)
            .deduce(graph, default_prop);

        DataNode *V_t   = make_stream<OperatorType::Transpose>(graph)
            .set_name("V_t")
            .set_input(V_cur_out)
			.deduce(graph, default_prop);

        const auto [K_out, V_out] = (enable_kvcache) ? connect_KVCache(graph, default_prop, K_rope, V_t) : reshape_KV(graph, default_prop, K_rope, V_t);

        DataNode *attn_out = attention_projection(graph, default_prop, Q_out, K_out, V_out);
        return attn_out;
    }

} // namespace spy