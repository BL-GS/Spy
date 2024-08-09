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
        k_cache = make_augmented_stream<OperatorType::Input>(graph, fmt::format("KCache-{}", layer_id),
            cache_prop, InputParam{ .shape = k_shape }
        );

        const int64_t v_num  = num_embedding_v_gqa * num_context;
        const Shape v_shape{{v_num}, v_cache_type};
        v_cache = make_augmented_stream<OperatorType::Input>(graph, fmt::format("VCache-{}", layer_id),
            cache_prop, InputParam{ .shape = v_shape }
        );
    }

    std::tuple<DataNode *, DataNode *, DataNode *> MultiHeadAttentionBlock::input_linear_map(Graph &graph,
            const DataNodeProperty &default_prop, DataNode *attn_norm) const {

        DataNode *Q_cur    = make_stream<OperatorType::MatMul>(graph, "Q_project",
            default_prop,
            weight.weight_q, attn_norm
        );
        DataNode *K_cur    = make_stream<OperatorType::MatMul>(graph, "K_project",
            default_prop,
            weight.weight_k, attn_norm 
        );
        DataNode *V_cur    = make_stream<OperatorType::MatMul>(graph, "V_project",
            default_prop,
            weight.weight_v, attn_norm
        );

        DataNode *Q_cur_out = (weight.bias_q == nullptr) ? Q_cur :
            make_stream<OperatorType::Add>(graph, "Q_project_bias",
                default_prop,
                Q_cur, weight.bias_q
            );
        DataNode *K_cur_out = (weight.bias_k == nullptr) ? K_cur :
            make_stream<OperatorType::Add>(graph, "K_project_bias",
                default_prop,
                K_cur, weight.bias_k
            );
        DataNode *V_cur_out = (weight.bias_v == nullptr) ? V_cur :
            make_stream<OperatorType::Add>(graph, "V_project_bias",
                default_prop,
                V_cur, weight.bias_v
            );

        return { Q_cur_out, K_cur_out, V_cur_out };
    }

    std::pair<DataNode *, DataNode *> MultiHeadAttentionBlock::position_encode(Graph &graph,
            const DataNodeProperty &default_prop, 
            DataNode *Q_cur, DataNode *K_cur) {

        DataNode *Q_reshaped_cur = make_dynamic_stream<OperatorType::Reshape>(graph, "Q_project",
            default_prop, [this]{ return ReshapeParam{ 
                    .new_shape = Shape({num_embedding_head, num_head, num_token}, NumberType::FP32)
                }; },
            Q_cur
        );
        DataNode *K_reshaped_cur = make_dynamic_stream<OperatorType::Reshape>(graph, "K_project",
            default_prop, [this]{ return ReshapeParam{ 
                    .new_shape = Shape({num_embedding_head, num_head, num_token}, NumberType::FP32)
                }; },
            K_cur
        );

        DataNode *Q_rope = make_dynamic_stream<OperatorType::Rope>(graph, "Q_rope",
            default_prop, [this]{
                RopeParam param = rope_param_draft;
                param.num_context = num_context;
                param.num_past = num_past_token;
                return param;
            },
            Q_reshaped_cur, input_pos
        );
        DataNode *K_rope = make_dynamic_stream<OperatorType::Rope>(graph, "K_rope",
            default_prop, [this]{
                RopeParam param = rope_param_draft;
                param.num_context = num_context;
                param.num_past = num_past_token;
                return param;
            },
            K_reshaped_cur, input_pos
        );

        return { Q_rope, K_rope };
    }

    DataNode *MultiHeadAttentionBlock::attention_projection(Graph &graph,
            const DataNodeProperty &default_prop, 
            DataNode *query, DataNode *key, DataNode *value) {

        DataNode *KQ_out         = make_stream<OperatorType::MatMul>(graph, "attn_score",
            default_prop,
            key, query
        );
        DataNode *KQ_softmax_out = make_augmented_stream<OperatorType::Softmax>(graph, "attn_score_softmax",
            default_prop, SoftmaxParam{ .scale = 1.0F / std::sqrt(static_cast<float>(static_cast<int>(num_embedding_head))) },
            KQ_out, KQ_mask
        );
        
        DataNode *KQV       = make_stream<OperatorType::MatMul>(graph, "KQV",
            default_prop, 
            value, KQ_softmax_out
        );
        DataNode *KQV_merge = make_augmented_stream<OperatorType::Permute>(graph, "KQV",
            default_prop, PermuteParam{ .axis = {0, 2, 1, 3} },
            KQV 
        );
        DataNode *KQV_merged_cont = make_stream<OperatorType::Contiguous>(graph, "KQV",
            default_prop, 
            KQV_merge
        );
        KQV_merged_cont = make_dynamic_stream<OperatorType::Reshape>(graph, "KQV", 
            default_prop, [this]{ return ReshapeParam{ .new_shape = Shape({num_embedding_head * num_head, num_token}, NumberType::FP32) }; },
            KQV_merged_cont
        );
        DataNode *KQV_w_out = make_stream<OperatorType::MatMul>(graph, "KQV_linear",
            default_prop,
            weight.weight_o, KQV_merged_cont
        );
        DataNode *KQV_out              = (weight.bias_o == nullptr) ?  KQV_w_out :
            make_stream<OperatorType::Add>(graph, "KQV_linear",
                default_prop,
                KQV_w_out, weight.bias_o
            );

        return KQV_out;
    }

    std::pair<DataNode *, DataNode *> MultiHeadAttentionBlock::reshape_KV(Graph &graph,
        const DataNodeProperty &default_prop,
        DataNode *key, DataNode *value) {
        DataNode *K_out = make_augmented_stream<OperatorType::View>(graph, "KCache",
            default_prop, ViewParam {
                .offset    = 0,
                .new_shape = Shape(
                        { num_embedding_head, num_token, num_head_kv }, // New dimensions
                        { get_type_size(NumberType::FP32),
                                        get_row_size(NumberType::FP32, num_embedding_k_gqa),
                                        get_row_size(NumberType::FP32, num_embedding_head) }, // New offsets
                        NumberType::FP32
                    )
            },
            key
        );
        // If we don't use KV cache, the value is incontiguous, which may incur performance degradation consequentially.
        // Therefore, we reconstruct it.
        DataNode *V_out = make_stream<OperatorType::Contiguous>(graph, "VCache",
            default_prop, 
            value
        );	

        V_out = make_dynamic_stream<OperatorType::Reshape>(graph, "VCache", 
            default_prop, [this]{ return ReshapeParam{
                .new_shape = Shape({ num_token, num_embedding_head, num_head_kv }, NumberType::FP32)
            }; },
            V_out
        );

        return {K_out, V_out}; 
    }

    std::pair<DataNode *, DataNode *> MultiHeadAttentionBlock::connect_KVCache(Graph &graph,
        const DataNodeProperty &default_prop,
        DataNode *key, DataNode *value) {

        // Update key cache
        spy_assert(k_cache != nullptr, "expect the k cache not to be null");
        spy_assert(v_cache != nullptr, "expect the v cache not to be null");

        DataNode *k_cache_view = make_dynamic_stream<OperatorType::View>(graph, "KCache_view",
            default_prop, [this]{ 
                const int64_t k_cache_update_offset = get_row_size(k_cache_type, num_embedding_k_gqa) * num_past_token;
                return ViewParam{
                    .offset = k_cache_update_offset,
                    .new_shape = Shape(
                        { num_token * num_embedding_k_gqa },
                        std::initializer_list<int64_t>{ get_type_size(k_cache_type) },
                        k_cache_type                        
                )}; 
            },
            k_cache
        );
        DataNode *k_cache_sync = make_stream<OperatorType::Copy>(graph, "KCache_cpy",
            default_prop, 
            key, k_cache_view
        );
        // Concat K cache and output
        DataNode *K_out = make_dynamic_stream<OperatorType::View>(graph, "KCache_total",
            default_prop, [this]{ 
                const int64_t num_kv = num_past_token + num_token;

                const int64_t k_cache_update_offset = get_row_size(k_cache_type, num_embedding_k_gqa) * num_past_token;
                return ViewParam{
                    .offset = -k_cache_update_offset,
                    .new_shape = Shape(
                        { num_embedding_head, num_kv, num_head_kv }, // New dimensions
                        std::initializer_list<int64_t>{ get_type_size(k_cache_type),
                                                        get_row_size(k_cache_type, num_embedding_k_gqa),
                                                        get_row_size(k_cache_type, num_embedding_head) }, // New offsets
                        k_cache_type                            
                    )
                };
            },
            k_cache_sync
        );

        // Update value cache
        spy_assert(v_cache != nullptr, "Expect the v cache not to be invalid node");

        DataNode *v_cache_view = make_dynamic_stream<OperatorType::View>(graph, "VCache_view",
            default_prop, [this]{ 
                const int64_t v_cache_update_offset = get_type_size(v_cache_type) * num_past_token;
                return ViewParam {
                    .offset = v_cache_update_offset,
                    .new_shape = Shape(
                        { num_token, num_embedding_v_gqa },
                        std::initializer_list<int64_t>{ get_type_size(v_cache_type), num_context * get_type_size(v_cache_type) },
                        v_cache_type
                )};
            },
            v_cache
        );
        // A fake output for synchronization
        DataNode *v_cache_sync = make_stream<OperatorType::Copy>(graph, "VCache_cpy",
            default_prop,
            value, v_cache_view
        );
        // Concat value cache and output
        DataNode *V_out = make_dynamic_stream<OperatorType::View>(graph, "VCache_total",
            default_prop, [this]{ 
                const int64_t num_kv = num_past_token + num_token;
                const int64_t v_cache_update_offset = get_type_size(v_cache_type) * num_past_token;
                return ViewParam{
                .offset = -v_cache_update_offset,
                .new_shape = Shape(
                    { num_kv, num_embedding_head, num_head_kv }, // New dimensions
                    std::initializer_list<int64_t>{ get_type_size(v_cache_type),
                                                    get_type_size(v_cache_type) * num_context,
                                                    get_type_size(v_cache_type) * num_context * num_embedding_head }, // New offset
                    v_cache_type                        
                )};
            },
            v_cache_sync
        );

        return {K_out, V_out};
    }

    DataNode * MultiHeadAttentionBlock::connect_attention(Graph &graph, int layer_id, int expert_id, bool enable_kvcache) {
        const DataNodeProperty default_prop {
            .node_type = DataNodeType::Dynamic,
            .layer_id  = layer_id,
            .expert_id = expert_id
        };

        DataNode *attn_norm = make_augmented_stream<OperatorType::NormRMS>(graph, "attn_norm",
            default_prop, norm_rms_param, 
            input_embedding
        );
        
        DataNode *attn_norm_weighted = make_stream<OperatorType::Mul>(graph, "attn_norm_linear",
            default_prop,
            attn_norm, weight.attention_norm
        );

        /* Self-attention */
        const auto [Q_cur_out, K_cur_out, V_cur_out] = input_linear_map(graph, default_prop, attn_norm_weighted);
        const auto [Q_rope, K_rope] = position_encode(graph, default_prop, Q_cur_out, K_cur_out);

        DataNode *Q_out = make_augmented_stream<OperatorType::Permute>(graph, "Q_out",
            default_prop, PermuteParam{ .axis={0, 2, 1, 3} },
            Q_rope
        );
        
        DataNode *V_t   = make_stream<OperatorType::Transpose>(graph, "V_t",
            default_prop,
            V_cur_out
        );

        const auto [K_out, V_out] = (enable_kvcache) ? connect_KVCache(graph, default_prop, K_rope, V_t) : reshape_KV(graph, default_prop, K_rope, V_t);

        DataNode *attn_out = attention_projection(graph, default_prop, Q_out, K_out, V_out);
        return attn_out;
    }

} // namespace spy