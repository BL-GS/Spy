#pragma once

#include "operator/type.h"
#include "operator/operator.h"
#include "model/plugin/graph_builder.h"

namespace spy {

    struct MultiHeadAttentionWeight {
        NodeCredit attention_norm  = Graph::INVALID_NODE_CREDIT;

        NodeCredit weight_q        = Graph::INVALID_NODE_CREDIT;
        NodeCredit weight_k        = Graph::INVALID_NODE_CREDIT;
        NodeCredit weight_v        = Graph::INVALID_NODE_CREDIT;
        NodeCredit weight_o        = Graph::INVALID_NODE_CREDIT;

        NodeCredit bias_q          = Graph::INVALID_NODE_CREDIT;
        NodeCredit bias_k          = Graph::INVALID_NODE_CREDIT;
        NodeCredit bias_v          = Graph::INVALID_NODE_CREDIT;
        NodeCredit bias_o          = Graph::INVALID_NODE_CREDIT;
    };

    struct MultiHeadAttentionBlock: public GraphBuilder {
        /* Hyper params */
        const float   ffn_norm_rms_eps;

        const size_t  num_embedding_head;
        const size_t  num_embedding_k_gqa;
        const size_t  num_embedding_v_gqa;
        const size_t  num_head_kv;
        const size_t  num_head;

        const size_t  num_context;
        const size_t  num_token;
        const size_t  num_past_token;

        const RopeContext rope_context;

        /* Weights */
        MultiHeadAttentionWeight weight;

        /* Buffer */
        NodeCredit KQ_mask;
        NodeCredit k_cache;
        NodeCredit v_cache;

        /* Input */
        NodeCredit  input_embedding;
        NodeCredit  input_pos;


        template<bool T_use_kv_cache>
        NodeCredit connect_attention(Graph &graph, const std::string &layer_suffix) const {

            NodeCredit attn_norm_buffer = make_stream<OperatorType::NormRMS>(graph, "attn norm buffer" + layer_suffix, { input_embedding }, ffn_norm_rms_eps);
            NodeCredit attn_norm        = make_stream<OperatorType::Mul>(graph, "attn norm" + layer_suffix, { attn_norm_buffer, weight.attention_norm });

            /* Self-attention */
            const auto [Q_cur_out, K_cur_out, V_cur_out] = input_linear_map(graph, layer_suffix, attn_norm);
            const auto [Q_rope, K_rope] = position_encode(graph, layer_suffix, Q_cur_out, K_cur_out);

            NodeCredit Q_out = make_stream<OperatorType::Permute>(graph, "Q - out" + layer_suffix, { Q_rope },
                std::initializer_list<size_t>{0, 2, 1, 3}
            );
            
            NodeCredit V_t   = make_stream<OperatorType::Transpose>(graph, "V - transpose" + layer_suffix, { V_cur_out });

            const auto [K_out, V_out] = (T_use_kv_cache) ? connect_KVCache(graph, layer_suffix, K_rope, V_t) : reshape_KV(graph, layer_suffix, K_rope, V_t);

            NodeCredit attn_out = attention_projection(graph, layer_suffix, Q_out, K_out, V_out);
            return attn_out;
        }

    protected:
        std::tuple<NodeCredit, NodeCredit, NodeCredit> input_linear_map(Graph &graph, const std::string &layer_suffix, 
                NodeCredit attn_norm) const {
            NodeCredit Q_cur    = make_stream<OperatorType::MatMul>(graph, "Qcur" + layer_suffix, { weight.weight_q, attn_norm });
            NodeCredit K_cur    = make_stream<OperatorType::MatMul>(graph, "Kcur" + layer_suffix, { weight.weight_k, attn_norm });
            NodeCredit V_cur    = make_stream<OperatorType::MatMul>(graph, "Vcur" + layer_suffix, { weight.weight_v, attn_norm });

            NodeCredit Q_cur_out = (weight.bias_q == Graph::INVALID_NODE_CREDIT) ? Q_cur :
                make_stream<OperatorType::Add>(graph, "Qcur - biased" + layer_suffix, { Q_cur, weight.bias_q });
            NodeCredit K_cur_out = (weight.bias_k == Graph::INVALID_NODE_CREDIT) ? K_cur :
                make_stream<OperatorType::Add>(graph, "Kcur - biased" + layer_suffix, { K_cur, weight.bias_k });
            NodeCredit V_cur_out = (weight.bias_v == Graph::INVALID_NODE_CREDIT) ? V_cur :
                make_stream<OperatorType::Add>(graph, "Vcur - biased" + layer_suffix, { V_cur, weight.bias_v });

            return { Q_cur_out, K_cur_out, V_cur_out };
        }

        std::pair<NodeCredit, NodeCredit> position_encode(Graph &graph, const std::string &layer_suffix, 
                NodeCredit Q_cur, NodeCredit K_cur) const {
            NodeCredit Q_reshaped_cur = make_stream<OperatorType::Reshape>(graph, "Qcur - reshaped" + layer_suffix, { Q_cur }, 
                std::initializer_list<size_t>{ num_embedding_head, num_head, num_token },
                NumberType::FP32
            );
            NodeCredit K_reshaped_cur = make_stream<OperatorType::Reshape>(graph, "Kcur - reshaped" + layer_suffix, { K_cur }, 
                std::initializer_list<size_t>{ num_embedding_head, num_head, num_token },
                NumberType::FP32
            );

            NodeCredit Q_rope = make_stream<OperatorType::Rope>(graph, "Qcur - rope" + layer_suffix, { Q_reshaped_cur, input_pos }, rope_context);
            NodeCredit K_rope = make_stream<OperatorType::Rope>(graph, "Kcur - rope" + layer_suffix, { K_reshaped_cur, input_pos }, rope_context);

            return { Q_rope, K_rope };
        }

        NodeCredit attention_projection(Graph &graph, const std::string &layer_suffix, 
                NodeCredit query, NodeCredit key, NodeCredit value) const {
			NodeCredit KQ_out         = make_stream<OperatorType::MatMul>(graph, "Attention Score" + layer_suffix, { key, query });
			NodeCredit KQ_softmax_out = make_stream<OperatorType::Softmax>(graph, "Attention Context" + layer_suffix, { KQ_out, KQ_mask },
				1.0F / std::sqrt(static_cast<float>(static_cast<int>(num_embedding_head)))  // Scale
			);
			
			NodeCredit KQV       = make_stream<OperatorType::MatMul>(graph, "K - Q - V" + layer_suffix, { value, KQ_softmax_out });
			NodeCredit KQV_merge = make_stream<OperatorType::Permute>(graph, "K - Q - V merged" + layer_suffix, { KQV }, 
				std::initializer_list<size_t>{0, 2, 1, 3}
			);
			NodeCredit KQV_merged_cont = make_stream<OperatorType::Contiguous>(graph, "K - Q - V contiguous" + layer_suffix, { KQV_merge },
				std::initializer_list<size_t>{num_embedding_head * num_head, num_token}, NumberType::FP32
			);
			NodeCredit KQV_w_out = make_stream<OperatorType::MatMul>(graph, "K - Q - V weight" + layer_suffix, { weight.weight_o, KQV_merged_cont });
			NodeCredit KQV_out              = (weight.bias_o == Graph::INVALID_NODE_CREDIT) ?  KQV_w_out :
				make_stream<OperatorType::Add>(graph, "Attention output" + layer_suffix, { KQV_w_out, weight.bias_o });

			return KQV_out;
        }

    protected: /* Connection about KV cache */

        /*!
         * @brief If we don't use KV cache. then it is necessary for the consequent MatMul operator to make Value tensor contiguous.
         * Reshape Key and Value tensor for grouped head attention.
         */
        std::pair<NodeCredit, NodeCredit> reshape_KV(Graph &graph, const std::string &layer_suffix, 
			NodeCredit key, NodeCredit value) const {
            NodeCredit K_out = make_stream<OperatorType::View>(graph, "K - out" + layer_suffix, { key }, 
                0, // offset
                std::initializer_list<size_t>{ num_embedding_head, num_token, num_head_kv }, // New dimensions
                std::initializer_list<size_t>{ get_type_size(NumberType::FP32),
                                                get_row_size(NumberType::FP32, num_embedding_k_gqa),
                                                get_row_size(NumberType::FP32, num_embedding_head) }, // New offsets
                NumberType::FP32
            );
            // If we don't use KV cache, the value is incontiguous, which may incur performance degradation consequentially.
            // Therefore, we reconstruct it.
            NodeCredit V_out = make_stream<OperatorType::Contiguous>(graph, "V - transpose - contiguous" + layer_suffix, { value },
                std::initializer_list<size_t>{ num_token, num_embedding_head, num_head_kv }, NumberType::FP32
            );	
            return {K_out, V_out}; 
        }

        /*! 
         * @brief If KV Cache is applied, it is necessary to concat Key and Value tensor with the past results.
         * @note In case of consequent incontinuous MatMul, we need to store transposed Value tensor.
         * @note The Key and Value tensors are stored as FP16 
         * @details KV Cache are organized separately:
         * - K cache is organized as 1D array: [////////// past key /////////|--- new key ---|     ];
         *      - The length of K cache should be larger than `len_context` * `num_embedding_k_gqa`. 
         *      - Each time we store `num_token` * `num_embedding_k_gqa` data continuously
         *
         * - V cache is organized as 2D array: 
         *       ⌈ ///////////| -----------|      ⌉
         *       | // past  //| -- new   --|     |
         *       | // value //| -- value --|     |
         *       ⌊ ///////////| -----------|     ⌋
         *      - The column number is `num_embedding_v_gqa` while the row number is `num_context`
         *      - Each time we store `num_token` columns of `num_embedding_v_gqa` elements
         *
         */
		std::pair<NodeCredit, NodeCredit> connect_KVCache(Graph &graph, const std::string &layer_suffix, 
			NodeCredit key, NodeCredit value) const {

			// TODO: Fix when implementing long context
			const size_t num_kv = num_past_token + num_token;

			// Update key cache
			SPY_ASSERT(k_cache != Graph::INVALID_NODE_CREDIT, "Expect the k cache not to be invalid node");

			const Tensor &k_cache_tensor = graph.get_node_content<DataNode>(k_cache)->tensor;
			const NumberType k_type = k_cache_tensor.get_number_type();

			const int64_t k_cache_update_offset = get_row_size(k_type, num_embedding_k_gqa) * num_past_token;
			const NodeCredit k_cache_view = make_stream<OperatorType::View>(graph, "K cache" + layer_suffix, {k_cache},
				k_cache_update_offset,
				std::initializer_list<size_t>{ num_token * num_embedding_k_gqa },
				std::initializer_list<size_t>{ get_type_size(NumberType::FP16) },
				NumberType::FP16
			);
			const NodeCredit k_cache_sync = make_stream<OperatorType::Copy>(graph, "K cache update", {key, k_cache_view});
			// Concat K cache and output
			const NodeCredit K_out = make_stream<OperatorType::View>(graph, "K - cache - view" + layer_suffix, { k_cache_sync },
               -k_cache_update_offset, // offset
               std::initializer_list<size_t>{ num_embedding_head, num_kv, num_head_kv }, // New dimensions
               std::initializer_list<size_t>{ get_type_size(NumberType::FP16),
                                              get_row_size(NumberType::FP16, num_embedding_k_gqa),
                                              get_row_size(NumberType::FP16, num_embedding_head) }, // New offsets
               NumberType::FP16
			);

			// Update value cache
			SPY_ASSERT(v_cache != Graph::INVALID_NODE_CREDIT, "Expect the v cache not to be invalid node");

			const Tensor &v_cache_tensor = graph.get_node_content<DataNode>(v_cache)->tensor;
			const NumberType v_type = v_cache_tensor.get_number_type();

			const int64_t v_cache_update_offset = get_type_size(v_type) * num_past_token;
			const NodeCredit v_cache_view = make_stream<OperatorType::View>(graph, "V cache" + layer_suffix, {v_cache},
				v_cache_update_offset,
				std::initializer_list<size_t>{ num_token, num_embedding_v_gqa },
				std::initializer_list<size_t>{ get_type_size(NumberType::FP16),
											num_context * get_type_size(NumberType::FP16) },
				NumberType::FP16
			);
			// A fake output for synchronization
			const NodeCredit v_cache_sync = make_stream<OperatorType::Copy>(graph, "V cache update", {value, v_cache_view});
			// Concat value cache and output
			const NodeCredit V_out = make_stream<OperatorType::View>(graph, "V - cache - view" + layer_suffix, { v_cache_sync },
	             -v_cache_update_offset, // offset
	             std::initializer_list<size_t>{ num_kv, num_embedding_head, num_head_kv }, // New dimensions
	             std::initializer_list<size_t>{ get_type_size(NumberType::FP16),
	                                            get_type_size(NumberType::FP16) * num_context,
	                                            get_type_size(NumberType::FP16) * num_context * num_embedding_head }, // New offset
	             NumberType::FP16
			);

			return {K_out, V_out};
		}

    };

} // namespace spy