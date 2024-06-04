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
        NodeCredit connect_attention(Graph &graph, int layer_id = -1, int expert_id = -1) const {

            NodeCredit attn_norm = make_stream<OperatorType::NormRMS>(graph,
                TensorType::V_AttentionNorm, layer_id, expert_id, 
                { input_embedding }, ffn_norm_rms_eps
            );
            NodeCredit attn_norm_weighted = make_stream<OperatorType::Mul>(graph, 
                TensorType::V_AttentionNormWeighted, layer_id, expert_id, 
                { attn_norm, weight.attention_norm }
            );

            /* Self-attention */
            const auto [Q_cur_out, K_cur_out, V_cur_out] = input_linear_map(graph, layer_id, expert_id, attn_norm_weighted);
            const auto [Q_rope, K_rope] = position_encode(graph, layer_id, expert_id, Q_cur_out, K_cur_out);

            NodeCredit Q_out = make_stream<OperatorType::Permute>(graph, 
                TensorType::V_QRope, layer_id, expert_id,
                { Q_rope },
                std::initializer_list<size_t>{0, 2, 1, 3}
            );
            
            NodeCredit V_t   = make_stream<OperatorType::Transpose>(graph, 
                TensorType::V_VWeightedBiased, layer_id, expert_id,
                { V_cur_out }
            );

            const auto [K_out, V_out] = (T_use_kv_cache) ? connect_KVCache(graph, layer_id, expert_id, K_rope, V_t) : reshape_KV(graph, layer_id, expert_id, K_rope, V_t);

            NodeCredit attn_out = attention_projection(graph, layer_id, expert_id, Q_out, K_out, V_out);
            return attn_out;
        }

    protected:
        std::tuple<NodeCredit, NodeCredit, NodeCredit> input_linear_map(Graph &graph, 
                int layer_id, int expert_id, 
                NodeCredit attn_norm) const {

            NodeCredit Q_cur    = make_stream<OperatorType::MatMul>(graph, 
                TensorType::V_QWeighted, layer_id, expert_id, 
                { weight.weight_q, attn_norm }
            );
            NodeCredit K_cur    = make_stream<OperatorType::MatMul>(graph,
                TensorType::V_KWeighted, layer_id, expert_id,
                { weight.weight_k, attn_norm }
            );
            NodeCredit V_cur    = make_stream<OperatorType::MatMul>(graph,
                TensorType::V_VWeighted, layer_id, expert_id, 
                { weight.weight_v, attn_norm }
            );

            NodeCredit Q_cur_out = (weight.bias_q == Graph::INVALID_NODE_CREDIT) ? Q_cur :
                make_stream<OperatorType::Add>(graph, 
                    TensorType::V_QWeightedBiased, layer_id, expert_id, 
                    { Q_cur, weight.bias_q }
                );
            NodeCredit K_cur_out = (weight.bias_k == Graph::INVALID_NODE_CREDIT) ? K_cur :
                make_stream<OperatorType::Add>(graph, 
                    TensorType::V_KWeightedBiased, layer_id, expert_id, 
                    { K_cur, weight.bias_k }
                );
            NodeCredit V_cur_out = (weight.bias_v == Graph::INVALID_NODE_CREDIT) ? V_cur :
                make_stream<OperatorType::Add>(graph, 
                    TensorType::V_VWeightedBiased, layer_id, expert_id, 
                    { V_cur, weight.bias_v }
                );

            return { Q_cur_out, K_cur_out, V_cur_out };
        }

        std::pair<NodeCredit, NodeCredit> position_encode(Graph &graph, 
                int layer_id, int expert_id, 
                NodeCredit Q_cur, NodeCredit K_cur) const {
            NodeCredit Q_reshaped_cur = make_stream<OperatorType::Reshape>(graph, 
                TensorType::V_QWeightedBiased, layer_id, expert_id, 
                { Q_cur }, 
                std::initializer_list<size_t>{ num_embedding_head, num_head, num_token },
                NumberType::FP32
            );
            NodeCredit K_reshaped_cur = make_stream<OperatorType::Reshape>(graph,
                TensorType::V_KWeightedBiased, layer_id, expert_id, 
                { K_cur }, 
                std::initializer_list<size_t>{ num_embedding_head, num_head, num_token },
                NumberType::FP32
            );

            NodeCredit Q_rope = make_stream<OperatorType::Rope>(graph, 
                TensorType::V_QRope, layer_id, expert_id, 
                { Q_reshaped_cur, input_pos }, rope_context
            );
            NodeCredit K_rope = make_stream<OperatorType::Rope>(graph,
                TensorType::V_KRope, layer_id, expert_id, 
                { K_reshaped_cur, input_pos }, rope_context
            );

            return { Q_rope, K_rope };
        }

        NodeCredit attention_projection(Graph &graph, 
                int layer_id, int expert_id, 
                NodeCredit query, NodeCredit key, NodeCredit value) const {
			NodeCredit KQ_out         = make_stream<OperatorType::MatMul>(graph, 
                TensorType::V_AttentionScore, layer_id, expert_id, 
                { key, query }
            );
			NodeCredit KQ_softmax_out = make_stream<OperatorType::Softmax>(graph, 
                TensorType::V_AttentionContext, layer_id, expert_id, 
                { KQ_out, KQ_mask },
				1.0F / std::sqrt(static_cast<float>(static_cast<int>(num_embedding_head)))  // Scale
			);
			
			NodeCredit KQV       = make_stream<OperatorType::MatMul>(graph, 
                TensorType::V_KQV, layer_id, expert_id, 
                { value, KQ_softmax_out }
            );
			NodeCredit KQV_merge = make_stream<OperatorType::Permute>(graph, 
                TensorType::V_KQV, layer_id, expert_id, 
                { KQV }, 
				std::initializer_list<size_t>{0, 2, 1, 3}
			);
			NodeCredit KQV_merged_cont = make_stream<OperatorType::Contiguous>(graph, 
                TensorType::V_KQV, layer_id, expert_id,
                { KQV_merge },
				std::initializer_list<size_t>{num_embedding_head * num_head, num_token}, NumberType::FP32
			);
			NodeCredit KQV_w_out = make_stream<OperatorType::MatMul>(graph, 
                TensorType::V_KQVWeighted, layer_id, expert_id, 
                { weight.weight_o, KQV_merged_cont }
            );
			NodeCredit KQV_out              = (weight.bias_o == Graph::INVALID_NODE_CREDIT) ?  KQV_w_out :
				make_stream<OperatorType::Add>(graph,
                TensorType::V_AttentionOutput, layer_id, expert_id,
                { KQV_w_out, weight.bias_o }
            );

			return KQV_out;
        }

    protected: /* Connection about KV cache */

        /*!
         * @brief If we don't use KV cache. then it is necessary for the consequent MatMul operator to make Value tensor contiguous.
         * Reshape Key and Value tensor for grouped head attention.
         */
        std::pair<NodeCredit, NodeCredit> reshape_KV(Graph &graph,
            int layer_id, int expert_id, 
			NodeCredit key, NodeCredit value) const {
            NodeCredit K_out = make_stream<OperatorType::View>(graph, 
                TensorType::KCache, layer_id, expert_id, 
                { key }, 
                0, // offset
                std::initializer_list<size_t>{ num_embedding_head, num_token, num_head_kv }, // New dimensions
                std::initializer_list<size_t>{ get_type_size(NumberType::FP32),
                                                get_row_size(NumberType::FP32, num_embedding_k_gqa),
                                                get_row_size(NumberType::FP32, num_embedding_head) }, // New offsets
                NumberType::FP32
            );
            // If we don't use KV cache, the value is incontiguous, which may incur performance degradation consequentially.
            // Therefore, we reconstruct it.
            NodeCredit V_out = make_stream<OperatorType::Contiguous>(graph, 
                TensorType::VCache, layer_id, expert_id, 
                { value },
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
		std::pair<NodeCredit, NodeCredit> connect_KVCache(Graph &graph, 
            int layer_id, int expert_id, 
			NodeCredit key, NodeCredit value) const {

			// TODO: Fix when implementing long context
			const size_t num_kv = num_past_token + num_token;

			// Update key cache
			spy_assert(k_cache != Graph::INVALID_NODE_CREDIT, "Expect the k cache not to be invalid node");

			const Tensor &k_cache_tensor = graph.get_node_content<DataNode>(k_cache)->tensor;
			const NumberType k_type = k_cache_tensor.get_number_type();

			const int64_t k_cache_update_offset = get_row_size(k_type, num_embedding_k_gqa) * num_past_token;
			const NodeCredit k_cache_view = make_stream<OperatorType::View>(graph, 
                TensorType::KCache, layer_id, expert_id, 
                {k_cache},
				k_cache_update_offset,
				std::initializer_list<size_t>{ num_token * num_embedding_k_gqa },
				std::initializer_list<size_t>{ get_type_size(NumberType::FP16) },
				NumberType::FP16
			);
			const NodeCredit k_cache_sync = make_stream<OperatorType::Copy>(graph, 
                TensorType::KCache, layer_id, expert_id, 
                {key, k_cache_view}
            );
			// Concat K cache and output
			const NodeCredit K_out = make_stream<OperatorType::View>(graph, 
                TensorType::KCache, layer_id, expert_id, 
                { k_cache_sync },
               -k_cache_update_offset, // offset
               std::initializer_list<size_t>{ num_embedding_head, num_kv, num_head_kv }, // New dimensions
               std::initializer_list<size_t>{ get_type_size(NumberType::FP16),
                                              get_row_size(NumberType::FP16, num_embedding_k_gqa),
                                              get_row_size(NumberType::FP16, num_embedding_head) }, // New offsets
               NumberType::FP16
			);

			// Update value cache
			spy_assert(v_cache != Graph::INVALID_NODE_CREDIT, "Expect the v cache not to be invalid node");

			const Tensor &v_cache_tensor = graph.get_node_content<DataNode>(v_cache)->tensor;
			const NumberType v_type = v_cache_tensor.get_number_type();

			const int64_t v_cache_update_offset = get_type_size(v_type) * num_past_token;
			const NodeCredit v_cache_view = make_stream<OperatorType::View>(graph, 
                TensorType::VCache, layer_id, expert_id, 
                {v_cache},
				v_cache_update_offset,
				std::initializer_list<size_t>{ num_token, num_embedding_v_gqa },
				std::initializer_list<size_t>{ get_type_size(NumberType::FP16),
											num_context * get_type_size(NumberType::FP16) },
				NumberType::FP16
			);
			// A fake output for synchronization
			const NodeCredit v_cache_sync = make_stream<OperatorType::Copy>(graph, 
                TensorType::VCache, layer_id, expert_id, 
                {value, v_cache_view}
            );
			// Concat value cache and output
			const NodeCredit V_out = make_stream<OperatorType::View>(graph,
                TensorType::VCache, layer_id, expert_id, 
                { v_cache_sync },
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
	
	struct KVCache {
	public:
		size_t head = 0;

		std::vector<Tensor> k_cache;
		std::vector<Tensor> v_cache;

		std::vector<std::unique_ptr<uint8_t []>> pointer_storage;

	public:
		KVCache() = default;

		~KVCache() noexcept {
			spy_info("Delete kv cache");
		}

	public:
		void reserve(uint32_t n_embd_k_gqa, uint32_t n_embd_v_gqa, uint32_t kv_size, uint32_t num_layer) {
			k_cache.reserve(num_layer);
			v_cache.reserve(num_layer);
			pointer_storage.reserve(2 * num_layer);

			for (uint32_t i = k_cache.size(); i < num_layer; ++i) {
				const size_t k_num  = n_embd_k_gqa * kv_size;
				const Shape k_shape{{k_num}, NumberType::FP16};
				const size_t k_size = k_shape.total_size();
				uint8_t *k_data = new uint8_t[k_size];
				k_cache.emplace_back(k_shape, k_data);
				pointer_storage.emplace_back(k_data);
			}

			for (uint32_t i = v_cache.size(); i < num_layer; ++i) {
				const size_t v_num  = n_embd_v_gqa * kv_size;
				const Shape v_shape{{v_num}, NumberType::FP16};
				const size_t v_size = v_shape.total_size();
				uint8_t *v_data = new uint8_t[v_size];
				v_cache.emplace_back(v_shape, v_data);
				pointer_storage.emplace_back(v_data);
			}
		}

		void step(size_t n) { head += n; }
	};

} // namespace spy