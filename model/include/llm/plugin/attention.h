#pragma once

#include "operator/type.h"
#include "operator/operator.h"
#include "llm/plugin/graph_builder.h"

namespace spy {

    struct MultiHeadAttentionWeight {
        DataNode *attention_norm  = nullptr;

        DataNode *weight_q        = nullptr;
        DataNode *weight_k        = nullptr;
        DataNode *weight_v        = nullptr;
        DataNode *weight_o        = nullptr;

        DataNode *bias_q          = nullptr;
        DataNode *bias_k          = nullptr;
        DataNode *bias_v          = nullptr;
        DataNode *bias_o          = nullptr;
    };

    struct MultiHeadAttentionBlock: public GraphBuilder {
        /* Hyper params */
        const NormRMSParam norm_rms_param;
        const RopeParam    rope_param;

        const int64_t  num_embedding_head;
        const int64_t  num_embedding_k_gqa;
        const int64_t  num_embedding_v_gqa;
        const int64_t  num_head_kv;
        const int64_t  num_head;

        int64_t  num_context;
        int64_t  num_token;
        int64_t  num_past_token;

        /* Weights */
        MultiHeadAttentionWeight weight;

        /* Buffer */
        DataNode *KQ_mask;
        DataNode *k_cache;
        DataNode *v_cache;

        /* Input */
        DataNode * input_embedding;
        DataNode * input_pos;


        DataNode *connect_attention(Graph &graph, int layer_id = -1, int expert_id = -1, bool enable_kvcache = true);

    protected:
        std::tuple<DataNode *, DataNode *, DataNode *> input_linear_map(Graph &graph, const DataNodeProperty &default_prop, 
                DataNode *attn_norm) const;

        std::pair<DataNode *, DataNode *> position_encode(Graph &graph, const DataNodeProperty &default_prop, 
                DataNode *Q_cur, DataNode *K_cur);

        DataNode *attention_projection(Graph &graph, const DataNodeProperty &default_prop, 
                DataNode *query, DataNode *key, DataNode *value);

    protected: /* Connection about KV cache */

        /*!
         * @brief If we don't use KV cache. then it is necessary for the consequent MatMul operator to make Value tensor contiguous.
         * Reshape Key and Value tensor for grouped head attention.
         */
        std::pair<DataNode *, DataNode *> reshape_KV(Graph &graph, const DataNodeProperty &default_prop,
			    DataNode *key, DataNode *value);

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
		std::pair<DataNode *, DataNode *> connect_KVCache(Graph &graph, const DataNodeProperty &default_prop,
			    DataNode *key, DataNode *value);

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
				const int64_t k_num  = n_embd_k_gqa * kv_size;
				const Shape k_shape{{k_num}, NumberType::FP16};
				const size_t k_size = k_shape.total_size();
				uint8_t *k_data = new uint8_t[k_size];
				k_cache.emplace_back(k_shape, k_data);
				pointer_storage.emplace_back(k_data);
			}

			for (uint32_t i = v_cache.size(); i < num_layer; ++i) {
				const int64_t v_num  = n_embd_v_gqa * kv_size;
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