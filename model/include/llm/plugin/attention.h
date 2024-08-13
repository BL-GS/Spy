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

    struct KVCacheData {
        int64_t  num_embedding_k_gqa;
        int64_t  num_embedding_v_gqa;
        int64_t  num_context;

        NumberType k_cache_type;
        NumberType v_cache_type;

        int64_t num_past_token;
        
        DataNode * k_cache = nullptr;
        DataNode * v_cache = nullptr; 
    };

	struct KVCache final: public GraphBuilder, KVCacheData {
    public:
        KVCache() = default;

        KVCache(const KVCacheData &data): KVCacheData(data) {}

        KVCache(KVCache &&other) = default;

	public:
		void connect_KVCache(Graph &graph, int layer_id);
	};

    struct MultiHeadAttentionBlockData {
        /* Hyper params */
        NormRMSParam norm_rms_param;
        RopeParam    rope_param_draft;

        int64_t  num_embedding_head;
        int64_t  num_embedding_k_gqa;
        int64_t  num_embedding_v_gqa;
        int64_t  num_head_kv;
        int64_t  num_head;

        int64_t  num_context;
        int64_t  num_token;
        int64_t  num_past_token;

        /* Weights */
        MultiHeadAttentionWeight weight;

        /* Buffer */
        DataNode *KQ_mask;
        DataNode *k_cache;
        DataNode *v_cache;

        NumberType k_cache_type;
        NumberType v_cache_type;

        /* Input */
        DataNode *input_embedding;
        DataNode *input_pos;
    };   

    struct MultiHeadAttentionBlock final: public GraphBuilder, MultiHeadAttentionBlockData {
        MultiHeadAttentionBlock() = default;

        MultiHeadAttentionBlock(const MultiHeadAttentionBlockData &data): MultiHeadAttentionBlockData(data) {}

        MultiHeadAttentionBlock(MultiHeadAttentionBlock &&other) = default;

        MultiHeadAttentionBlock &operator=(MultiHeadAttentionBlock &&other) = default;

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
	


} // namespace spy