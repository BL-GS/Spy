/*
 * @author: BL-GS 
 * @date:   2024/3/31
 */

#pragma once

#include <vector>

#include "graph/graph.h"
#include "llm/model/abstract_model.h"
#include "llm/plugin/graph_builder.h"
#include "llm/plugin/attention.h"
#include "llm/plugin/ffn.h"
#include "llm/plugin/inout.h"

namespace spy {

	struct LLAMALayer final: MultiHeadAttentionWeight, FFNWeight {
		/* KV Cache */
		DataNode *k_cache				= nullptr;
		DataNode *v_cache				= nullptr;
	};

	struct LLAMAWeight final: InputWeight, OutputWeight {
		/* Attention & FFN */
		std::vector<LLAMALayer> 	layers;
	};

	class LLAMAModel final : public AbstractModel, GraphBuilder {
	public:
		static constexpr std::string_view MODEL_ARCH = "llama";
		static constexpr bool	   USE_KV_CACHE = true;

		using Layer = LLAMALayer;

	private: /* Graph structure */
		/// The pre-train parameter of the model
		LLAMAWeight							 pre_train;
		/// The input and the output of model
		InputBlock							 input_block;
		OutputBlock							 output_block;
		/// Attention block
		std::vector<MultiHeadAttentionBlock> attention_block_array;
		/// Feed-forward block
		std::vector<FFNBlock>				 ffn_block_array;

		std::vector<float>               	 kq_mask_;

		KVCache         					 kv_cache_;

	public:
		LLAMAModel(ModelMetaContext &&context_ptr, const HyperParam &hyper_param):
				AbstractModel(std::forward<ModelMetaContext>(context_ptr), hyper_param) { }
		
		~LLAMAModel() noexcept override = default;

	protected: /* Initialization */
		void init_metadata() override {
			AbstractModel::init_metadata();

			metadata_.ffn_norm_rms_eps  = context_.find_gguf_value(LLMKey::ATTENTION_LAYERNORM_RMS_EPS).get_value<float>();

			spy_assert(metadata_.num_rot == metadata_.num_embedding / metadata_.num_head, 
					"Invalid num_rot: {}, expect: {}", metadata_.num_rot, metadata_.num_embedding / metadata_.num_head);
		}

		void init_hyper_param() override {
			AbstractModel::init_hyper_param();

			hyper_param_.rope_type = RopeType::Norm;
		}

		void init_tokenizer() override {
			tokenizer_ptr = std::make_unique<Tokenizer>(ModelVocabType::SentencePiece, context_);
		}

	public: /* Graph building */
		void build_graph(Graph &graph, ModelIO &model_io) override;

		void propagate(ModelIO &model_io) override;

	private: /* Constant Graph component */
		void build_input(Graph &graph);

		void build_output(Graph &graph);

        void build_attention(Graph &graph, int layer_id);

        void build_ffn(Graph &graph, int layer_id);

		void build_kv_cache(Graph &graph, int layer_id);
	};

}  // namespace spy