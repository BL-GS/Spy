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

	struct LLAMALayer final: MultiHeadAttentionWeight, FFNWeight {};

	struct LLAMAWeight final: InputWeight, OutputWeight {
		/* Attention & FFN */
		std::vector<LLAMALayer> 	layers;
	};

	class LLAMAModel final : public Model, GraphBuilder {
	public:
		static constexpr std::string_view MODEL_ARCH = "llama";

		using Layer = LLAMALayer;

	private: /* Graph structure */
		/// The pre-train parameter of the model
		LLAMAWeight							 pre_train;
		/// The input and the output of model
		InputBlock							 input_block;
		OutputBlock							 output_block;
		/// Attention block
		std::vector<MultiHeadAttentionBlock> attention_block_array;
		std::vector<KVCache>				 kv_cache_array;
		/// Feed-forward block
		std::vector<FFNBlock>				 ffn_block_array;

	public: /* Interface */
		InputBlockResult    input;
		OutputBlockResult   output;
		std::vector<float>  kq_mask_;

	public:
		LLAMAModel(const HyperParam &hyper_param): Model(hyper_param) { }
		
		~LLAMAModel() noexcept override = default;

	protected: /* Initialization */
		void init_metadata(ModelMetaContext &context) override {
			Model::init_metadata(context);

			metadata_.ffn_norm_rms_eps  = context.find_gguf_value(LLMKey::ATTENTION_LAYERNORM_RMS_EPS).get_value<float>();

			spy_assert(metadata_.num_rot == metadata_.num_embedding / metadata_.num_head, 
					"Invalid num_rot: {}, expect: {}", metadata_.num_rot, metadata_.num_embedding / metadata_.num_head);
		}

		void init_hyper_param(ModelMetaContext &context) override {
			Model::init_hyper_param(context);

			hyper_param_.rope_type = RopeType::Norm;
		}

		void init_tokenizer(ModelMetaContext &context) override {
			tokenizer_ptr = std::make_unique<Tokenizer>(ModelVocabType::SentencePiece, context);
		}

	public: /* Graph building */
		void build_graph(ModelMetaContext &context, Graph &graph, ModelIO &model_io) override;

		void propagate(Graph &graph, ModelIO &model_io) override;

	private: /* Constant Graph component */
		void build_input(ModelMetaContext &context, Graph &graph);

		void build_output(ModelMetaContext &context, Graph &graph);

        void build_attention(ModelMetaContext &context, Graph &graph, int layer_id);

        void build_ffn(ModelMetaContext &context, Graph &graph, int layer_id);
	};

}  // namespace spy