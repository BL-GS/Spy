/*
 * @author: BL-GS 
 * @date:   2024/3/31
 */

#pragma once

#include <vector>

#include "model/file/config.h"
#include "model/model_impl/abstract_model.h"

#include "model/plugin/graph_builder.h"
#include "model/plugin/attention.h"
#include "model/plugin/ffn.h"

namespace spy {

	struct LLAMALayer: MultiHeadAttentionWeight, FFNWeight {
		/* Normalization */
		NodeCredit layer_out_norm		= Graph::INVALID_NODE_CREDIT;
		/* KV Cache */
		NodeCredit k_cache				= Graph::INVALID_NODE_CREDIT;
		NodeCredit v_cache				= Graph::INVALID_NODE_CREDIT;
	};

	struct LLAMAGraph final: public Graph {
		/* Input */
		NodeCredit 					token_embedding	= Graph::INVALID_NODE_CREDIT;
		/* Output */
		NodeCredit 					output_norm		= Graph::INVALID_NODE_CREDIT;
		NodeCredit 					output			= Graph::INVALID_NODE_CREDIT;
		/* Attention & FFN */
		std::vector<LLAMALayer> 	layers;

		LLAMAGraph(const std::string_view graph_name, uint32_t num_layer): Graph(graph_name), layers(num_layer) {}
	};

	class LLAMAModel final : public AbstractModel, GraphBuilder {
	public:
		static constexpr ModelType MODEL_TYPE   = ModelType::LLaMa;
		static constexpr bool	   USE_KV_CACHE = true;

		using Layer = LLAMALayer;

	private: /* Graph structure */
		std::unique_ptr<LLAMAGraph> graph_ptr_;

		std::vector<float> kq_mask_;

		std::unique_ptr<KVCache> kv_cache_ptr_;

	public:
		LLAMAModel(std::unique_ptr<GGUFContext> &&context_ptr, const HyperParam &hyper_param):
				AbstractModel(std::move(context_ptr), hyper_param) { }
		
		~LLAMAModel() noexcept override = default;

	protected: /* Initialization */
		void init_metadata() override {
			const GGUFContext &context = *context_ptr_;
			AbstractModel::init_metadata();

			metadata_.model_type 		= ModelType::LLaMa;
			metadata_.ffn_norm_rms_eps  = context.find_gguf_value(LLMKey::ATTENTION_LAYERNORM_RMS_EPS).get_value<float>();

			spy_assert(metadata_.num_rot == metadata_.num_embedding / metadata_.num_head, 
					"Invalid num_rot: {}, expect: {}", metadata_.num_rot, metadata_.num_embedding / metadata_.num_head);
		}

		void init_hyper_param() override {
			AbstractModel::init_hyper_param();

			hyper_param_.rope_type = ModelRopeType::Norm;
		}

		void init_tokenizer() override {
			const GGUFContext &context = *context_ptr_;
			tokenizer_ptr = std::make_unique<Tokenizer>(ModelVocabType::SentencePiece, context);
		}

	public: /* Graph building */
		std::unique_ptr<Graph> build_graph(ModelIO &model_io) override {
			const uint32_t num_layer 		 = metadata_.num_layer;

			const size_t num_token 			 = model_io.num_token();
			const size_t num_vocab			 = metadata_.num_vocab;
			const size_t num_head		 	 = metadata_.num_head;
			const size_t num_head_kv		 = metadata_.num_head_kv;
			const size_t num_embedding_head  = metadata_.num_embedding_head_v;
			const size_t num_embedding_k_gqa = metadata_.num_embedding_k_gqa;
			const size_t num_embedding_v_gqa = metadata_.num_embedding_v_gqa;
			spy_assert(num_embedding_head == metadata_.num_embedding_head_k);
			spy_assert(num_embedding_head == metadata_.num_rot);
			const size_t num_context		 = (USE_KV_CACHE) ? hyper_param_.num_context: num_token;

			const RopeContext rope_context = {
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


			graph_ptr_ = std::make_unique<LLAMAGraph>("LLaMa", num_layer);
			LLAMAGraph &graph = *graph_ptr_;

			/*
				Build all constant tensors、input tensors and buffered tensors.
				NOTE: All buffered tensor should allocated at fixed location each time.
			 */
			
			// Set KV Cache
			if (USE_KV_CACHE) {
				if (kv_cache_ptr_ == nullptr) { kv_cache_ptr_ = std::make_unique<KVCache>(); }
				kv_cache_ptr_->reserve(num_embedding_k_gqa, num_embedding_v_gqa, 
					num_context, num_layer);
				for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
					build_kv_cache(graph, layer_id);
				}
			}

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

			// Set constant tensor
			build_input(graph);
			build_output(graph);
			for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
				build_attention(graph, layer_id);
				build_ffn(graph, layer_id);
			}

			/* Set input */
			NodeCredit input_embedding = LLAMAGraph::INVALID_NODE_CREDIT;
			if (!model_io.token_id_array.empty()) {
				NodeCredit input_token_id = create_input_tensor(graph,
					NumberType::INT32, 1, { num_token }, model_io.token_id_array.data(),
					TensorType::InputTokenId
				);
				input_embedding = make_stream<OperatorType::GetRow>(graph, 
					TensorType::InputTokenEmbedding, -1, -1,
					{ graph.token_embedding, input_token_id }
				);
			} else {
				input_embedding = create_input_tensor(graph,
					NumberType::FP32, 1, { num_token }, model_io.embedding.data(),
					TensorType::InputTokenEmbedding
				);
			}
			const NodeCredit input_pos = create_input_tensor(graph,
				NumberType::INT32, 1, { num_token }, model_io.positions.data(),
				TensorType::InputPosition
			);

			const NodeCredit KQ_mask = create_input_tensor(graph,
				NumberType::INT32, 2, { num_context, num_token }, kq_mask_.data(),
				TensorType::InputKQMask
			);

			
			/* Set output */
			model_io.logits.resize(num_token * num_vocab, 0.0F);
			const NodeCredit output = create_output_tensor(graph,
				NumberType::FP32, 2, {num_vocab, num_token}, model_io.logits.data(),
				TensorType::OutputLogits
			);

			/* Connection */
			{
				for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
					LLAMALayer &layer = graph.layers[layer_id];

					const NodeCredit attn_out = MultiHeadAttentionBlock {
						/* Hyper param */
						.ffn_norm_rms_eps    = metadata_.ffn_norm_rms_eps,
						.num_embedding_head  = num_embedding_head,
						.num_embedding_k_gqa = num_embedding_k_gqa,
						.num_embedding_v_gqa = num_embedding_v_gqa,
						.num_head_kv         = num_head_kv,
						.num_head            = num_head,
						.num_context         = num_context,
						.num_token           = num_token,
						.num_past_token      = kv_cache_ptr_->head,
						.rope_context        = rope_context,
						/* Weights */
						.weight = layer,
						/* Buffer */
						.KQ_mask = KQ_mask,
						.k_cache = layer.k_cache,
						.v_cache = layer.v_cache,
						/* Input */
						.input_embedding = input_embedding,
						.input_pos		 = input_pos,
					}.connect_attention<USE_KV_CACHE>(graph, layer_id);

					/* Feed-forward network */
					const NodeCredit ffn_inp  = make_stream<OperatorType::Add>(graph, 
						TensorType::V_FFNInput, layer_id, -1, 
						{ attn_out, input_embedding }
					);
					const NodeCredit ffn_out  = FFNBlock {
						.ffn_norm_rms_eps = metadata_.ffn_norm_rms_eps,
						.weight			  = layer,
						.ffn_input 		  = ffn_inp,
					}.connect_ffn(graph, layer_id);

					/* Output */
					const NodeCredit logit_out = make_stream<OperatorType::Add>(graph, 
						TensorType::V_FFNOutput, layer_id, -1,
						{ ffn_inp, ffn_out }
					);

					input_embedding = logit_out;
				}

				const NodeCredit result_norm = make_stream<OperatorType::NormRMS>(graph, 
					TensorType::V_ResultNorm, -1, -1, 
					{ input_embedding }, metadata_.ffn_norm_rms_eps
				);
				const NodeCredit result_norm_weighted = make_stream<OperatorType::Mul>(graph, 
					TensorType::V_ResultNormWeighted, -1, -1,
					{ result_norm, graph.output_norm }
				);
				
				// Final output
				make_determined_stream<OperatorType::MatMul>(graph, 
					{ graph.output, result_norm_weighted }, output
				);
			}

			if constexpr (USE_KV_CACHE) {
				kv_cache_ptr_->step(num_token);
			}			

			return std::move(graph_ptr_);
		}

	private: /* Graph component */
		void build_input(LLAMAGraph &graph) {
			const GGUFContext &context = *context_ptr_;
			graph.token_embedding = create_weight_tensor(context, graph, 
					2, { metadata_.num_embedding, metadata_.num_vocab }, 
					TensorType::TokenEmbedding);
		}

		void build_output(LLAMAGraph &graph) {
			const GGUFContext &context = *context_ptr_;

			graph.output_norm = create_weight_tensor(context, graph,
					1, { metadata_.num_embedding }, 
					TensorType::OutputNorm);
			graph.output = create_weight_tensor(context, graph,
					2, { metadata_.num_embedding, metadata_.num_vocab }, 
					TensorType::Output);
			if (graph.output == Graph::INVALID_NODE_CREDIT) {
				graph.output = create_weight_tensor(context, graph,
					2, { metadata_.num_embedding, metadata_.num_vocab }, 
					TensorType::TokenEmbedding);
			}
		}

        void build_attention(LLAMAGraph &graph, int layer_id) {
			const GGUFContext &context = *context_ptr_;
			auto  &layer               = graph.layers[layer_id];

			layer.attention_norm = create_weight_tensor(context, graph, 
					1, { metadata_.num_embedding }, 
					TensorType::AttentionNorm, layer_id);
			layer.weight_q = create_weight_tensor(context, graph, 
					2, { metadata_.num_embedding, metadata_.num_embedding }, 
					TensorType::AttentionQ, layer_id);
			layer.weight_k = create_weight_tensor(context, graph,
					2, { metadata_.num_embedding, metadata_.num_embedding_k_gqa }, 
					TensorType::AttentionK, layer_id);
			layer.weight_v = create_weight_tensor(context, graph, 
					2, { metadata_.num_embedding, metadata_.num_embedding_v_gqa }, 
					TensorType::AttentionV, layer_id);
			layer.weight_o = create_weight_tensor(context, graph, 
					2, { metadata_.num_embedding, metadata_.num_embedding }, 
					TensorType::AttentionOutput, layer_id);

			layer.bias_q = create_bias_tensor(context, graph, 
					1, { metadata_.num_embedding }, 
					TensorType::AttentionK, layer_id);
			layer.bias_k = create_bias_tensor(context, graph, 
					1, { metadata_.num_embedding_k_gqa }, 
					TensorType::AttentionK, layer_id);
			layer.bias_v = create_bias_tensor(context, graph, 
					1, { metadata_.num_embedding_v_gqa }, 
					TensorType::AttentionV, layer_id);
			layer.bias_o = create_bias_tensor(context, graph, 
					1, {metadata_.num_embedding }, 
					TensorType::AttentionOutput, layer_id);
        }

        void build_ffn(LLAMAGraph &graph, int layer_id) {
			const GGUFContext &context = *context_ptr_;
			auto &layer = graph.layers[layer_id];

			layer.ffn_norm = create_weight_tensor(context, graph, 
					1, { metadata_.num_embedding }, 
					TensorType::FFNNorm, layer_id);
			layer.ffn_up   = create_weight_tensor(context, graph, 
					2, { metadata_.num_embedding, metadata_.num_ffn }, 
					TensorType::FFNUp, layer_id);
			layer.ffn_gate = create_weight_tensor(context, graph, 
					2, { metadata_.num_embedding, metadata_.num_ffn }, 
					TensorType::FFNGate, layer_id);
			layer.ffn_down = create_weight_tensor(context, graph, 
					2, { metadata_.num_ffn, metadata_.num_embedding }, 
					TensorType::FFNDown, layer_id);
        }

		void build_kv_cache(LLAMAGraph &graph, int layer_id) {
			auto &layer = graph.layers[layer_id];
			const size_t num_embedding_k_gqa = metadata_.num_embedding_k_gqa;
			const size_t num_embedding_v_gqa = metadata_.num_embedding_v_gqa;
			const size_t num_context		 = hyper_param_.num_context;

			layer.k_cache = create_buffer_tensor(graph, 
				NumberType::FP16, 1, { num_embedding_k_gqa * num_context }, kv_cache_ptr_->k_cache[layer_id].get(),
				TensorType::KCache, layer_id);
			layer.v_cache = create_buffer_tensor(graph,
				NumberType::FP16, 1, { num_embedding_v_gqa * num_context }, kv_cache_ptr_->v_cache[layer_id].get(),
				TensorType::VCache, layer_id);
		}
	};

}  // namespace spy