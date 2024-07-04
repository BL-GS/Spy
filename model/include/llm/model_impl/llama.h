/*
 * @author: BL-GS 
 * @date:   2024/3/31
 */

#pragma once

#include <vector>
#include <string_view>

#include "graph/graph.h"
#include "metadata.h"
#include "llm/model_impl/abstract_model.h"
#include "llm/plugin/graph_builder.h"
#include "llm/plugin/attention.h"
#include "llm/plugin/ffn.h"

namespace spy {

	struct LLAMALayer: MultiHeadAttentionWeight, FFNWeight {
		/* KV Cache */
		DataNode *k_cache				= nullptr;
		DataNode *v_cache				= nullptr;
	};

	struct LLAMAGraph final: public Graph {
		/* Input */
		DataNode *					token_embedding	= nullptr;
		/* Output */
		DataNode *					output_norm		= nullptr;
		DataNode *					output			= nullptr;
		/* Attention & FFN */
		std::vector<LLAMALayer> 	layers;

		LLAMAGraph(const std::string_view graph_name, uint32_t num_layer): Graph(graph_name), layers(num_layer) {}
		~LLAMAGraph() noexcept override = default;
	};

	class LLAMAModel final : public AbstractModel, GraphBuilder {
	public:
		static constexpr ModelType MODEL_TYPE   = ModelType::LLaMa;
		static constexpr bool	   USE_KV_CACHE = true;

		using Layer = LLAMALayer;

	private: /* Graph structure */
		std::unique_ptr<LLAMAGraph>      constant_graph_ptr_;

		std::unique_ptr<Graph>           variable_graph_ptr_;

		std::vector<float>               kq_mask_;

		std::unique_ptr<KVCache>         kv_cache_ptr_;

	public:
		LLAMAModel(std::unique_ptr<ModelMetaContext> &&context_ptr, const HyperParam &hyper_param):
				AbstractModel(std::move(context_ptr), hyper_param) { }
		
		~LLAMAModel() noexcept override = default;

	protected: /* Initialization */
		void init_metadata() override {
			const ModelMetaContext &context = *context_ptr_;
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
			const ModelMetaContext &context = *context_ptr_;
			tokenizer_ptr = std::make_unique<Tokenizer>(ModelVocabType::SentencePiece, context);
		}

	public: /* Graph building */
		GraphView build_graph(ModelIO &model_io) override {
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

			/*
				Build all constant tensors、input tensors and buffered tensors.
				NOTE: All buffered tensor should be allocated at fixed location each time.
			 */

			if (constant_graph_ptr_ == nullptr) [[unlikely]] {
				constant_graph_ptr_ = std::make_unique<LLAMAGraph>("LLaMa", num_layer);
				LLAMAGraph &constant_graph = *constant_graph_ptr_;

				// Set KV Cache
				if constexpr (USE_KV_CACHE) {
					if (kv_cache_ptr_ == nullptr) { kv_cache_ptr_ = std::make_unique<KVCache>(); }
					kv_cache_ptr_->reserve(num_embedding_k_gqa, num_embedding_v_gqa,
					                       num_context, num_layer);
					for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
						build_kv_cache(constant_graph, layer_id);
					}
				}

				// Set constant tensor
				build_input(constant_graph);
				build_output(constant_graph);
				for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
					build_attention(constant_graph, layer_id);
					build_ffn(constant_graph, layer_id);
				}
			} else {
				constant_graph_ptr_->clear_data_connection();
			}
			LLAMAGraph &constant_graph = *constant_graph_ptr_;

			if (variable_graph_ptr_ == nullptr) [[unlikely]] {
				variable_graph_ptr_ = std::make_unique<Graph>("LLaMa");
			} else {
				variable_graph_ptr_->clear_data_node();
				variable_graph_ptr_->clear_op_node();
			}
			Graph &variable_graph = *variable_graph_ptr_;

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

			/* Set input */
			DataNode *input_embedding = nullptr;
			if (!model_io.token_id_array.empty()) {
				DataNode *input_token_id = create_input_tensor(variable_graph,
					NumberType::INT32, 1, { num_token }, model_io.token_id_array.data(),
					TensorType::InputTokenId
				);
				input_embedding = make_stream<OperatorType::GetRow>(variable_graph,
					TensorType::InputTokenEmbedding, -1, -1,
					{ constant_graph.token_embedding, input_token_id }
				);
			} else {
				input_embedding = create_input_tensor(variable_graph,
					NumberType::FP32, 1, { num_token }, model_io.embedding.data(),
					TensorType::InputTokenEmbedding
				);
			}
			DataNode *input_pos = create_input_tensor(variable_graph,
				NumberType::INT32, 1, { num_token }, model_io.positions.data(),
				TensorType::InputPosition
			);

			DataNode *KQ_mask = create_input_tensor(variable_graph,
				NumberType::INT32, 2, { num_context, num_token }, kq_mask_.data(),
				TensorType::InputKQMask
			);

			
			/* Set output */
			model_io.logits.resize(num_token * num_vocab, 0.0F);
			DataNode *output = create_output_tensor(variable_graph,
				NumberType::FP32, 2, {num_vocab, num_token}, model_io.logits.data(),
				TensorType::OutputLogits
			);

			/* Connection */
			{
				for (int layer_id = 0; layer_id < num_layer; ++layer_id) {
					LLAMALayer &layer = constant_graph.layers[layer_id];

					DataNode *attn_out = MultiHeadAttentionBlock {
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
					}.connect_attention<USE_KV_CACHE>(variable_graph, layer_id);

					/* Feed-forward network */
					DataNode *ffn_inp  = make_stream<OperatorType::Add>(variable_graph,
						TensorType::V_FFNInput, layer_id, -1, 
						{ attn_out, input_embedding }
					);
					DataNode *ffn_out  = FFNBlock {
						.ffn_norm_rms_eps = metadata_.ffn_norm_rms_eps,
						.weight			  = layer,
						.ffn_input 		  = ffn_inp,
					}.connect_ffn(variable_graph, layer_id);

					/* Output */
					DataNode *logit_out = make_stream<OperatorType::Add>(variable_graph,
						TensorType::V_FFNOutput, layer_id, -1,
						{ ffn_inp, ffn_out }
					);

					input_embedding = logit_out;
				}

				DataNode *result_norm = make_stream<OperatorType::NormRMS>(variable_graph,
					TensorType::V_ResultNorm, -1, -1, 
					{ input_embedding }, metadata_.ffn_norm_rms_eps
				);
				DataNode *result_norm_weighted = make_stream<OperatorType::Mul>(variable_graph,
					TensorType::V_ResultNormWeighted, -1, -1,
					{ result_norm, constant_graph.output_norm }
				);
				
				// Final output
				make_determined_stream<OperatorType::MatMul>(variable_graph,
					{ constant_graph.output, result_norm_weighted }, output
				);
			}

			if constexpr (USE_KV_CACHE) {
				kv_cache_ptr_->step(num_token);
			}			

			return { constant_graph_ptr_.get(), variable_graph_ptr_.get() };
		}

	private: /* Constant Graph component */
		void build_input(LLAMAGraph &graph) {
			const ModelMetaContext &context = *context_ptr_;
			graph.token_embedding = create_weight_tensor(context, graph, 
					2, { metadata_.num_embedding, metadata_.num_vocab }, 
					TensorType::TokenEmbedding);
		}

		void build_output(LLAMAGraph &graph) {
			const ModelMetaContext &context = *context_ptr_;

			graph.output_norm = create_weight_tensor(context, graph,
					1, { metadata_.num_embedding }, 
					TensorType::OutputNorm);
			graph.output = create_weight_tensor(context, graph,
					2, { metadata_.num_embedding, metadata_.num_vocab }, 
					TensorType::Output);
			if (graph.output == nullptr) {
				graph.output = create_weight_tensor(context, graph,
					2, { metadata_.num_embedding, metadata_.num_vocab }, 
					TensorType::TokenEmbedding);
			}
		}

        void build_attention(LLAMAGraph &graph, int layer_id) {
			const ModelMetaContext &context = *context_ptr_;
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
			const ModelMetaContext &context = *context_ptr_;
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