/*
 * @author: BL-GS 
 * @date:   2024/3/31
 */

#pragma once

#include <cstdint>
#include <magic_enum.hpp>

#include "llm/vocab/vocab.h"
#include "graph/graph.h"
#include "llm/model/config.h"
#include "adapter/type.h"

// Operator header is necessary here for template initialization
#include "operator/operator.h"

namespace spy {
    
    struct ModelMetadata {
        /* basic */
        std::string     model_name;
        int32_t         num_layer;
        uint32_t        len_context__train;

        /* output */
        uint32_t        num_embedding;

        /* attention */
        uint32_t        num_head;
        uint32_t        num_head_kv;
        uint32_t        num_embedding_head_k;
        uint32_t        num_embedding_head_v;
        uint32_t        num_embedding_k_gqa;
        uint32_t        num_embedding_v_gqa;

        uint32_t        num_rot;

        /* ffn */
        uint32_t        num_ffn;
        float           ffn_norm_rms_eps;

        /* vocab */
        ModelVocabType  vocab_type;
        uint32_t        num_vocab;
    };

    class Model {

    protected:
        HyperParam                      hyper_param_;

        ModelMetadata                   metadata_;
        
        std::unique_ptr<Tokenizer>      tokenizer_ptr;


    public:
        explicit Model(const HyperParam &hyper_param): hyper_param_(hyper_param) {}

        virtual ~Model() = default;

    public:
        virtual void init(ModelMetaContext &context) {
            init_metadata(context);
			init_hyper_param(context);
            init_tokenizer(context);
        }

    protected: /* Initialization */
        virtual void init_metadata(ModelMetaContext &context) {
            metadata_.model_name = context.arch_name;
            metadata_.len_context__train = context.find_gguf_value(LLMKey::CONTEXT_LENGTH).get_value<uint32_t>();
            metadata_.num_embedding     = context.find_gguf_value(LLMKey::EMBEDDING_LENGTH).get_value<uint32_t>();
            metadata_.num_ffn           = context.find_gguf_value(LLMKey::FEED_FORWARD_LENGTH).get_value<uint32_t>();
            metadata_.num_head          = context.find_gguf_value(LLMKey::ATTENTION_HEAD_COUNT).get_value<uint32_t>();
            metadata_.num_layer         = static_cast<int32_t>(context.find_gguf_value(LLMKey::BLOCK_COUNT).get_value<uint32_t>());

            {
                const auto num_vocab_option = context.find_gguf_value_option(LLMKey::VOCAB_SIZE);
                if (num_vocab_option.has_value()) {
                    metadata_.num_vocab = num_vocab_option->get_value<uint32_t>();
                } else {
                    metadata_.num_vocab = context.find_gguf_value(LLMKey::TOKENIZER_LIST).get_value<ModelMetaArray>().size();
                }
            }

            metadata_.num_head_kv = context.find_gguf_value_option<uint32_t>(LLMKey::ATTENTION_HEAD_COUNT_KV, 0);

            const uint32_t num_embedding_head_k_default = (metadata_.num_head == 0) ? 0 : metadata_.num_embedding / metadata_.num_head;
            metadata_.num_embedding_head_k = context.find_gguf_value_option(LLMKey::ATTENTION_KEY_LENGTH, num_embedding_head_k_default);
            const uint32_t num_embedding_head_v_default = (metadata_.num_head == 0) ? 0 : metadata_.num_embedding / metadata_.num_head;
            metadata_.num_embedding_head_v = context.find_gguf_value_option(LLMKey::ATTENTION_VALUE_LENGTH, num_embedding_head_v_default);

            metadata_.num_embedding_k_gqa = metadata_.num_embedding_head_k * metadata_.num_head_kv;
            metadata_.num_embedding_v_gqa = metadata_.num_embedding_head_v * metadata_.num_head_kv;

            metadata_.num_rot = context.find_gguf_value_option(LLMKey::ROPE_DIMENSION_COUNT,
                    (metadata_.num_head == 0) ? 0 : metadata_.num_embedding / metadata_.num_head);
        }

		virtual void init_hyper_param(ModelMetaContext &context) {
            if (hyper_param_.num_context == 0) {
                hyper_param_.num_context = metadata_.len_context__train;
            }

            hyper_param_.rope_type = RopeType::None;

			if (hyper_param_.rope_freq_base == 0.0F) {
				hyper_param_.rope_freq_base = context.find_gguf_value_option<float>(LLMKey::ROPE_FREQ_BASE, 10000.0F);
			}

			if (hyper_param_.rope_scaling_type == ModelRopeScalingType::Unspecified) {
				hyper_param_.rope_scaling_type = 
                    HyperParam::parse_rope_scaling_type(context.find_gguf_value_option<std::string>(LLMKey::ROPE_SCALING_TYPE, "linear"));
			}

			if (hyper_param_.rope_scaling_type == ModelRopeScalingType::None) {
				hyper_param_.rope_freq_scale = 1.0F;
			} else if (hyper_param_.rope_freq_scale == 0.0F) {
				const auto rope_scaling_factor_option = context.find_gguf_value_option(LLMKey::ROPE_SCALING_FACTOR);
                const auto rope_scaling_linear_option = context.find_gguf_value_option(LLMKey::ROPE_SCALE_LINEAR);
				if (rope_scaling_factor_option.has_value()) {
					hyper_param_.rope_freq_scale = 1.0F / rope_scaling_factor_option->get_value<float>();
				} else if (rope_scaling_linear_option.has_value()) {
                    hyper_param_.rope_freq_scale = 1.0F / rope_scaling_linear_option->get_value<float>();
                } else {
                    hyper_param_.rope_freq_scale  = 1.0F;
                }
			}

			if (hyper_param_.rope_pooling_type == ModelPoolingType::Unspecific) {
				hyper_param_.rope_pooling_type = ModelPoolingType::None;
			}

			if (hyper_param_.yarn_orig_ctx == 0) {
				hyper_param_.yarn_orig_ctx = context.find_gguf_value_option<uint32_t>(LLMKey::ROPE_SCALING_ORIG_CTX_LEN, metadata_.len_context__train);
			}

			if (hyper_param_.yarn_ext_factor < 0.0F) {
				hyper_param_.yarn_ext_factor = (hyper_param_.rope_scaling_type == ModelRopeScalingType::Yarn) ? 1.0F : 0.0F;
			}
		}

        virtual void init_tokenizer(ModelMetaContext &context) = 0;

    public: /* Graph building */
        virtual void build_graph(ModelMetaContext &context, Graph &graph, ModelIO &model_io) = 0;

        virtual void propagate(Graph &graph, ModelIO &model_io) = 0;

    public:
        const ModelMetadata &   get_info()      const { return metadata_;       }

        Tokenizer &             get_tokenizer() const { return *tokenizer_ptr;  }

    };

} // namespace spy