#pragma once

#include <string>
#include <string_view>

#include "util/type/enum.h"

namespace spy {

	enum class GGUFDataType : int {
		UInt8   = 0 , Int8    = 1,
		UInt16  = 2 , Int16   = 3,
		UInt32  = 4 , Int32   = 5,
		Float32 = 6 , Bool    = 7,
		String  = 8 , Array   = 9,
		UInt64  = 10, Int64   = 11,
		Float64 = 12,

		GGUFDataTypeEnd
	};

	enum class LLMKey: int {
		GENERAL_ARCHITECTURE = 0,
		GENERAL_QUANTIZATION_VERSION,
		GENERAL_ALIGNMENT,
		GENERAL_NAME,
		GENERAL_AUTHOR,
		GENERAL_URL,
		GENERAL_DESCRIPTION,
		GENERAL_LICENSE,
		GENERAL_SOURCE_URL,
		GENERAL_SOURCE_HF_REPO,

		VOCAB_SIZE,
		CONTEXT_LENGTH,
		EMBEDDING_LENGTH,
		BLOCK_COUNT,
		FEED_FORWARD_LENGTH,
		USE_PARALLEL_RESIDUAL,
		TENSOR_DATA_LAYOUT,
		EXPERT_COUNT,
		EXPERT_USED_COUNT,
		POOLING_TYPE,
		LOGIT_SCALE,

		ATTENTION_HEAD_COUNT,
		ATTENTION_HEAD_COUNT_KV,
		ATTENTION_MAX_ALIBI_BIAS,
		ATTENTION_CLAMP_KQV,
		ATTENTION_KEY_LENGTH,
		ATTENTION_VALUE_LENGTH,
		ATTENTION_LAYERNORM_EPS,
		ATTENTION_LAYERNORM_RMS_EPS,
		ATTENTION_CAUSAL,

		ROPE_DIMENSION_COUNT,
		ROPE_FREQ_BASE,
		ROPE_SCALE_LINEAR,
		ROPE_SCALING_TYPE,
		ROPE_SCALING_FACTOR,
		ROPE_SCALING_ORIG_CTX_LEN,
		ROPE_SCALING_FINETUNED,

		SSM_INNER_SIZE,
		SSM_CONV_KERNEL,
		SSM_STATE_SIZE,
		SSM_TIME_STEP_RANK,

		TOKENIZER_MODEL,
		TOKENIZER_LIST,
		TOKENIZER_TOKEN_TYPE,
		TOKENIZER_TOKEN_TYPE_COUNT,
		TOKENIZER_SCORES,
		TOKENIZER_MERGES,
		TOKENIZER_BOS_ID,
		TOKENIZER_EOS_ID,
		TOKENIZER_UNK_ID,
		TOKENIZER_SEP_ID,
		TOKENIZER_PAD_ID,
		TOKENIZER_ADD_BOS,
		TOKENIZER_ADD_EOS,
		TOKENIZER_ADD_PREFIX,
		TOKENIZER_HF_JSON,
		TOKENIZER_RWKV,
	};

	inline constexpr std::string_view get_llm_kv_name(LLMKey key) {
		switch (key) {
			case LLMKey::GENERAL_ARCHITECTURE:          return "general.architecture";
			case LLMKey::GENERAL_QUANTIZATION_VERSION:  return "general.quantization_version";
			case LLMKey::GENERAL_ALIGNMENT:             return "general.alignment";
			case LLMKey::GENERAL_NAME:                  return "general.name";
			case LLMKey::GENERAL_AUTHOR:                return "general.author";
			case LLMKey::GENERAL_URL:                   return "general.url";
			case LLMKey::GENERAL_DESCRIPTION:           return "general.description";
			case LLMKey::GENERAL_LICENSE:               return "general.license";
			case LLMKey::GENERAL_SOURCE_URL:            return "general.source.url";
			case LLMKey::GENERAL_SOURCE_HF_REPO:        return "general.source.huggingface.repository";

			case LLMKey::VOCAB_SIZE:                    return "%s.vocab_size";
			case LLMKey::CONTEXT_LENGTH:                return "%s.context_length";
			case LLMKey::EMBEDDING_LENGTH:              return "%s.embedding_length";
			case LLMKey::BLOCK_COUNT:                   return "%s.block_count";
			case LLMKey::FEED_FORWARD_LENGTH:           return "%s.feed_forward_length";
			case LLMKey::USE_PARALLEL_RESIDUAL:         return "%s.use_parallel_residual";
			case LLMKey::TENSOR_DATA_LAYOUT:            return "%s.tensor_data_layout";
			case LLMKey::EXPERT_COUNT:                  return "%s.expert_count";
			case LLMKey::EXPERT_USED_COUNT:             return "%s.expert_used_count";
			case LLMKey::POOLING_TYPE :                 return "%s.pooling_type";
			case LLMKey::LOGIT_SCALE:                   return "%s.logit_scale";

			case LLMKey::ATTENTION_HEAD_COUNT:          return "%s.attention.head_count";
			case LLMKey::ATTENTION_HEAD_COUNT_KV:       return "%s.attention.head_count_kv";
			case LLMKey::ATTENTION_MAX_ALIBI_BIAS:      return "%s.attention.max_alibi_bias";
			case LLMKey::ATTENTION_CLAMP_KQV:           return "%s.attention.clamp_kqv";
			case LLMKey::ATTENTION_KEY_LENGTH:          return "%s.attention.key_length";
			case LLMKey::ATTENTION_VALUE_LENGTH:        return "%s.attention.value_length";
			case LLMKey::ATTENTION_LAYERNORM_EPS:       return "%s.attention.layer_norm_epsilon";
			case LLMKey::ATTENTION_LAYERNORM_RMS_EPS:   return "%s.attention.layer_norm_rms_epsilon";
			case LLMKey::ATTENTION_CAUSAL:              return "%s.attention.causal";

			case LLMKey::ROPE_DIMENSION_COUNT:          return "%s.rope.dimension_count";
			case LLMKey::ROPE_FREQ_BASE:                return "%s.rope.freq_base";
			case LLMKey::ROPE_SCALE_LINEAR:             return "%s.rope.scale_linear";
			case LLMKey::ROPE_SCALING_TYPE:             return "%s.rope.scaling.type";
			case LLMKey::ROPE_SCALING_FACTOR:           return "%s.rope.scaling.factor";
			case LLMKey::ROPE_SCALING_ORIG_CTX_LEN:     return "%s.rope.scaling.original_context_length";
			case LLMKey::ROPE_SCALING_FINETUNED:        return "%s.rope.scaling.finetuned";

			case LLMKey::SSM_CONV_KERNEL:               return "%s.ssm.conv_kernel";
			case LLMKey::SSM_INNER_SIZE:                return "%s.ssm.inner_size";
			case LLMKey::SSM_STATE_SIZE:                return "%s.ssm.state_size";
			case LLMKey::SSM_TIME_STEP_RANK:            return "%s.ssm.time_step_rank";

			case LLMKey::TOKENIZER_MODEL:               return "tokenizer.ggml.model";
			case LLMKey::TOKENIZER_LIST:                return "tokenizer.ggml.tokens";
			case LLMKey::TOKENIZER_TOKEN_TYPE:          return "tokenizer.ggml.token_type";
			case LLMKey::TOKENIZER_TOKEN_TYPE_COUNT:    return "tokenizer.ggml.token_type_count";
			case LLMKey::TOKENIZER_SCORES:              return "tokenizer.ggml.scores";
			case LLMKey::TOKENIZER_MERGES:              return "tokenizer.ggml.merges";
			case LLMKey::TOKENIZER_BOS_ID:              return "tokenizer.ggml.bos_token_id";
			case LLMKey::TOKENIZER_EOS_ID:              return "tokenizer.ggml.eos_token_id";
			case LLMKey::TOKENIZER_UNK_ID:              return "tokenizer.ggml.unknown_token_id";
			case LLMKey::TOKENIZER_SEP_ID:              return "tokenizer.ggml.seperator_token_id";
			case LLMKey::TOKENIZER_PAD_ID:              return "tokenizer.ggml.padding_token_id";
			case LLMKey::TOKENIZER_ADD_BOS:             return "tokenizer.ggml.add_bos_token";
			case LLMKey::TOKENIZER_ADD_EOS:             return "tokenizer.ggml.add_eos_token";
			case LLMKey::TOKENIZER_ADD_PREFIX:          return "tokenizer.ggml.add_space_prefix";
			case LLMKey::TOKENIZER_HF_JSON:             return "tokenizer.huggingface.json";
			case LLMKey::TOKENIZER_RWKV:                return "tokenizer.rwkv.world";
		}
		return "unknown";
	}

	inline std::string get_LLM_name(LLMKey key, std::string_view arch_name) {
		std::string_view llm_value_name = get_llm_kv_name(key);
		std::string res { llm_value_name };
		auto iter = res.find("%s");
		if (iter != std::string::npos) { res.replace(iter, iter + 2, arch_name); }
		return res;
	}
	
}  // namespace spy

SPY_ENUM_FORMATTER(spy::GGUFDataType);
SPY_ENUM_FORMATTER(spy::LLMKey);