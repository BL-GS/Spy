#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <variant>
#include <map>
#include <optional>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "util/type/enum.h"
#include "util/type/printable.h"
#include "util/shell/logger.h"
#include "number/number.h"

namespace spy {

	enum class ModelMetaDataType : int {
		UInt8   = 0 , Int8    = 1,
		UInt16  = 2 , Int16   = 3,
		UInt32  = 4 , Int32   = 5,
		Float32 = 6 , Bool    = 7,
		String  = 8 , Array   = 9,
		UInt64  = 10, Int64   = 11,
		Float64 = 12,

		ModelMetaDataTypeEnd
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


	using ModelMetaArray   = std::vector<std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
								float, bool, std::string, uint64_t, int64_t, double>>;
	using ModelMetaElement = std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float, bool,
								std::string, ModelMetaArray, uint64_t, int64_t, double>;

	template<ModelMetaDataType T_type>
	using ModelMetaTypeMap = std::variant_alternative_t<static_cast<size_t>(T_type), ModelMetaElement>;

	struct ModelMetaValue {
	public:
		ModelMetaElement value;

	public:
		ModelMetaValue() = default;

		template<class T>
		ModelMetaValue(T &&new_val): value(std::forward<T>(new_val)) {}

	public:
		ModelMetaDataType get_type() const { return static_cast<ModelMetaDataType>(value.index()); }

		template<class T>
		T get_value() const { return std::get<T>(value); }
	};

	struct ModelMetaHeader: public PropertyInterface {
        char                magic[16];
		uint32_t            version;
		uint64_t            num_tensor;
		uint64_t            num_kv;

		constexpr ModelMetaHeader() : magic{0}, version(3), num_tensor(0), num_kv(0) {}

		std::map<std::string_view, std::string> property() const override {
			return {
				{ "magic", 		magic 	},
				{ "version", 	std::to_string(version) },
				{ "#tensor",	std::to_string(num_tensor) },
				{ "#kv",		std::to_string(num_kv) }
			};
		}
	};

	struct ModelMetaTensorInfo: public PropertyInterface {
		static constexpr uint32_t MAX_DIMS = 4;

		uint32_t     						num_dim;
		std::array<int64_t, MAX_DIMS>     	num_element;
		NumberType   						type;
		uint64_t     						offset;

		void * 								data_ptr;

		ModelMetaTensorInfo(): num_dim(0), num_element{0}, type(NumberType::FP32),
			offset(0), data_ptr(nullptr) {}

		std::map<std::string_view, std::string> property() const override {
			return {
				{ "dimension", 	std::to_string(num_dim) },
				{ "element", 	fmt::format("{}", num_element) },
				{ "type", 		std::string{ magic_enum::enum_name(type) } },
				{ "offset", 	std::to_string(offset) }
			};
		}
	};

	struct ModelMetaContext: public PropertyInterface {
	public:
		static constexpr size_t DEFAULT_ALIGNMENT = 32;

	public:
		ModelMetaHeader                              header;
		std::map<std::string, ModelMetaValue>        kv_pairs;
		std::map<std::string, ModelMetaTensorInfo>   infos;

		std::string arch_name;
		size_t 		alignment;
		size_t 		offset;
		size_t 		size;

	public:
		ModelMetaContext() : arch_name("unknown"), alignment(DEFAULT_ALIGNMENT), offset(0), size(0) {}
		
		ModelMetaContext(ModelMetaContext &&other) noexcept = default;

	public:
		template<class T>
		void add_gguf_value(const std::string &key, T &&value) {
			auto iter_pair = kv_pairs.insert({key, {std::forward<T>(value)}});
			spy_assert(iter_pair.second, "Cannot insert gguf value by key: {}", key);
		}

	public:
		ModelMetaValue find_gguf_value(const std::string &key) const {
			auto iter = kv_pairs.find(key);
			spy_assert(iter != kv_pairs.end(), "Cannot find gguf value by key: {}", key);
			return iter->second;
		}

		ModelMetaValue find_gguf_value(const LLMKey key) const {
			const std::string key_name = get_LLM_name(key, arch_name);
			return find_gguf_value(key_name);
		}

		template<class T>
		T find_gguf_value_option(const LLMKey key, const T &default_val) const {
			const std::string key_name = get_LLM_name(key, arch_name);
			const auto option = find_gguf_value_option(key_name);
			if (option.has_value()) { return option->get_value<T>(); }
			return default_val;
		}

		std::optional<ModelMetaValue> find_gguf_value_option(const std::string &key) const {
			auto iter = kv_pairs.find(key);
			if (iter == kv_pairs.end()) { return std::nullopt; }
			return iter->second;
		}

		std::optional<ModelMetaValue> find_gguf_value_option(const LLMKey key) const {
			const std::string key_name = get_LLM_name(key, arch_name);
			return find_gguf_value_option(key_name);
		}

	public:
		std::map<std::string_view, std::string> property() const override {
			auto prop = header.property();
			prop["arch"] 	  = arch_name;
			prop["alignment"] = std::to_string(alignment);
			return prop;
		}

		template<class T_Stream>
		void print_tensor(T_Stream &stream) const {
			for (auto &[name, info]: infos) {
				stream << name << ":\n";
				stream << info.to_string();
				stream << '\n';
			}
		}
	};

} // namespace spy

SPY_ENUM_FORMATTER(spy::ModelMetaDataType);
SPY_ENUM_FORMATTER(spy::LLMKey);

template <>
struct fmt::formatter<spy::ModelMetaArray>: fmt::formatter<std::string> {
	auto format(const spy::ModelMetaArray &array, fmt::format_context& ctx) const {
		constexpr size_t MAX_PRINT_SIZE = 4;

		const size_t size = array.size();
		std::string str = fmt::format("[Array({})]: ", size);

		str += '{';
		const size_t print_size = std::min(size, MAX_PRINT_SIZE);
		for (size_t i = 0; i < print_size; ++i) {
			if (i != 0) { str += ", "; }

			const auto &cur = array[i];
			switch (cur.index()) {
			case 0:  str += std::to_string(std::get<0>(cur)); break;
			case 1:  str += std::to_string(std::get<1>(cur)); break;
			case 2:  str += std::to_string(std::get<2>(cur)); break;
			case 3:  str += std::to_string(std::get<3>(cur)); break;
			case 4:  str += std::to_string(std::get<4>(cur)); break;
			case 5:  str += std::to_string(std::get<5>(cur)); break;
			case 6:  str += std::to_string(std::get<6>(cur)); break;
			case 7:  str += std::to_string(std::get<7>(cur)); break;
			case 8:  str += std::get<8>(cur); break;
			case 9:  str += std::to_string(std::get<9>(cur)); break;
			case 10: str += std::to_string(std::get<10>(cur)); break;
			case 11: str += std::to_string(std::get<11>(cur)); break;
			}
		}
		if (print_size != size) { str += "..."; }
		str += '}';
		return fmt::formatter<std::string>::format(str, ctx);
	}
};

template <>
struct fmt::formatter<spy::ModelMetaElement>: fmt::formatter<std::string> {
	auto format(const spy::ModelMetaElement &element, fmt::format_context& ctx) const {
		const auto type = static_cast<spy::ModelMetaDataType>(element.index());
		std::string str = fmt::format("[{}]: ", type);
		switch (element.index()) {
			case 0:  str += std::to_string(std::get<0>(element)); break;
			case 1:  str += std::to_string(std::get<1>(element)); break;
			case 2:  str += std::to_string(std::get<2>(element)); break;
			case 3:  str += std::to_string(std::get<3>(element)); break;
			case 4:  str += std::to_string(std::get<4>(element)); break;
			case 5:  str += std::to_string(std::get<5>(element)); break;
			case 6:  str += std::to_string(std::get<6>(element)); break;
			case 7:  str += std::to_string(std::get<7>(element)); break;
			case 8:  str += std::get<8>(element); break;
			case 9:  str =  fmt::format("{}", std::get<9>(element)); break;
			case 10: str += std::to_string(std::get<10>(element)); break;
			case 11: str += std::to_string(std::get<11>(element)); break;
			case 12: str += std::to_string(std::get<12>(element)); break;
		}
		return fmt::formatter<std::string>::format(str, ctx);
	}
};