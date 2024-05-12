#pragma once

#include <string_view>
#include <array>
#include <fmt/format.h>
#include <magic_enum.hpp>

#include "util/logger.h"
#include "model/type.h"

namespace spy {
    
	struct ModelArchitectureTable {
		static constexpr size_t NUM_TENSOR_TYPE = static_cast<size_t>(ModelTensorType::ModelTensorTypeEnd);

		std::string_view                                 architecture_name;
		std::array<std::string_view, NUM_TENSOR_TYPE>    tensor_fmt_array;
	};

	inline constexpr ModelArchitectureTable get_tensor_array(ModelType model_type) {
		switch (model_type) {
			case ModelType::LLaMa:
				return {
					.architecture_name = "llama",
					.tensor_fmt_array  = []() constexpr{
						std::array<std::string_view, ModelArchitectureTable::NUM_TENSOR_TYPE> res;
						res.fill("");
						res[static_cast<size_t>(ModelTensorType::TokenEmbedding)]             = "token_embd";
						res[static_cast<size_t>(ModelTensorType::OutputNorm)]                = "output_norm";
						res[static_cast<size_t>(ModelTensorType::Output)]                    = "output";
						res[static_cast<size_t>(ModelTensorType::RopeFrequency)]             = "rope_freqs";
						res[static_cast<size_t>(ModelTensorType::AttentionNorm)]             = "blk.{}.attn_norm";
						res[static_cast<size_t>(ModelTensorType::AttentionQ)]                = "blk.{}.attn_q";
						res[static_cast<size_t>(ModelTensorType::AttentionK)]                = "blk.{}.attn_k";
						res[static_cast<size_t>(ModelTensorType::AttentionV)]                = "blk.{}.attn_v";
						res[static_cast<size_t>(ModelTensorType::AttentionOutput)]           = "blk.{}.attn_output";
						res[static_cast<size_t>(ModelTensorType::AttentionRotationEmbedding)] = "blk.{}.attn_rot_embd";
						res[static_cast<size_t>(ModelTensorType::FFNGateInp)]                = "blk.{}.ffn_gate_inp";
						res[static_cast<size_t>(ModelTensorType::FFNNorm)]                   = "blk.{}.ffn_norm";
						res[static_cast<size_t>(ModelTensorType::FFNGate)]                   = "blk.{}.ffn_gate";
						res[static_cast<size_t>(ModelTensorType::FFNDown)]                   = "blk.{}.ffn_down";
						res[static_cast<size_t>(ModelTensorType::FFNUp)]                     = "blk.{}.ffn_up";
						res[static_cast<size_t>(ModelTensorType::FFNGateExp)]                = "blk.{}.ffn_gate.{}";
						res[static_cast<size_t>(ModelTensorType::FFNDownExp)]                = "blk.{}.ffn_down.{}";
						res[static_cast<size_t>(ModelTensorType::FFNUpExp)]                  = "blk.{}.ffn_up.{}";
						return res;
					}()
				};
			default:
				spy_assert(false, "Unsupported model architecture: {}", magic_enum::enum_name(model_type));
		}
		return {};
	}

	inline constexpr ModelType get_arch_type_from_name(std::string_view name) {
		constexpr size_t NUM_ARCH_NUM = static_cast<size_t>(ModelType::ModelTypeEnd);
		for (size_t type = 0; type < NUM_ARCH_NUM; ++type) {
			auto cur_table = get_tensor_array(static_cast<ModelType>(type));
			if (cur_table.architecture_name == name) { return static_cast<ModelType>(type); }
		}
		return ModelType::ModelTypeEnd;
	}

	struct TensorNameTable {
	public:
		ModelType               model_type;
		ModelArchitectureTable  arch_table;

	public:
		constexpr TensorNameTable(): model_type(ModelType::ModelTypeEnd) {}
		constexpr TensorNameTable(ModelType model_type): model_type(model_type) { load(model_type); }

		constexpr void load(const ModelType new_model_type) {
			model_type = new_model_type;
			arch_table = get_tensor_array(model_type);
		}

	public:
		std::string operator()(ModelTensorType tensor_type) const {
			return std::string(get_tensor_fmt(tensor_type));
		}

		template<class ...Args>
		std::string operator()(ModelTensorType tensor_type, Args &&...args) const {
			return fmt::vformat(get_tensor_fmt(tensor_type), fmt::make_format_args(std::forward<Args>(args)...));
		}

		constexpr std::string_view get_tensor_fmt(ModelTensorType tensor_type) const {
			return arch_table.tensor_fmt_array[static_cast<size_t>(tensor_type)];
		}
	};

} // namespace spy