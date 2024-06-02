#pragma once

#include <string_view>
#include <array>
#include <fmt/format.h>
#include <magic_enum.hpp>

#include "util/shell/logger.h"
#include "model/type.h"

namespace spy {
    
	struct ModelArchitectureTable {
		std::string_view                                 architecture_name;
	};

	inline constexpr ModelArchitectureTable get_tensor_array(ModelType model_type) {
		switch (model_type) {
			case ModelType::LLaMa:
				return {
					.architecture_name = "llama",
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
	};

} // namespace spy