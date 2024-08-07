#pragma once

#include <string_view>

#include "adapter/abstract_adapter.h"

namespace spy {

	/// 
	/// @brief An file adapter for gguf files
	/// 
	class GGUFAdapter final: public FileAdapter {
	public:
		GGUFAdapter() = default;

		virtual ~GGUFAdapter() = default;

	public:
		ModelMetaContext init_from_file(std::string_view filename) override;
	};

}  // namespace spy