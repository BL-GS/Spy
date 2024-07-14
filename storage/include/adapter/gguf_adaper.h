#pragma once

#include <string_view>

#include "adapter/abstract_adapter.h"

namespace spy {

	class GGUFAdapter final: public AbstractFileAdapter {
	public:
		GGUFAdapter() = default;

		virtual ~GGUFAdapter() = default;

	public:
		void init_from_file(const std::string_view filename) override;
	};

}  // namespace spy