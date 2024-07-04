#pragma once

#include <memory>
#include <magic_enum.hpp>

#include "llm/sampler/type.h"
#include "llm/sampler/abstract_sample.h"

namespace spy {

	///
	/// @brief The factory of sampler
	/// @details Please look up the SamplerImpl<SamplerType> for the detail of sampler
	///
    class SamplerFactory {
    public:
        static std::unique_ptr<Sampler> build_sampler(SamplerType sampler_type);
    };
    
} // namespace spy