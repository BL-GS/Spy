#pragma once

#include <memory>
#include <magic_enum.hpp>

#include "model/sample/type.h"
#include "model/sample/config.h"
#include "model/sample/abstract_sample.h"
#include "model/sample/greedy.h"

namespace spy {

	///
	/// @brief The factory of sampler
	/// @details Please look up the SamplerImpl<SamplerType> for the detail of sampler
	///
    class SamplerFactory {
    public:
        static std::unique_ptr<Sampler> build_sampler(SamplerType sampler_type) {
	        return magic_enum::enum_switch([](const auto T_sampler_type){
				return static_cast<std::unique_ptr<Sampler>>(std::make_unique<SamplerImpl<T_sampler_type>>());
			}, sampler_type);
        }

		template<SamplerType T_sampler_type>
		static auto build_sampler() {
			return SamplerImpl<T_sampler_type>();
		}

    };
    
} // namespace spy