/*
 * @author: BL-GS 
 * @date:   2024/4/2
 */

#pragma once

#include <cstdint>

#include "model/vocab/config.h"
#include "model/sample/config.h"

namespace spy {

	class Sampler {
	public:
		Sampler() = default;

		virtual ~Sampler() noexcept = default;

	public:
		virtual TokenID sample(TokenCandidateArray &candidate) = 0;
	};

	template<SamplerType T_sampler_type>
	struct SamplerImpl {};

}  // namespace spy