/*
 * @author: BL-GS 
 * @date:   2024/4/2
 */

#pragma once

#include <cstdint>

#include "model/vocab/config.h"
#include "model/sample/config.h"

namespace spy {

	class AbstractSampler {
	public:
		AbstractSampler() = default;

		virtual ~AbstractSampler() noexcept = default;

	public:
		virtual TokenID sample(TokenCandidateArray &candidate) = 0;

	};

}  // namespace spy