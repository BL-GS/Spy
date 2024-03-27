/*
 * @author: BL-GS 
 * @date:   2024/4/2
 */

#pragma once

#include "model/vocab/config.h"

namespace spy {

	struct TokenCandidate {
		TokenID token_id;
		float   logit;
	};

	using TokenCandidateArray = std::vector<TokenCandidate>;

} // namespace spy