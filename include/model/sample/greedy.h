/*
 * @author: BL-GS 
 * @date:   2024/4/2
 */

#pragma once

#include "model/sample/abstract_sample.h"

namespace spy {

	class GreedySampler final: public AbstractSampler {
	public:
		GreedySampler() = default;

		~GreedySampler() noexcept override = default;

	public:
		TokenID sample(TokenCandidateArray &candidate) override {
			const auto max_iter = std::max_element(candidate.begin(), candidate.end(), [](const TokenCandidate &candidate_a, const TokenCandidate &candidate_b) {
				return candidate_a.logit < candidate_b.logit;
			});
			return max_iter->token_id;
		}

	};

}  // namespace spy