/*
 * @author: BL-GS 
 * @date:   2024/3/31
 */

#pragma once

#include <cstdint>
#include <vector>

#include "llm/type.h"
#include "llm/vocab/config.h"
#include "operator/operator.h"

namespace spy {

    struct ModelIO {
    public:
        /* Input */

		size_t								acc_token = 0;
        /// The token ids of the input (used when embedding is empty)
        std::vector<TokenID>                token_id_array;
        /// The token embedding
        std::vector<float>                  embedding;
        /// The position of the respective token in the sequence
        std::vector<int32_t>                positions;
        /// The sequence id
        std::vector<std::vector<int32_t>>   sequence_id;
        /// The logits (and / or the embeddings) for the respecitive token will not be output if false
        std::vector<bool>                   enable_logits;

        /* Output*/

        std::vector<float>                  logits;

	public:
		void reset() {
			token_id_array.clear();
			embedding.clear();
			positions.clear();
			sequence_id.clear();
			enable_logits.clear();
			logits.clear();
		}

        void add(TokenID token_id, int32_t pos, std::vector<int32_t> &&seq_ids, bool enable_logit) {
            token_id_array.push_back(token_id);
            positions.push_back(pos);
            sequence_id.push_back(std::forward<std::vector<int32_t>>(seq_ids));
            enable_logits.push_back(enable_logit);
			++acc_token;
        }

        void add(TokenID token_id, std::vector<int32_t> &&seq_ids, bool enable_logit) {
            token_id_array.push_back(token_id);
            positions.push_back(acc_token);
            sequence_id.push_back(std::forward<std::vector<int32_t>>(seq_ids));
            enable_logits.push_back(enable_logit);
			++acc_token;
        }

		size_t num_token() const { return token_id_array.size(); }
    };

	struct HyperParam {
		uint32_t 				num_context;

		RopeType				rope_type;
		ModelRopeScalingType 	rope_scaling_type;
		ModelPoolingType     	rope_pooling_type;
		/// RoPE base frequency, 0 = from model
		float    				rope_freq_base;   
		/// RoPE frequency scaling factor, 0 = from model
		float    				rope_freq_scale;  
		/// YaRN extrapolation mix factor, negative = from model
		float    				yarn_ext_factor;  
		/// YaRN magnitude scaling factor
		float    				yarn_attn_factor; 
		/// YaRN low correction dim
		float    				yarn_beta_fast;   
		/// YaRN high correction dim
		float    				yarn_beta_slow;   
		/// YaRN original context size
		uint32_t 				yarn_orig_ctx;   

		static ModelRopeScalingType parse_rope_scaling_type(const std::string_view rope_scaling) {
			if (rope_scaling == "none") {
				return ModelRopeScalingType::None;
			} else if (rope_scaling == "linear") {
				return ModelRopeScalingType::Linear;
			} else if (rope_scaling == "yarn") {
				return ModelRopeScalingType::Yarn;
			}
			return ModelRopeScalingType::Unspecified;
		}

		static ModelPoolingType parse_pooling_type(const std::string_view pooling) {
			if (pooling == "none") {
				return ModelPoolingType::None;
			} else if (pooling == "mean") {
				return ModelPoolingType::Mean;
			} else if (pooling == "cls") {
				return ModelPoolingType::Cls;
			}
			return ModelPoolingType::Unspecific;
		}
	};

} // namespace spy