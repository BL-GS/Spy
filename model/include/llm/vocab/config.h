/*
 * @author: BL-GS 
 * @date:   2024/4/2
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

#include "util/shell/logger.h"
#include "llm/vocab/type.h"

namespace spy {

	using TokenID = int;

	struct TokenData {
		ModelTokenType  type;
		float           score;
		std::string     text;
	};

	enum class VocabSpecialAdd: int { Unknown, Add, NotAdd };

	enum class FragmentBufferType { RawText, Token };

	struct FragmentBufferVariant {
		const FragmentBufferType    type;
		const TokenID               token_id;
		const std::string           dummy;
		const std::string &         raw_text;
		const uint64_t              offset;
		const uint64_t              length;

		FragmentBufferVariant(TokenID token_id):
				type(FragmentBufferType::Token), token_id(token_id),
				raw_text(dummy), offset(0), length(0) {}

		FragmentBufferVariant(const std::string &new_raw_text, int64_t new_offset, int64_t new_length):
				type(FragmentBufferType::RawText), token_id(-1),
				raw_text(new_raw_text), offset(new_offset), length(new_length) {
			spy_assert(offset >= 0);
			spy_assert(length >= 1);
			spy_assert(offset + length <= raw_text.length());
		}
	};

	struct LLMSymbol {
		using index = int;

		index           prev;
		index           next;
		const char *    text;
		size_t          n;
	};

}  // namespace spy