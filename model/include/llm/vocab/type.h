/*
 * @author: BL-GS 
 * @date:   2024/3/31
 */

#pragma once

#include "util/type/enum.h"

namespace spy {

	enum class ModelVocabType : int {
		None             = 0,
		SentencePiece    = 1,
		BytePairEncoding = 2,
		WordPiece        = 3
	};

	enum class ModelTokenType: int {
		Undefined   = 0,
		Normal      = 1,
		Unknown     = 2,
		Control     = 3,
		UserDefined = 4,
		Unused      = 5,
		Byte        = 6
	};

} // namespace spy

SPY_ENUM_FORMATTER(spy::ModelVocabType);
SPY_ENUM_FORMATTER(spy::ModelTokenType);

