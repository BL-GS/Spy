/*
 * @author: BL-GS 
 * @date:   2024/4/2
 */

#pragma once

#include "util/unicode.h"
#include "model/vocab/type.h"
#include "model/vocab/abstract_vocab.h"

namespace spy {

	struct WordPieceVocab: AbstractVocab {
	public:
		WordPieceVocab(const GGUFContext &context): AbstractVocab(context) {
			vocab_type = ModelVocabType::WordPiece;
		}

		~WordPieceVocab() noexcept override = default;

	public:
		void init_newline_token([[maybe_unused]] const spy::GGUFContext &context) override {
			linefeed_id = special_pad_id;
		}

	public:
		std::vector<TokenID> tokenize(const std::string &raw_text, bool bos, bool special) override {
			SPY_ASSERT(false, "Unimplement");
			return {};
		}

	public:
		TokenID byte_to_token_id(uint8_t ch) const override {
			return token_id_table.at(unicode_byte_to_utf8(ch));
		}

		uint8_t token_id_to_byte([[maybe_unused]] TokenID token_id) const override {
			SPY_ASSERT(false, "Try toe convert token id using unsupported vocab");
		}

		std::string token_to_piece(TokenID token_id) const override {
			if (0 <= token_id && token_id < num_vocab) {
				const TokenData token_data 		= token_data_table.at(token_id);
				const ModelTokenType token_type = token_data.type;

				switch (token_type) {
				case ModelTokenType::Normal: {
					std::string res = token_data.text;
					unescape_whitespace(res);
					return res;
				}

				case ModelTokenType::Byte:
					return { static_cast<char>(token_id_to_byte(token_id)) };

				case ModelTokenType::UserDefined:
					return token_data.text;

				case ModelTokenType::Unknown:
					return "\xe2\x96\x85";

				case ModelTokenType::Control:
					return {};

				case ModelTokenType::Undefined:
				case ModelTokenType::Unused:
				default:
					SPY_ASSERT_FMT(false, "Unknown model token type: {}", magic_enum::enum_name(token_type));
				}
			}
			return {};
		}
	};


}