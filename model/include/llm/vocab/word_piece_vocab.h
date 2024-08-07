/*
 * @author: BL-GS 
 * @date:   2024/4/2
 */

#pragma once

#include <string>
#include <vector>

#include "util/unicode.h"
#include "llm/vocab/type.h"
#include "llm/vocab/abstract_vocab.h"

namespace spy {

	struct WordPieceVocab final: Vocab {
	public:
		WordPieceVocab(const ModelMetaContext &context): Vocab(context) {
			vocab_type = ModelVocabType::WordPiece;
		}

		~WordPieceVocab() noexcept override = default;

	public:
		void init_newline_token([[maybe_unused]] const spy::ModelMetaContext &context) override {
			linefeed_id = special_pad_id;
		}

	public:
		std::vector<TokenID> tokenize(const std::string &raw_text, bool bos, bool special) override {
			spy_abort("Unimplemented");
			return {};
		}

	public:
		TokenID byte_to_token_id(uint8_t ch) const override {
			return token_id_table.at(unicode_byte_to_utf8(ch));
		}

		uint8_t token_id_to_byte([[maybe_unused]] TokenID token_id) const override {
			spy_abort("Try toe convert token id using unsupported vocab");
			return 0;
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
					spy_abort("Unknown model token type: {}", token_type);
				}
			}
			return {};
		}
	};


}