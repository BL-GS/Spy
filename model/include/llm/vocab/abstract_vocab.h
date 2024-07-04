/*
 * @author: BL-GS 
 * @date:   2024/3/31
 */

#pragma once

#include <forward_list>
#include <string>
#include <vector>

#include "metadata.h"
#include "llm/vocab/type.h"
#include "llm/vocab/config.h"

namespace spy {

	struct AbstractVocab {
	public:
		/* Special ID definition */
		TokenID special_bos_id;
		TokenID special_eos_id;
		TokenID special_unk_id;
		TokenID special_sep_id;
		TokenID special_pad_id;

		TokenID linefeed_id;
		TokenID special_prefix_id;
		TokenID special_middle_id;
		TokenID special_suffix_id;
		TokenID special_eot_id;

		VocabSpecialAdd special_add_bos;
		VocabSpecialAdd special_add_eos;

		bool add_space_prefix;

		ModelVocabType vocab_type;
		TokenID        num_vocab;

		/* Lookup table */
		std::map<std::string, TokenID> token_id_table;
		std::vector<TokenData>         token_data_table;
		std::map<std::string, TokenID> special_tokens_cache;

	public:
		AbstractVocab(const ModelMetaContext &context):
			special_bos_id(1), special_eos_id(2), special_unk_id(0), special_sep_id(-1), special_pad_id(-1),
			linefeed_id(13), special_prefix_id(32007), special_middle_id(32009), special_suffix_id(32008), special_eot_id(32010),
			special_add_bos(VocabSpecialAdd::Unknown), special_add_eos(VocabSpecialAdd::Unknown), add_space_prefix(true),
			vocab_type(ModelVocabType::SentencePiece), num_vocab(0) {
		}

		virtual ~AbstractVocab() noexcept = default;

	protected: /* Initialization */
		void init(const ModelMetaContext &context) {
			/* Initialize token_id_table and token_data_table */
			init_token_table(context);
			/* Determine the newline token */
			init_newline_token(context);
			/* Special token */
			init_special_token(context);
			/* Handle ad_bos_token and add_eos_token */
			init_eos_bos(context);
			/* TODO: validate */
		}

		void init_token_table(const ModelMetaContext &context) {
			const auto &token = context.find_gguf_value(LLMKey::TOKENIZER_LIST).get_value<ModelMetaArray>();
			num_vocab         = token.size();

			const auto score_option      = context.find_gguf_value_option(LLMKey::TOKENIZER_SCORES);
			const auto token_type_option = context.find_gguf_value_option(LLMKey::TOKENIZER_TOKEN_TYPE);

			token_data_table.resize(num_vocab);
			for (int i = 0; i < num_vocab; ++i) {
				const std::string word = std::get<std::string>(token[i]);

				token_id_table.insert_or_assign(std::string(word), static_cast<TokenID>(i));
				token_data_table[i] = {
						.type  = ModelTokenType::Normal,
						.score = 0.0f,
						.text  = word
				};
			}

			if (score_option.has_value()) {
				const ModelMetaArray &score_array = score_option->get_value<ModelMetaArray>();
				for (int i = 0; i < num_vocab; ++i) {
					token_data_table[i].score = std::get<float>(score_array[i]);
				}
			}

			if (token_type_option.has_value()) {
				const ModelMetaArray &type_array = token_type_option->get_value<ModelMetaArray>();
				for (int i = 0; i < num_vocab; ++i) {
					token_data_table[i].type = static_cast<ModelTokenType>(std::get<int32_t>(type_array[i]));
				}
			}
		}

		void init_special_token(const ModelMetaContext &context) {
			auto reset_special_token_id = [this, &context](int32_t &token_id, LLMKey key) {
				const auto new_id_option = context.find_gguf_value_option(key);
				if (!new_id_option.has_value()) { return; }
				const uint32_t new_id = new_id_option.value().get_value<uint32_t>();
				if (new_id >= token_id_table.size()) {
					spy_warn("Bad special token for {}: {}, using default id {}", key, new_id, token_id);
				} else {
					token_id = new_id;
				}
			};
			reset_special_token_id(special_bos_id, LLMKey::TOKENIZER_BOS_ID);
			reset_special_token_id(special_eos_id, LLMKey::TOKENIZER_EOS_ID);
			reset_special_token_id(special_unk_id, LLMKey::TOKENIZER_UNK_ID);
			reset_special_token_id(special_sep_id, LLMKey::TOKENIZER_SEP_ID);
			reset_special_token_id(special_pad_id, LLMKey::TOKENIZER_PAD_ID);
		}

		void init_eos_bos(const ModelMetaContext &context) {
			const auto add_bos_option = context.find_gguf_value_option(LLMKey::TOKENIZER_ADD_BOS);
			if (add_bos_option.has_value()) { special_add_bos = add_bos_option->get_value<bool>() ? VocabSpecialAdd::Add : VocabSpecialAdd::NotAdd; }
			const auto add_eos_option = context.find_gguf_value_option(LLMKey::TOKENIZER_ADD_EOS);
			if (add_eos_option.has_value()) { special_add_eos = add_eos_option->get_value<bool>() ? VocabSpecialAdd::Add : VocabSpecialAdd::NotAdd; }
		}

		void init_special_token_cache(const ModelMetaContext &context) {
			uint32_t special_tokens_count_by_type           = 0;
			uint32_t special_tokens_count_from_verification = 0;
			bool     special_tokens_definition_mismatch     = false;

			for (const auto &token_pair: token_id_table) {
				const auto &token_str = token_pair.first;
				const auto &token_id  = token_pair.second;

				// Count all non-normal tokens in the vocab with iterating
				if (token_data_table[token_id].type != ModelTokenType::Normal) { ++special_tokens_count_by_type; }

				// Skip single character tokens
				if (token_str.length() > 1) {
					bool is_tokenizable = false;

					// Split token string representation in two, in all possible ways and check if both halves can be matched to a valid token
					for (size_t i = 0; i < token_str.length(); ++i) {
						const auto left  = token_str.substr(0, 1);
						const auto right = token_str.substr(1);

						// Check if we didn't partition in the middle of a utf sequence
						const size_t utf = utf8_len(left.back());

						if (utf == 1) {
							if (token_id_table.find(left) != token_id_table.end() && token_id_table.find(right) != token_id_table.end()) {
								is_tokenizable = true;
								break;
							}
							++i;
						} else {
							i += utf - 1;
						}
					}

					if (!is_tokenizable) {
						// Some tokens are multibyte, but they are utf sequences with equivalent text length of 1
						//  it's faster to re-filter them here, since there are way less candidates now

						// Calculate a total "utf" length of a token string representation
						size_t utf8_str_len = 0;
						for (size_t i =0; i < token_str.length();) {
							++utf8_str_len;
							i += utf8_len(token_str.at(i));
						}

						// And skip the ones which are one character
						if (utf8_str_len > 1) {
							// At this point what we have left are special tokens only
							special_tokens_cache[token_str] = token_id;
							// Count manually found special tokens
							special_tokens_count_from_verification++;
							// If this manually found special token is not marked as such, flag a mismatch
							if (token_data_table[token_id].type == ModelTokenType::Normal) {
								special_tokens_definition_mismatch = true;
							}
						}
					}
				}
			}

			if (special_tokens_definition_mismatch || special_tokens_count_from_verification != special_tokens_count_by_type) {
				spy_warn("mismatch in special tokens definition ( {}/{} vs {}/{} ).\n",
				               special_tokens_count_from_verification, token_data_table.size(),
				               special_tokens_count_by_type, token_data_table.size()
				);
			} else {
				spy_info("special tokens definition check successful ( {}/{} ).\n",
				               special_tokens_count_from_verification, token_data_table.size()
				);
			}
		}

		virtual void init_newline_token(const ModelMetaContext &context) = 0;

	public:
		virtual std::vector<TokenID> tokenize(const std::string &raw_text, bool bos, bool special) = 0;

	public:
		TokenID   get_token_id(const TokenData& token_data) { return token_id_table[token_data.text]; }

		TokenData get_token_data(const TokenID id) const { return token_data_table[id]; }

	public:
		ModelTokenType get_token_type(TokenID token_id)     const { return token_data_table[token_id].type; }
		bool is_normal_token(TokenID token_id)              const { return token_data_table[token_id].type == ModelTokenType::Normal; }
		bool is_unknown_token(TokenID token_id)             const { return token_data_table[token_id].type == ModelTokenType::Unknown; }
		bool is_control_token(TokenID token_id)             const { return token_data_table[token_id].type == ModelTokenType::Control; }
		bool is_byte_token(TokenID token_id)                const { return token_data_table[token_id].type == ModelTokenType::Byte; }
		bool is_user_defined_token(TokenID token_id)        const { return token_data_table[token_id].type == ModelTokenType::UserDefined; }

	public:
		virtual TokenID             byte_to_token_id(uint8_t ch)       const = 0;
		virtual uint8_t             token_id_to_byte(TokenID token_id) const = 0;
		virtual std::string   		token_to_piece(TokenID token_id)   const = 0;

		static constexpr size_t utf8_len(char src) {
			constexpr size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
			const uint8_t highbits = static_cast<uint8_t>(src) >> 4;
			return lookup[highbits];
		}

		static std::string escape_whitespace(const std::string &text) {
			return replace_all(text, " ", "\xe2\x96\x81");
		}

		static std::string unescape_whitespace(const std::string &word) {
			return replace_all(word, "\xe2\x96\x81", " ");
		}


		static std::string replace_all(const std::string & s, const std::string & search, const std::string & replace) {
			std::string result;
			for (size_t pos = 0; ; pos += search.length()) {
				auto new_pos = s.find(search, pos);
				if (new_pos == std::string::npos) {
					result += s.substr(pos, s.size() - pos);
					break;
				}
				result += s.substr(pos, new_pos - pos) + replace;
				pos = new_pos;
			}
			return result;
		}

		void token_special_token_partition(std::forward_list<FragmentBufferVariant> &fragment_buffer) const {
			for (const auto &special_token: special_tokens_cache) {
				const auto &special_token_str = special_token.first;
				const auto &special_token_id  = special_token.second;

				auto iter = fragment_buffer.begin();
				while (iter != fragment_buffer.end()) {
					auto &fragment = *iter;

					if (fragment.type == FragmentBufferType::RawText) {
						const auto &raw_text = fragment.raw_text;

						uint64_t raw_text_base_offset = fragment.offset;
						uint64_t raw_text_base_length = fragment.length;

						// loop over the text
						while (true) {
							// find the first occurrence of a given special token in this fragment
							//  passing offset argument only limit the "search area" but match coordinates
							//  are still relative to the source full raw_text
							const auto match = raw_text.find(special_token_str, raw_text_base_offset);
							// no occurrences found, stop processing this fragment for a given special token
							if (match == std::string::npos) break;
							// check if match is within bounds of offset <-> length
							if (match + special_token_str.length() > raw_text_base_offset + raw_text_base_length) { break; }

							auto source = std::distance(fragment_buffer.begin(), iter);

							// if match is further than base offset
							//  then we have some text to the left of it
							if (match > raw_text_base_offset) {
								// left
								const int64_t left_reminder_offset = raw_text_base_offset + 0;
								const int64_t left_reminder_length = match - raw_text_base_offset;
								fragment_buffer.emplace_after(iter, raw_text, left_reminder_offset, left_reminder_length);
								iter++;
							}

							// special token
							fragment_buffer.emplace_after(iter, special_token_id);
							iter++;

							// right
							if (match + special_token_str.length() < raw_text_base_offset + raw_text_base_length) {
								const int64_t right_reminder_offset = match + special_token_str.length();
								const int64_t right_reminder_length = raw_text_base_length - ((match - raw_text_base_offset) + special_token_str.length());
								fragment_buffer.emplace_after(iter, raw_text, right_reminder_offset, right_reminder_length);

								iter++;

								if (source == 0) {
									fragment_buffer.erase_after(fragment_buffer.before_begin());
								} else {
									fragment_buffer.erase_after(std::next(fragment_buffer.begin(), (source - 1)));
								}

								// repeat for the right side
								raw_text_base_offset = right_reminder_offset;
								raw_text_base_length = right_reminder_length;

							} else {
								if (source == 0) {
									fragment_buffer.erase_after(fragment_buffer.before_begin());
								} else {
									fragment_buffer.erase_after(std::next(fragment_buffer.begin(), (source - 1)));
								}
								break;
							}
						}
					}

					++iter;
				}
			}
		}
	};

}