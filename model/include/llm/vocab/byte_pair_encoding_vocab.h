
#pragma once

#include <forward_list>
#include <queue>
#include <vector>
#include <string>
#include <map>

#include "util/unicode.h"
#include "llm/vocab/abstract_vocab.h"

namespace spy {

	struct BytePairEncodingVocab final: AbstractVocab {
	protected:
		struct LLMBigram {
		public:
			struct comparator {
				bool operator()(LLMBigram & lhs, LLMBigram & rhs) {
					return (lhs.rank < rhs.rank) || (lhs.rank == rhs.rank && lhs.left > rhs.left);
				}
			};

		public:
			using queue_storage = std::vector<LLMBigram>;
			using queue = std::priority_queue<LLMBigram, queue_storage, comparator>;

		public:
			LLMSymbol::index    left;
			LLMSymbol::index    right;
			std::string 		text;
			int					rank;
			size_t              size;
		};

	protected:
		std::vector<LLMSymbol> 								symbols;
		std::vector<LLMSymbol> 								symbols_final;

		LLMBigram::queue        							work_queue;
		std::map<std::pair<std::string, std::string>, int> 	bpe_ranks;

	public:
		BytePairEncodingVocab(const ModelMetaContext &context): AbstractVocab(context) {
			vocab_type = ModelVocabType::BytePairEncoding;
		}

		~BytePairEncodingVocab() noexcept override = default;

	public:
		void init_newline_token([[maybe_unused]] const ModelMetaContext &context) override {
			std::vector<int> ids = tokenize("\u010A", false, false);
			spy_assert(!ids.empty(), "Missing newline token in model vocab");
			linefeed_id = ids[0];
		}

	public:
		std::vector<TokenID> tokenize(const std::string &raw_text, bool bos, bool special) override {
			std::vector<TokenID> output;

			if (bos && special_bos_id != -1) { output.push_back(special_bos_id); }

			if (raw_text.empty()) { return output; }

			std::forward_list<FragmentBufferVariant> fragment_buffer;
			fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
			if (special) { token_special_token_partition(fragment_buffer); }

			for (const auto &fragment: fragment_buffer) {
				if (fragment.type == FragmentBufferType::RawText) {
					// without adding this leading whitespace, we do not get the same results as the original tokenizer
					auto new_raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);
					tokenize_inner(new_raw_text, output);
				} else {
					output.push_back(fragment.token_id);
				}
			}

			return output;
		}

		void tokenize_inner(const std::string & text, std::vector<TokenID> & output) {
			int final_prev_index = -1;
			auto word_collection = bpe_gpt2_preprocess(text);

			symbols_final.clear();

			for (auto & word : word_collection) {
				work_queue = LLMBigram::queue();
				symbols.clear();

				int    index  = 0;
				size_t offset = 0;

				while (offset < word.size()) {
					LLMSymbol sym{};
					size_t char_len  = std::min(word.size() - offset, utf8_len(word[offset]));
					       sym.text  = word.c_str() + offset;
					       sym.n     = char_len;
					       offset   += sym.n;
					       sym.prev  = index - 1;
					       sym.next  = offset == word.size() ? -1 : index + 1;
					index++;
					symbols.emplace_back(sym);
				}
				for (size_t i = 1; i < symbols.size(); ++i) {
					add_new_bigram(i - 1, i);
				}

				// build token(s)
				while (!work_queue.empty()) {
					auto bigram = work_queue.top();
					work_queue.pop();

					auto & left_symbol = symbols[bigram.left];
					auto & right_symbol = symbols[bigram.right];

					if (left_symbol.n == 0 || right_symbol.n == 0) {
						continue;
					}
					std::string left_token = std::string(left_symbol.text, left_symbol.n);
					std::string right_token = std::string(right_symbol.text, right_symbol.n);
					if (left_token + right_token != bigram.text) {
						continue;  // Skip this bigram if it's outdated
					}

					// merge the right sym into the left one
					left_symbol.n += right_symbol.n;
					right_symbol.n = 0;

					// remove the right sym from the chain
					left_symbol.next = right_symbol.next;
					if (right_symbol.next >= 0) {
						symbols[right_symbol.next].prev = bigram.left;
					}

					add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
					add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
				}

				// add the finished tokens to the final list keeping correct order for next and prev
				for (auto & sym : symbols) {
					if (sym.n > 0) {
						sym.prev = final_prev_index;
						sym.next = -1;
						if (final_prev_index != -1) {
							symbols_final[final_prev_index].next = symbols_final.size();
						}
						symbols_final.emplace_back(sym);
						final_prev_index = symbols_final.size() - 1;
					}
				}
			}

			symbols = symbols_final;

			if (!symbols.empty()) {
				for (int i = 0; i != -1; i = symbols[i].next) {
					auto & symbol = symbols[i];
					if (symbol.n == 0) {
						continue;
					}

					const std::string str = std::string(symbol.text, symbol.n);
					const auto token = token_id_table.find(str);

					if (token == token_id_table.end()) {
						for (char byte_str : str) {
							auto token_multibyte = token_id_table.find({byte_str});
							if (token_multibyte == token_id_table.end()) {
								throw std::runtime_error("ERROR: byte not found in vocab");
							}
							output.push_back((*token_multibyte).second);
						}
					} else {
						output.push_back((*token).second);
					}
				}
			}
		}

	public:
		TokenID byte_to_token_id(uint8_t ch) const override {
			return token_id_table.at(unicode_byte_to_utf8(ch));
		}

		uint8_t token_id_to_byte([[maybe_unused]] TokenID token_id) const override {
			spy_abort("Try toe convert token id using unsupported vocab");
			return 0;
		}

		std::string token_to_piece(spy::TokenID token_id) const override {
			if (0 <= token_id && token_id < num_vocab) {
				const TokenData token_data 		= token_data_table.at(token_id);
				const ModelTokenType token_type = token_data.type;

				switch (token_type) {
				case ModelTokenType::Normal: {
					const std::string &text = token_data.text;
					std::string res;
					const auto unicode_sequences = unicode_cpts_from_utf8(text);
					for (const auto & unicode_sequence : unicode_sequences) {
						res += static_cast<char>(unicode_utf8_to_byte(unicode_cpt_to_utf8(unicode_sequence)));
					}

					return res;
				}

				case ModelTokenType::UserDefined:
					return token_data.text;

				case ModelTokenType::Control:
					return {};

				default:
					spy_abort("Unknown model token type: {}", token_type);
				}
			}
			return {};
		}

	private:
		void add_new_bigram(int left, int right) {
			if (left == -1 || right == -1) {
				return;
			}

			std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
			std::string right_token = std::string(symbols[right].text, symbols[right].n);

			int rank_found = -1;

			rank_found = find_bpe_rank(left_token, right_token);

			if (rank_found < 0) { return; }

			LLMBigram bigram;

			bigram.left  = left;
			bigram.right = right;
			bigram.text  = left_token + right_token;
			bigram.size  = left_token.size() + right_token.size();
			bigram.rank  = rank_found;

			work_queue.push(bigram);
		}

		static std::vector<std::string> bpe_gpt2_preprocess(const std::string & text) {
			std::vector<std::string> bpe_words;
			std::vector<std::string> bpe_encoded_words;

			std::string token;
			// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
			bool collecting_numeric              = false;
			bool collecting_letter               = false;
			bool collecting_special              = false;
			bool collecting_whitespace_lookahead = false;
			bool collecting                      = false;

			std::vector<std::string> text_utf;
			text_utf.reserve(text.size());
			bpe_words.reserve(text.size());
			bpe_encoded_words.reserve(text.size());

			const auto cpts = unicode_cpts_from_utf8(text);
			for (unsigned int cpt : cpts) { text_utf.emplace_back(unicode_cpt_to_utf8(cpt)); }

			for (int i = 0; i < static_cast<int>(text_utf.size()); i++) {
				const std::string & utf_char = text_utf[i];
				bool split_condition = false;
				int bytes_remain = text_utf.size() - i;
				// forward backward lookups
				const std::string & utf_char_next = (i + 1 < static_cast<int>(text_utf.size())) ? text_utf[i + 1] : "";
				const std::string & utf_char_next_next = (i + 2 < static_cast<int>(text_utf.size())) ? text_utf[i + 2] : "";

				// handling contractions
				if (!split_condition && bytes_remain >= 2) {
					// 's|'t|'m|'d
					if (utf_char == "\'" && (utf_char_next == "s" || utf_char_next == "t" || utf_char_next == "m" || utf_char_next == "d")) {
						split_condition = true;
					}
					if (split_condition) {
						if (!token.empty()) {
							bpe_words.emplace_back(token); // push previous content as token
						}
						token = utf_char + utf_char_next;
						bpe_words.emplace_back(token);
						token = "";
						i++;
						continue;
					}
				}
				if (!split_condition && bytes_remain >= 3) {
					// 're|'ve|'ll
					if (utf_char == "\'" && (
						(utf_char_next == "r" && utf_char_next_next == "e") ||
						(utf_char_next == "v" && utf_char_next_next == "e") ||
						(utf_char_next == "l" && utf_char_next_next == "l"))
						) {
						split_condition = true;
					}
					if (split_condition) {
						// current token + next token can be defined
						if (!token.empty()) {
							bpe_words.emplace_back(token); // push previous content as token
						}
						token = utf_char + utf_char_next + utf_char_next_next;
						bpe_words.emplace_back(token); // the contraction
						token = "";
						i += 2;
						continue;
					}
				}

				if (!split_condition && !collecting) {
					if (unicode_cpt_type(utf_char) == CodePointType::Letter || (token.empty() && utf_char == " " && unicode_cpt_type(utf_char_next) == CodePointType::Letter)) {
						collecting_letter = true;
						collecting        = true;
					}
					else if (unicode_cpt_type(utf_char) == CodePointType::Digit || (token.empty() && utf_char == " " && unicode_cpt_type(utf_char_next) == CodePointType::Digit)) {
						collecting_numeric = true;
						collecting         = true;
					}
					else if (
						((unicode_cpt_type(utf_char) != CodePointType::Letter && unicode_cpt_type(utf_char) != CodePointType::Digit) && (unicode_cpt_type(utf_char) != CodePointType::WhiteSpace)) ||
						(token.empty() && utf_char == " " && unicode_cpt_type(utf_char_next) != CodePointType::Letter && unicode_cpt_type(utf_char_next) != CodePointType::Digit && unicode_cpt_type(utf_char_next) != CodePointType::WhiteSpace)
						) {
						collecting_special = true;
						collecting = true;
					}
					else if (unicode_cpt_type(utf_char) == CodePointType::WhiteSpace && unicode_cpt_type(utf_char_next) == CodePointType::WhiteSpace) {
						collecting_whitespace_lookahead = true;
						collecting = true;
					}
					else if (unicode_cpt_type(utf_char) == CodePointType::WhiteSpace) {
						split_condition = true;
					}
				}
				else if (!split_condition && collecting) {
					if (collecting_letter && unicode_cpt_type(utf_char) != CodePointType::Letter) {
						split_condition = true;
					}
					else if (collecting_numeric && unicode_cpt_type(utf_char) != CodePointType::Digit) {
						split_condition = true;
					}
					else if (collecting_special && (unicode_cpt_type(utf_char) == CodePointType::Letter || unicode_cpt_type(utf_char) == CodePointType::Digit || unicode_cpt_type(utf_char) == CodePointType::WhiteSpace)) {
						split_condition = true;
					}
					else if (collecting_whitespace_lookahead && (unicode_cpt_type(utf_char_next) == CodePointType::Letter || unicode_cpt_type(utf_char_next) == CodePointType::Digit)) {
						split_condition = true;
					}
				}

				if (utf_char_next.empty()) {
					split_condition = true; // final
					token += utf_char;
				}

				if (split_condition) {
					if (!token.empty()) {
						bpe_words.emplace_back(token);
					}
					token                           = utf_char;
					collecting                      = false;
					collecting_letter               = false;
					collecting_numeric              = false;
					collecting_special              = false;
					collecting_whitespace_lookahead = false;
				}
				else {
					token += utf_char;
				}
			}

			for (std::string & word : bpe_words) {
				std::string encoded_token;
				for (char & c : word) {
					encoded_token += unicode_byte_to_utf8(c);
				}
				bpe_encoded_words.emplace_back(encoded_token);
			}

			return bpe_encoded_words;
		}

		int find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
        	spy_assert(token_left.find(' ') 	== std::string::npos);
			spy_assert(token_left.find('\n') 	== std::string::npos);
			spy_assert(token_right.find(' ') 	== std::string::npos);
			spy_assert(token_right.find('\n') 	== std::string::npos);

			auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
			if (it == bpe_ranks.end()) { return -1; }

			return it->second;
		}
	};

} // namespace spy