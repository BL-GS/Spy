/*
 * @author: BL-GS 
 * @date:   2024/4/2
 */

#pragma once

#include <charconv>
#include <cstddef>
#include <string>
#include <queue>
#include <vector>
#include <map>
#include <magic_enum.hpp>

#include "util/unicode.h"
#include "model/vocab/type.h"
#include "model/vocab/config.h"
#include "model/vocab/abstract_vocab.h"

namespace spy {

	struct SentencePieceVocab: AbstractVocab {
	protected:
		struct LLMBigram {
		public:
			struct comparator {
				bool operator()(LLMBigram & lhs, LLMBigram & rhs) {
					return (lhs.score < rhs.score) || (lhs.score == rhs.score && lhs.left > rhs.left);
				}
			};

		public:
			using queue_storage = std::vector<LLMBigram>;
			using queue = std::priority_queue<LLMBigram, queue_storage, comparator>;

		public:
			LLMSymbol::index    left;
			LLMSymbol::index    right;
			float               score;
			size_t              size;
		};


	protected:
		std::vector<LLMSymbol>  symbols;
		LLMBigram::queue        work_queue;
		std::map<std::string, std::pair<int, int>> rev_merge;

	public:
		SentencePieceVocab(const GGUFContext &context): AbstractVocab(context) {
			vocab_type = ModelVocabType::SentencePiece;
			init(context);
		}

		~SentencePieceVocab() noexcept override = default;

	public:
		void init_newline_token([[maybe_unused]] const GGUFContext &context) override {
			try {
				linefeed_id = byte_to_token_id('\n');
			} catch (const std::exception &err) {
				SPY_WARN_FMT("Special Piece Vocabulary: cannot find newline token: {}. Using `special_pad_id` instead.", err.what());
			}
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
					if (&fragment == &fragment_buffer.front()) {
						if (add_space_prefix) {
							new_raw_text.insert(0, " "); // prefix with space if the first token is not special
						}
					}

					new_raw_text = escape_whitespace(new_raw_text);
					tokenize_inner(new_raw_text, output);
				} else {
					output.push_back(fragment.token_id);
				}
			}

			return output;
		}

		void tokenize_inner(const std::string & text, std::vector<TokenID> & output) {
			// split string into utf8 chars
			int index = 0;
			size_t offs = 0;
			while (offs < text.size()) {
				LLMSymbol sym;
				size_t len = utf8_len(text[offs]);
				sym.text = text.c_str() + offs;
				sym.n = std::min(len, text.size() - offs);
				offs += sym.n;
				sym.prev = index - 1;
				sym.next = offs == text.size() ? -1 : index + 1;
				index++;
				symbols.emplace_back(sym);
			}

			// seed the work queue with all possible 2-character tokens.
			for (size_t i = 1; i < symbols.size(); ++i) {
				try_add_bigram(i - 1, i);
			}

			// keep substituting the highest frequency pairs for as long as we can.
			while (!work_queue.empty()) {
				auto bigram = work_queue.top();
				work_queue.pop();

				auto & left_sym = symbols[bigram.left];
				auto & right_sym = symbols[bigram.right];

				// if one of the symbols already got merged, skip it.
				if (left_sym.n == 0 || right_sym.n == 0 ||
				    left_sym.n + right_sym.n != bigram.size) {
					continue;
				}

				// merge the right sym into the left one
				left_sym.n += right_sym.n;
				right_sym.n = 0;

				//LLAMA_LOG_SPY_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

				// remove the right sym from the chain
				left_sym.next = right_sym.next;
				if (right_sym.next >= 0) {
					symbols[right_sym.next].prev = bigram.left;
				}

				// find more substitutions
				try_add_bigram(left_sym.prev, bigram.left);
				try_add_bigram(bigram.left, left_sym.next);
			}

			for (int i = 0; i != -1; i = symbols[i].next) {
				auto & symbol = symbols[i];
				resegment(symbol, output);
			}
		}

		void resegment(LLMSymbol & symbol, std::vector<TokenID> & output) {
			auto text = std::string(symbol.text, symbol.n);
			auto token = token_id_table.find(text);

			// Do we need to support is_unused?
			if (token != token_id_table.end()) {
				output.push_back((*token).second);
				return;
			}

			const auto p = rev_merge.find(text);

			if (p == rev_merge.end()) {
				// output any symbols that did not form tokens as bytes.
				output.reserve(output.size() + symbol.n);
				for (int j = 0; j < static_cast<int>(symbol.n); ++j) {
					TokenID token_id = byte_to_token_id(symbol.text[j]);
					output.push_back(token_id);
				}
				return;
			}

			resegment(symbols[p->second.first],  output);
			resegment(symbols[p->second.second], output);
		}

		void try_add_bigram(int left, int right) {
			if (left == -1 || right == -1) {
				return;
			}

			const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
			auto token = token_id_table.find(text);

			if (token == token_id_table.end()) {
				return;
			}

			if (static_cast<size_t>((*token).second) >= token_data_table.size()) {
				return;
			}

			const auto & tok_data = token_data_table[(*token).second];

			LLMBigram bigram {
					.left  = left,
					.right = right,
					.score = tok_data.score,
					.size  = text.size()
			};

			work_queue.push(bigram);

			// Do we need to support is_unused?
			rev_merge[text] = std::make_pair(left, right);
		}
	public:
		TokenID byte_to_token_id(uint8_t ch) const override {
			constexpr std::string_view HEX = "0123456789ABCDEF";
			const std::string buffer { '<', '0', 'x', HEX[ch >> 4U], HEX[ch & 15U], '>'};

			const auto iter = token_id_table.find(buffer);
			if (iter != token_id_table.end()) { return iter->second; }

			// Try to fall back to just the byte as a string
			const std::string fall_back_buffer { static_cast<char>(ch) };
			return token_id_table.at(fall_back_buffer);
		}

		uint8_t token_id_to_byte(spy::TokenID token_id) const override {
			const auto &token_data = token_data_table[token_id];
			uint8_t num = 0;
			std::from_chars(token_data.text.data() + 3, token_data.text.data() + 5, num, 16);
			return num;
		}

		std::string token_to_piece(spy::TokenID token_id) const override {
			if (0 <= token_id && token_id < num_vocab) {
				const TokenData token_data 		= token_data_table.at(token_id);
				const ModelTokenType token_type = token_data.type;

				switch (token_type) {
				case ModelTokenType::Normal: {
					return unescape_whitespace(token_data.text);
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
