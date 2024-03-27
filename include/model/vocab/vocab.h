/*
 * @author: BL-GS 
 * @date:   2024/4/2
 */

#pragma once

#include <cstdint>
#include <memory>
#include <magic_enum.hpp>

#include "util/logger.h"
#include "model/vocab/type.h"
#include "model/vocab/abstract_vocab.h"
#include "model/vocab/sentence_piece_vocab.h"
#include "model/vocab/byte_pair_encoding_vocab.h"
#include "model/vocab/word_piece_vocab.h"

namespace spy {

    class Tokenizer {

    private:
        std::unique_ptr<AbstractVocab> vocab_;

    public:
        Tokenizer(ModelVocabType vocab_type, const GGUFContext &context) {
            switch (vocab_type) {
            case ModelVocabType::SentencePiece:
                vocab_ = std::make_unique<SentencePieceVocab>(context);
                break;
            case ModelVocabType::BytePairEncoding:
                vocab_ = std::make_unique<BytePairEncodingVocab>(context);
                break;
            case ModelVocabType::WordPiece:
                vocab_ = std::make_unique<WordPieceVocab>(context);
                break;
            default:
                SPY_ASSERT_FMT(false, "Unknown vocab type: {}", magic_enum::enum_name(vocab_type));
            }
        }
        
    public:
        std::vector<TokenID> tokenize(const std::string &text, bool add_bos, bool special) {
            return vocab_->tokenize(text, add_bos, special);
        }

        std::string token_to_piece(TokenID token_id) {
            return vocab_->token_to_piece(token_id);
        }

    public:
        const std::unique_ptr<AbstractVocab> &get_vocab() const { return vocab_; }

		TokenID get_special_bos_id() const { return vocab_->special_bos_id; }
		TokenID get_special_eos_id() const { return vocab_->special_eos_id; }
		TokenID get_special_unk_id() const { return vocab_->special_unk_id; }
		TokenID get_special_sep_id() const { return vocab_->special_sep_id; }
		TokenID get_special_pad_id() const { return vocab_->special_pad_id; }
    };

} // namespace spy 