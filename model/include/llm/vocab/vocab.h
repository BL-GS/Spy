/*
 * @author: BL-GS 
 * @date:   2024/4/2
 */

#pragma once

#include <vector>
#include <magic_enum.hpp>

#include "llm/vocab/type.h"
#include "llm/vocab/abstract_vocab.h"

namespace spy {

    struct ModelMetaContext;

    class Tokenizer {

    private:
        std::unique_ptr<Vocab> vocab_;

    public:
        Tokenizer(ModelVocabType vocab_type, const ModelMetaContext &context) {
            init_vocab(vocab_type, context);
        }

    private:
        void init_vocab(ModelVocabType vocab_type, const ModelMetaContext &context);
        
    public:
        std::vector<TokenID> tokenize(const std::string &text, bool add_bos, bool special) {
            return vocab_->tokenize(text, add_bos, special);
        }

        std::string token_to_piece(TokenID token_id) {
            return vocab_->token_to_piece(token_id);
        }

    public:
        const std::unique_ptr<Vocab> &get_vocab() const { return vocab_; }

		TokenID get_special_bos_id() const { return vocab_->special_bos_id; }
		TokenID get_special_eos_id() const { return vocab_->special_eos_id; }
		TokenID get_special_unk_id() const { return vocab_->special_unk_id; }
		TokenID get_special_sep_id() const { return vocab_->special_sep_id; }
		TokenID get_special_pad_id() const { return vocab_->special_pad_id; }
    };

} // namespace spy 