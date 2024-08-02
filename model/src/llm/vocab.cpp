#include "llm/vocab/byte_pair_encoding_vocab.h"
#include "llm/vocab/sentence_piece_vocab.h"
#include "llm/vocab/word_piece_vocab.h"
#include "llm/vocab/vocab.h"
#include "adapter/type.h"

namespace spy {

    void Tokenizer::init_vocab(ModelVocabType vocab_type, const ModelMetaContext &context) {
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
            spy_abort("Unknown vocab type: {}", vocab_type);
        }
    }

} // namespace spy