#pragma once

#include <memory>
#include <magic_enum.hpp>

#include "llm/model/abstract_model.h"

namespace spy {

    struct ModelBuilder {
    public:
        static std::unique_ptr<AbstractModel> build_model(const ModelType model_type, std::unique_ptr<ModelMetaContext> &&context_ptr, const HyperParam &hyper_param);
    };

} // namespace spy