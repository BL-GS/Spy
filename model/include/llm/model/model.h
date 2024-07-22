#pragma once

#include <memory>
#include <magic_enum.hpp>

#include "llm/model/abstract_model.h"

namespace spy {

    struct ModelBuilder {
    public:
        static std::unique_ptr<AbstractModel> build_model(std::string_view model_type, 
            ModelMetaContext &&context, const HyperParam &hyper_param);
    };

} // namespace spy