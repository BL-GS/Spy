#pragma once

#include <memory>
#include <magic_enum.hpp>

#include "model/model_impl/abstract_model.h"
#include "model/model_impl/llama.h"

namespace spy {

    struct ModelBuilder {
    public:
        static std::unique_ptr<AbstractModel> build_model(const ModelType model_type, 
                std::unique_ptr<GGUFContext> &&context_ptr, const HyperParam &hyper_param) {
            std::unique_ptr<AbstractModel> model_ptr;
            switch (model_type) {
            case ModelType::LLaMa:
                model_ptr = std::make_unique<LLAMAModel>(std::move(context_ptr), hyper_param);
                break;

            default:
                SPY_ASSERT_FMT(false, "Unsupported model type: {}", magic_enum::enum_name(model_type));
            }

            model_ptr->init();
            return model_ptr;
        }
    };

} // namespace spy