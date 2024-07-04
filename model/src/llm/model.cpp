#include <memory>

#include "util/shell/logger.h"
#include "llm/model_impl/llama.h"
#include "llm/model_impl/model.h"

namespace spy {

    std::unique_ptr<AbstractModel> ModelBuilder::build_model(const ModelType model_type, 
            std::unique_ptr<ModelMetaContext> &&context_ptr, const HyperParam &hyper_param) {
        std::unique_ptr<AbstractModel> model_ptr;
        switch (model_type) {
        case ModelType::LLaMa:
            model_ptr = std::make_unique<LLAMAModel>(std::move(context_ptr), hyper_param);
            break;

        default:
            spy_assert(false, "Unsupported model type: {}", model_type);
        }

        model_ptr->init();
        return model_ptr;
    }

} // namespace spy