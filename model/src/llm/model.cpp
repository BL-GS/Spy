#include <memory>

#include "util/shell/logger.h"
#include "llm/model/llama.h"
#include "llm/model/model.h"

namespace spy {

    std::unique_ptr<AbstractModel> ModelBuilder::build_model(const std::string_view model_type, 
            ModelMetaContext &&context_ptr, const HyperParam &hyper_param) {
        std::unique_ptr<AbstractModel> model_ptr;

        if (model_type == "llama") {
            model_ptr = std::make_unique<LLAMAModel>(std::forward<ModelMetaContext>(context_ptr), hyper_param);
        } else {
            spy_assert(false, "Unsupported model type: {}", model_type);
        }

        model_ptr->init();
        return model_ptr;
    }

} // namespace spy