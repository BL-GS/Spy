#include <memory>

#include "util/log/logger.h"
#include "llm/model/llama.h"
#include "llm/model/model.h"

namespace spy {

    std::unique_ptr<Model> ModelBuilder::build_model(ModelMetaContext &context, const HyperParam &hyper_param) {
        std::unique_ptr<Model> model_ptr;
        const std::string_view model_type = context.arch_name;

        if (model_type == "llama") {
            model_ptr = std::make_unique<LLAMAModel>(hyper_param);
        } else {
            spy_abort("Unsupported model type: {}", model_type);
        }

        model_ptr->init(context);
        return model_ptr;
    }

} // namespace spy