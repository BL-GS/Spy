#include <string_view>
#include <memory>

#include "loader/loader.h"
#include "loader/simple_loader.h"

namespace spy {

    std::unique_ptr<ModelLoader> ModelLoaderFactory::build_model_loader(std::string_view policy, std::string_view filename) {
        if (policy == "simple") {
            return std::make_unique<SimpleModeLoader>(filename);
        }
        spy_warn("unknown policy for model loader: {}, use default(simple) instead", policy);
        return std::make_unique<SimpleModeLoader>(filename);
    }

} // namespace spy