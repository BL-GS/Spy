#include <memory>
#include <string_view>

#include "adapter/abstract_adapter.h"
#include "adapter/gguf_adaper.h"
#include "adapter/adapter.h"

namespace spy {

    using FileAdapterPointer = std::unique_ptr<AbstractFileAdapter>;

    FileAdapterPointer FileAdapterFactory::build_file_adapter(std::string_view type) {
        if (type == "gguf") {
            return std::make_unique<GGUFAdapter>();
        }
        return nullptr;
    }

} // namespace

