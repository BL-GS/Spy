#pragma once

#include <memory>

#include "adapter/abstract_adapter.h"

namespace spy {

    class FileAdapterFactory {
    public:
        static std::unique_ptr<AbstractFileAdapter> build_file_adapter(std::string_view type);
    };

} // namespace spy