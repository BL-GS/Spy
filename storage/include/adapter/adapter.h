#pragma once

#include <memory>

#include "adapter/abstract_adapter.h"

namespace spy {

    class FileAdapterFactory {
    public:
        /*!
         * @brief Build up a derived adapter for different file recogonization
         * @param type the type name of the adapter (e.g. "gguf")
         */
        static std::unique_ptr<AbstractFileAdapter> build_file_adapter(std::string_view type);
    };

} // namespace spy