#pragma once

#include <memory>
#include <filesystem>

#include "adapter/abstract_adapter.h"

namespace spy {

    class FileAdapterFactory {
    public:
        /*!
         * @brief Build up a derived adapter for different file recogonization
         * @param[in] type the type name of the adapter (e.g. "gguf")
         */
        static std::unique_ptr<FileAdapter> build_file_adapter(std::string_view type);

        /*!
         * @brief Automatically recognize the type of model and build up a corresponding adapter.
         * @param[in] path the location of the model
         */
        static std::unique_ptr<FileAdapter> auto_build_file_adapter(std::filesystem::path path);
    };

} // namespace spy