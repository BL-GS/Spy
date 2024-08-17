#pragma once

#include <string>

#include "adapter/type.h"

namespace spy {

    class FileAdapter {
    public:
        FileAdapter() = default;

        virtual ~FileAdapter() = default;

    public:
        /*!
         * @brief Load metadata of the model file
         * @param filename the path to the file
         * @return The metadata of the model file
         */
        virtual ModelMetaContext init_from_file(std::string_view filename) = 0;
    };

} // namespace spy