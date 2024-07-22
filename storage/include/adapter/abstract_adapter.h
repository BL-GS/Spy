#pragma once

#include <string>

#include "adapter/type.h"

namespace spy {

    class AbstractFileAdapter {
    public:
        AbstractFileAdapter() = default;

        virtual ~AbstractFileAdapter() = default;

    public:
        /*!
         * @brief Load metadata of the model file
         * @param filename the path to the file
         * @return The metadata of the model file
         */
        virtual ModelMetaContext init_from_file(const std::string_view filename) = 0;
    };

} // namespace spy