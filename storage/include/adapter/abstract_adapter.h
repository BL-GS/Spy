#pragma once

#include <string>

#include "adapter/type.h"

namespace spy {

    class AbstractFileAdapter {
    public:
        std::string      filename;
        ModelMetaContext context;

    public:
        AbstractFileAdapter() = default;

        virtual ~AbstractFileAdapter() = default;

    public:
        virtual void init_from_file(const std::string_view filename) = 0;
    };

} // namespace spy