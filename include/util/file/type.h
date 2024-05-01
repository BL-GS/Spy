#pragma once

#include "util/exception.h"

namespace spy {

    class SpyOSFileException: public SpyOSException {
    public:
        SpyOSFileException(): SpyOSException("undefined file operation") {}
        SpyOSFileException(const std::string &reason): SpyOSException(reason) {}
    };

} // namespace spy