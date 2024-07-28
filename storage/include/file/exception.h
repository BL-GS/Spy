#pragma once

#include "util/exception.h"

namespace spy {

    class SpyOSFileException final: public SpyOSException {
	public:
		static constexpr std::string_view PREFIX			= "SpyFileException: ";
		static constexpr std::string_view UNKNOWN_EXCEPTION = "SpyFileException: Unknown exception";

	public:
		SpyOSFileException(): SpyOSException(UNKNOWN_EXCEPTION.data()) {}

		SpyOSFileException(const std::string_view &reason): SpyOSException(PREFIX.data() + std::string(reason)) {}

		template<class ...Args>
		SpyOSFileException(const std::string_view &reason, Args &&...args): 
			SpyOSException(PREFIX.data() + std::string(reason), std::forward<Args>(args)...) {}
    };

} // namespace spy