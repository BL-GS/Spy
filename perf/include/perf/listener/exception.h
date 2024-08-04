#pragma once

#include "util/log/exception.h"

namespace spy::perf {

    class SpyPerfException: public SpyException {
    public:
		static constexpr std::string_view PREFIX			= "SpyPerfException: ";
		static constexpr std::string_view UNKNOWN_EXCEPTION = "SpyPerfException: Unknown exception";

	public:
		SpyPerfException(): SpyException(UNKNOWN_EXCEPTION.data()) {}

		SpyPerfException(const std::string_view &reason): SpyException(PREFIX.data() + std::string(reason)) {}

		template<class ...Args>
		SpyPerfException(const std::string_view &reason, Args &&...args): 
			SpyException(PREFIX.data() + std::string(reason), std::forward<Args>(args)...) {}
    };

} // namespace spy::perf