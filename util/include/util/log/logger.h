/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstring>
#include <cerrno>
#include <variant>
#include <vector>
#include <string>
#include <utility>
#include <source_location>
#include <spdlog/spdlog.h>

#include "util/log/exception.h"

namespace spy {

	struct LogParam {
	public:
		using Level = spdlog::level::level_enum;

	public:
		std::string log_level = "info";

	public:
		Level get_log_level() const {
			if (log_level == "trace") 	{ return Level::trace; 		}
			if (log_level == "debug") 	{ return Level::debug; 		}
			if (log_level == "info") 	{ return Level::info; 		}
			if (log_level == "warn") 	{ return Level::warn; 		}
			if (log_level == "error") 	{ return Level::err; 		}
			if (log_level == "fatal") 	{ return Level::critical; 	}
			if (log_level == "off")		{ return Level::off; 		}

			spdlog::warn("unknown log level: {}, use debug as default", log_level);
			return Level::debug;
		}
	};

	/*!
	 * @brief Init the global logger
	 * @details Log level will be cast to `spdlog::level::level_enum`
	 * - 0: trace
	 * - 1: debug
	 * - 2: info
	 * - 3: warn
	 * - 4: error
	 * - 5: fatal
	 * - 6: off
	 */
	inline void init_logger_format(const LogParam &context) {
		spdlog::set_level(context.get_log_level());
		spdlog::set_pattern("[%^%l%$: %t]: %v"); // level-tid-content
	}

	enum class DebugFlag: bool {
		Graph		= false,
		Execute		= true,
		Memory		= true
	};

	template<class T>
	inline void spy_debug(const T &msg) { spdlog::debug(msg); }

	template<class T>
	inline void spy_debug(DebugFlag option, const T &msg) { if (static_cast<bool>(option)) { spdlog::debug(msg); } }

	template<class ...Args>
	inline void spy_debug(spdlog::format_string_t<Args...> fmt, Args &&...args) { spdlog::debug(fmt, std::forward<Args>(args)...); }

	template<class ...Args>
	inline void spy_debug(DebugFlag option, spdlog::format_string_t<Args...> fmt, Args &&...args) { if (static_cast<bool>(option)) { spdlog::debug(fmt, std::forward<Args>(args)...); } }

	template<class T>
	inline void spy_info(const T &msg) { spdlog::info(msg); }

	template<class ...Args>
	inline void spy_info(spdlog::format_string_t<Args...> fmt, Args &&...args) { spdlog::info(fmt, std::forward<Args>(args)...); }

	template<class T>
	inline void spy_warn(const T &msg) { spdlog::warn(msg); }

	template<class ...Args>
	inline void spy_warn(spdlog::format_string_t<Args...> fmt, Args &&...args) { spdlog::warn(fmt, std::forward<Args>(args)...); }

	template<class T>
	inline void spy_error(const T &msg) { spdlog::error(msg); }

	template<class ...Args>
	inline void spy_error(spdlog::format_string_t<Args...> fmt, Args &&...args) { spdlog::error(fmt, std::forward<Args>(args)...); }

	template<class T>
	inline void spy_fatal(const T &msg) { 
		spdlog::critical(msg); 
		std::terminate();
	}

	template<class ...Args>
	inline void spy_fatal(spdlog::format_string_t<Args...> fmt, Args &&...args) { 
		spdlog::critical(fmt, std::forward<Args>(args)...); 
		std::terminate();
	}

	inline constexpr void spy_assert(bool expression, std::source_location loc = std::source_location::current()) { 
		if (!expression) {
			spdlog::critical("[Assert fault] {}:{}:{} Function: {}", loc.file_name(), loc.line(), loc.column(), loc.function_name());
			spdlog::critical("System Error: {}", system_error());
			std::terminate();			
		}
	}

	template<bool T_exception = false, class T>
	inline constexpr void spy_assert(bool expression, const T &msg, std::source_location loc = std::source_location::current()) { 
		if (!expression) {
			spdlog::critical("[Assert fault] {}:{}:{} Function: {}", loc.file_name(), loc.line(), loc.column(), loc.function_name());
			spdlog::critical(msg); 
			spdlog::critical("System Error: {}", system_error());
			if constexpr (T_exception) {
				throw SpyAssertException{msg};
			}
			std::terminate();			
		}
	}

	template<bool T_exception = false, class ...Args>
	inline constexpr void spy_assert(bool expression, spdlog::format_string_t<Args...> fmt, Args &&...args) { 
		if (!expression) {
			spdlog::critical("Assert fault");
			spdlog::critical(fmt, std::forward<Args>(args)...); 
			spdlog::critical("System Error: {}", system_error());
			if constexpr (T_exception) {
				throw SpyAssertException{fmt, std::forward<Args>(args)...};
			}
			std::terminate();			
		}
	}

	template<bool T_exception = false, class ...Args>
	inline constexpr void spy_abort(spdlog::format_string_t<Args...> fmt, Args &&...args) {
		spdlog::critical("Abort");
		spdlog::critical(fmt, std::forward<Args>(args)...); 
		spdlog::critical("System Error: {}", system_error());
		if constexpr (T_exception) {
			throw SpyAbortException{fmt, std::forward<Args>(args)...};
		}
		std::terminate();	
	}

#ifndef NDEBUG
	template<bool T_exception = false, class T>
	inline void spy_assert_debug(bool expression, const T &msg, std::source_location loc = std::source_location::current()) { 
		if (!expression) {
			spdlog::critical("[Assert fault] File: {} Line: {} Function: {}", loc.file_name(), loc.line(), loc.function_name());
			spdlog::critical(msg); 
			spdlog::critical("System Error: {}", system_error());
			if constexpr (T_exception) {
				throw SpyAssertException{msg};
			}
			std::terminate();			
		}
	}

	template<bool T_exception = false, class ...Args>
	inline void spy_assert_debug(bool expression, spdlog::format_string_t<Args...> fmt, Args &&...args) { 
		if (!expression) {
			spdlog::critical("Assert fault");
			spdlog::critical(fmt, std::forward<Args>(args)...); 
			spdlog::critical("System Error: {}", system_error());
			if constexpr (T_exception) {
				throw SpyAssertException{fmt, std::forward<Args>(args)...};
			}
			std::terminate();			
		}
	}
#else
	template<bool T_exception = false, class T>
	inline void spy_assert_debug([[maybe_unused]]bool expression, [[maybe_unused]]const T &msg, [[maybe_unused]]std::source_location loc = std::source_location::current()) { }

	template<bool T_exception = false, class ...Args>
	inline void spy_assert_debug([[maybe_unused]]bool expression, [[maybe_unused]]spdlog::format_string_t<Args...> fmt, [[maybe_unused]]Args &&...args) { }
#endif

	[[noreturn]] inline void spy_unreachable() {
#ifdef __GNUC__
		__builtin_unreachable();
#elif defined(_MSC_VER)
		__assume(false);
#endif
	}

};  // namespace spy


