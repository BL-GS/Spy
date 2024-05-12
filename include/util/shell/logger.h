/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <iostream>
#include <cstring>
#include <cerrno>
#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/format.h>
#include <source_location>
#include <spdlog/spdlog.h>

#include "util/exception.h"

namespace spy {

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
	inline void init_logger_format(int log_level) {
		spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
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

	inline void spy_assert(bool expression, std::source_location loc = std::source_location::current()) { 
		if (!expression) {
			spdlog::critical("[Assert fault] File: {} Line: {} Function: {}", loc.file_name(), loc.line(), loc.function_name());
			spdlog::critical("System Error: {}", system_error());
			std::terminate();			
		}
	}

	template<class T>
	inline void spy_assert(bool expression, const T &msg, std::source_location loc = std::source_location::current()) { 
		if (!expression) {
			spdlog::critical("[Assert fault] File: {} Line: {} Function: {}", loc.file_name(), loc.line(), loc.function_name());
			spdlog::critical(msg); 
			spdlog::critical("System Error: {}", system_error());
			std::terminate();			
		}
	}

	template<class ...Args>
	inline void spy_assert(bool expression, spdlog::format_string_t<Args...> fmt, Args &&...args) { 
		if (!expression) {
			spdlog::critical("Assert fault");
			spdlog::critical(fmt, std::forward<Args>(args)...); 
			spdlog::critical("System Error: {}", system_error());
			std::terminate();			
		}
	}

}  // namespace spy
