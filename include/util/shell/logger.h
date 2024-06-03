/*
 * @author: BL-GS 
 * @date:   2024/3/22
 */

#pragma once

#include <cstring>
#include <cerrno>
#include <variant>
#include <source_location>
#include <spdlog/spdlog.h>
#include <utility>

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

#ifdef NDEBUG
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


	class SpyProperty {
	public:
		struct PropertyElement {
			std::string key;
			std::variant<char, int32_t, uint32_t, int64_t, uint64_t,
				float_t, double_t, std::string, void *, std::vector<PropertyElement>
			> value;

			PropertyElement() = default;

			template<class T>
			PropertyElement(std::string &&key, const T &value): key(std::forward<std::string>(key)), value(value) {}
		};

	private:
		std::string                  property_name_;
		std::vector<PropertyElement> property_array_;

	public:
		SpyProperty(std::string &&name): property_name_(std::forward<std::string>(name)) {}

	public:
		template<class T>
		void add(std::string &&key, const T &value) {
			property_array_.emplace_back(std::forward<std::string>(key), value);
		}

	public:
		std::string to_string() const { return property_to_string(property_array_); }

	private:
		static std::string property_to_string(const std::vector<PropertyElement> &property_array) {
			std::string res;
			for (const PropertyElement &prop: property_array) {
				switch (prop.value.index()) {
					case 0: res += fmt::format("{:32}: {}\n", prop.key, std::get<0>(prop.value));
						break;
					case 1: res += fmt::format("{:32}: {}\n", prop.key, std::get<1>(prop.value));
						break;
					case 2: res += fmt::format("{:32}: {}\n", prop.key, std::get<2>(prop.value));
						break;
					case 3: res += fmt::format("{:32}: {}\n", prop.key, std::get<3>(prop.value));
						break;
					case 4: res += fmt::format("{:32}: {}\n", prop.key, std::get<4>(prop.value));
						break;
					case 5: res += fmt::format("{:32}: {}\n", prop.key, std::get<5>(prop.value));
						break;
					case 6: res += fmt::format("{:32}: {}\n", prop.key, std::get<6>(prop.value));
						break;
					case 7: res += fmt::format("{:32}: {}\n", prop.key, std::get<7>(prop.value));
						break;
					case 8: res += fmt::format("{:32}: {}\n", prop.key, std::get<8>(prop.value));
						break;
					case 9: {
						std::string sub_prop = property_to_string(std::get<9>(prop.value));
						for (size_t pos = 0; pos < sub_prop.size(); pos = sub_prop.find('\n', pos + 5)) {
							sub_prop.insert(pos, "---- ");
						}
						res += fmt::format("{:32}: \n", prop.key, sub_prop);
					} break;
					default: spy_unreachable();
				}
			}
			return res;
		}
	};

};  // namespace spy


