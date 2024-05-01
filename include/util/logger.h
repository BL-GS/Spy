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

#include "util/exception.h"

namespace spy {

	enum class DebugFlag: bool {
		Graph		= false,
		Execute		= true,
		Memory		= true
	};

	#define SPY_INFO(output_fmt) \
		do { fmt::print("[INFO]: " output_fmt "\n"); } while (0)

	#define SPY_INFO_FMT(output_fmt, ...) \
		do { fmt::print("[INFO]: " output_fmt "\n", __VA_ARGS__); } while (0)

	#define SPY_WARN(output_fmt) \
		do { fmt::print(fg(fmt::color::yellow), "[WARN]: " output_fmt "\n"); } while (0)

	#define SPY_WARN_FMT(output_fmt, ...) \
		do { fmt::print(fg(fmt::color::yellow), "[WARN]: " output_fmt "\n", __VA_ARGS__); } while (0)

	#define SPY_ERROR(output_fmt) \
		do { fmt::print(fg(fmt::color::red), 	"[ERROR]: " output_fmt "\n"); } while (0)

	#define SPY_ERROR_FMT(output_fmt, ...) \
		do { fmt::print(fg(fmt::color::red), 	"[ERROR]: " output_fmt "\n", __VA_ARGS__); } while (0)

	#define SPY_FATAL(output_fmt) \
		do { fmt::print(fg(fmt::color::red), 	"[FATAL]: " output_fmt "\n"); throw SpyAssertException(output_fmt); } while (0)

	#define SPY_FATAL_FMT(output_fmt, ...) \
		do { fmt::print(fg(fmt::color::red), 	"[FATAL]: " output_fmt "\n", __VA_ARGS__); throw SpyAssertException(output_fmt); } while (0)


	#define SPY_ASSERT(expression, ...)     	\
		do {                            		\
			if (!(expression)) {        		\
				fmt::print(fg(fmt::color::red), "Assert fault[{}:{}]: " #expression "\n", __FILE__, __LINE__); 	\
				fmt::print(fg(fmt::color::red), "System: {}", system_error());									\
				SPY_FATAL(__VA_ARGS__ "");     	\
			}                           		\
		} while (0)

	#define SPY_ASSERT_FMT(expression, output_fmt, ...)    \
		do {                                    \
			if (!(expression)) {                \
				fmt::print(fg(fmt::color::red), "Assert fault[{}:{}]: " #expression "\n", __FILE__, __LINE__);  \
				fmt::print(fg(fmt::color::red), "System: {}", system_error());									\
				SPY_FATAL_FMT(output_fmt, __VA_ARGS__);    \
			}                                   		   \
		} while (0)


	#define SPY_ASSERT_NOEXCEPTION(expression, ...)     \
		do {                            \
			if (!(expression)) {        \
				fmt::print(fg(fmt::color::red), "Assert fault[{}:{}]: " #expression "\n", __FILE__, __LINE__); 	\
				fmt::print(fg(fmt::color::red), "System: {}", system_error());									\
				fmt::print(fg(fmt::color::red), "[FATAL]: " __VA_ARGS__ "\n");     								\
			}                           																		\
		} while (0)

	#define SPY_ASSERT_FMT_NOEXCEPTION(expression, output_fmt, ...)    \
		do {                                    \
			if (!(expression)) {                \
				fmt::print(fg(fmt::color::red), "Assert fault[{}:{}]: " #expression "\n", __FILE__, __LINE__);  \
				fmt::print(fg(fmt::color::red), "System: {}", system_error());									\
				fmt::print(fg(fmt::color::red), "[FATAL]: " output_fmt "\n", __VA_ARGS__);     					\
			}                                   		   														\
		} while (0)


#ifdef NDEBUG
	#define SPY_DEBUG(output_fmt)

	#define SPY_DEBUG_FMT(output_fmt, ...)

	#define SPY_DEBUG_OPTION(flag, output_fmt)

	#define SPY_DEBUG_FMT_OPTION(flag, output_fmt, ...)

	#define SPY_ASSERT_DEBUG(expression, ...) (expression)

	#define SPY_ASSERT_FMT_DEBUG(expression, output_fmt, ...) (expression)

#else 
	#define SPY_DEBUG(output_fmt) \
		do { fmt::print(fg(fmt::color::light_green), 	"[DEBUG]: " output_fmt "\n"); std::fflush(stdout); } while (0)

	#define SPY_DEBUG_FMT(output_fmt, ...) \
		do { fmt::print(fg(fmt::color::light_green), 	"[DEBUG]: " output_fmt "\n", __VA_ARGS__); std::fflush(stdout); } while (0)

	#define SPY_DEBUG_OPTION(flag, output_fmt) \
		do { if (static_cast<bool>(DebugFlag:: flag)) { SPY_DEBUG(output_fmt); } } while (0)

	#define SPY_DEBUG_FMT_OPTION(flag, output_fmt, ...) \
		do { if (static_cast<bool>(DebugFlag:: flag)) { SPY_DEBUG_FMT(output_fmt, __VA_ARGS__); } } while (0)


	#define SPY_ASSERT_DEBUG(expression, ...)                                                                \
		do {                                                                                                 \
			if (!(expression)) {                                                                             \
				SPY_ERROR_FMT("Assert fault[{}:{}]: " #expression "\n", __FILE__, __LINE__); 			 	 \
				SPY_FATAL(__VA_ARGS__);                                                                      \
			}                                                                                                \
		} while (0)

	#define SPY_ASSERT_FMT_DEBUG(expression, output_fmt, ...)                                                \
		do {                                                                                                 \
			if (!(expression)) {                                                                             \
				SPY_ERROR_FMT("Assert fault[{}:{}]: " #expression "\n", __FILE__, __LINE__); 		     	 \
				SPY_FATAL_FMT(output_fmt, __VA_ARGS__);                                                      \
			}                                                                                                \
		} while (0)

#endif

}  // namespace spy
