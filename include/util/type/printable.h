/*
 * @author: BL-GS 
 * @date:   24-6-3
 */

#pragma once

#include <string>
#include <fmt/core.h>
#include <fmt/format.h>

#ifndef SPY_PRINTABLE_FORMATTER
#define SPY_PRINTABLE_FORMATTER(ClassType)                                      \
	template <>                                                                 \
	struct fmt::formatter<ClassType>: fmt::formatter<std::string> {             \
		auto format(const ClassType &c, fmt::format_context& ctx) const {       \
            std::string str = c.to_string();                                    \
			return fmt::formatter<std::string>::format(str, ctx);               \
		}                                                                       \
	}
#endif // SPY_PRINTABLE_FORMATTER