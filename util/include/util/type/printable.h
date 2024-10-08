/*
 * @author: BL-GS 
 * @date:   24-6-3
 */

#pragma once

#include <string>
#include <string_view>
#include <map>
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

namespace spy {

	struct PrintableInterface {
		virtual std::string to_string() const = 0;
	};

	struct PropertyInterface: PrintableInterface {
		virtual std::map<std::string_view, std::string> property() const = 0;

		std::string to_string() const override {
			std::string str;
			const auto prop = property();
			for (auto [key, value]: prop) {
				str += fmt::format("{:8}: {}\n", key, value);
			}
			return str;
		}
	};

} // namespace