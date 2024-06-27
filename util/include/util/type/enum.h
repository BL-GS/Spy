#pragma once

/**
 * @brief I make some simple utilities because nvcc isn't compatible to magic-enum
 * @ref [magic-enum](https://github.com/Neargye/magic_enum)
 */

#include <utility>
#include <string_view>
#include <magic_enum.hpp>
#include <fmt/core.h>
#include <fmt/format.h>

namespace spy {

    template<class T_Enum, T_Enum T_end = static_cast<T_Enum>(256), T_Enum T_begin = static_cast<T_Enum>(0)>
		requires std::is_enum_v<T_Enum>
	struct EnumMapper {
		template<class T_Func>
		static constexpr auto map(T_Func &&func, T_Enum value) { 
			constexpr auto cur_enum 			= std::integral_constant<T_Enum, T_begin>{};
			constexpr auto cur_enum_integer 	= static_cast<std::underlying_type_t<T_Enum>>(T_begin);

			if constexpr (T_begin == T_end) {
				return std::forward<T_Func>(func)(cur_enum);
			} else {
				if (value == cur_enum) { return std::forward<T_Func>(func)(cur_enum); }
				return EnumMapper<T_Enum, T_end, static_cast<T_Enum>(cur_enum_integer + 1)>::map(std::forward<T_Func>(func), value);
			}
		}

		template<class T_Func, class T_Err_Func>
		static constexpr auto map(T_Func &&func, T_Err_Func &&err_func, T_Enum value) { 
			if constexpr (T_begin > T_end) {
				return std::forward<T_Err_Func>(err_func)(value);
			} else {
                constexpr auto cur_enum 			= std::integral_constant<T_Enum, T_begin>{};
			    constexpr auto cur_enum_integer 	= static_cast<std::underlying_type_t<T_Enum>>(T_begin);

				if (value == cur_enum) { return std::forward<T_Func>(func)(cur_enum); }
				return EnumMapper<T_Enum, T_end, static_cast<T_Enum>(cur_enum_integer + 1)>::map(std::forward<T_Func>(func), value);
			}
		}

		template<class T_Func, T_Enum ...T_temp>
		static constexpr auto product_map_inner(T_Func &&func, T_Enum value) { 
			constexpr auto cur_enum 			= std::integral_constant<T_Enum, T_begin>{};
			constexpr auto cur_enum_integer 	= static_cast<std::underlying_type_t<T_Enum>>(T_begin);

			if constexpr (T_begin == T_end) {
				return std::forward<T_Func>(func)(std::integral_constant<T_Enum, T_temp>{}..., cur_enum);
			} else {
				if (value == cur_enum) { return std::forward<T_Func>(func)(std::integral_constant<T_Enum, T_temp>{}..., cur_enum); }
				return EnumMapper<T_Enum, T_end, static_cast<T_Enum>(cur_enum_integer + 1)>::template product_map_inner<T_Func, T_temp...>(std::forward<T_Func>(func), value);
			}
		}

		template<class T_Func, T_Enum ...T_temp, class ...T_Left>
		static constexpr auto product_map_inner(T_Func &&func, T_Enum value, T_Left ...left_value) { 
			constexpr auto cur_enum 			= std::integral_constant<T_Enum, T_begin>{};
			constexpr auto cur_enum_integer 	= static_cast<std::underlying_type_t<T_Enum>>(T_begin);

			if constexpr (T_begin == T_end) {
                return EnumMapper<T_Enum, T_end, static_cast<T_Enum>(0)>::template product_map_inner<T_Func,  T_temp..., T_begin>(std::forward<T_Func>(func), (left_value)...);
			} else {
				if (value == cur_enum) { return EnumMapper<T_Enum, T_end, static_cast<T_Enum>(0)>::template product_map_inner<T_Func,  T_temp..., T_begin>(std::forward<T_Func>(func), (left_value)...); }
				return EnumMapper<T_Enum, T_end, static_cast<T_Enum>(cur_enum_integer + 1)>::template product_map_inner<T_Func, T_temp...>(std::forward<T_Func>(func), value, (left_value)...);
			}
		}

		template<class T_Func, class ...T_Left>
		static constexpr auto product_map(T_Func &&func, T_Enum value, T_Left ...left_value) { 
			constexpr auto cur_enum 			= std::integral_constant<T_Enum, T_begin>{};
			constexpr auto cur_enum_integer 	= static_cast<std::underlying_type_t<T_Enum>>(T_begin);

			if constexpr (T_begin == T_end) {
                return EnumMapper<T_Enum, T_end, static_cast<T_Enum>(0)>::template product_map_inner<T_Func, T_begin>(std::forward<T_Func>(func), (left_value)...);
			} else {
				if (value == cur_enum) { return EnumMapper<T_Enum, T_end, static_cast<T_Enum>(0)>::template product_map_inner<T_Func, T_begin>(std::forward<T_Func>(func), (left_value)...); }
				return EnumMapper<T_Enum, T_end, static_cast<T_Enum>(cur_enum_integer + 1)>::template product_map<T_Func>(std::forward<T_Func>(func), value, (left_value)...);
			}
		}
	};

#ifndef SPY_ENUM_FORMATTER
#define SPY_ENUM_FORMATTER(EnumType)                                            \
	template <>                                                                 \
	struct fmt::formatter<EnumType>: fmt::formatter<std::string_view> {         \
		auto format(EnumType e, fmt::format_context& ctx) const {               \
            std::string_view name = magic_enum::enum_name(e);                   \
			return fmt::formatter<string_view>::format(name, ctx);              \
		}                                                                       \
	}
#endif // SPY_ENUM_FORMATTER
}