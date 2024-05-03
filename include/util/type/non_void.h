/*
 * @author: BL-GS 
 * @date:   24-5-1
 */

#pragma once

#include <utility>

namespace spy {

	template <class T = void>
	struct NonVoidHelper { using Type = T; };

	template <>
	struct NonVoidHelper<void> {
		using Type = NonVoidHelper;

		explicit NonVoidHelper() = default;

		template <class T>
		constexpr friend T &&operator,(T &&t, NonVoidHelper) { return std::forward<T>(t); }
	};

} // namespace spy