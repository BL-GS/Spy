/*
 * @author: BL-GS 
 * @date:   24-5-1
 */

#pragma once

#include <type_traits>
#include <functional>

#include "async/task.h"

namespace spy {

	inline Task<> just_void() {
		co_return;
	}

	template <class T>
	Task<T> just_value(T t) {
		co_return std::move(t);
	}

	template <class F, class... Args>
	Task<std::invoke_result_t<F, Args...>> just_invoke(F &&f, Args &&...args) {
		co_return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
	}

} // namespace spy