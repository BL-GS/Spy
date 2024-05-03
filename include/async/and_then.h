/*
 * @author: BL-GS 
 * @date:   24-5-1
 */

#pragma once

#include <concepts>

#include "async/concept.h"
#include "async/task.h"

namespace spy {

	template <Awaitable A, Awaitable F>
	requires(!std::is_invocable_v<F> &&
	         !std::is_invocable_v<F, typename AwaitableTraits<A>::RetType>)
		Task<typename AwaitableTraits<F>::RetType> and_then(A a, F f) {
		co_await std::move(a);
		co_return co_await std::move(f);
	}

} // namespace spy