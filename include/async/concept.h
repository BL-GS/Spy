/*
 * @author: BL-GS 
 * @date:   24-5-1
 */

#pragma once

#include <coroutine>
#include <utility>

#include "util/type/non_void.h"

namespace spy {

	template <class T>
	concept Awaiter = requires(T awaiter, std::coroutine_handle<> h) {
		{ awaiter.await_ready()     };
		{ awaiter.await_suspend(h)  };
		{ awaiter.await_resume()    };
	};

	template<class T>
	concept Awaitable = Awaiter<T> || requires(T awaiter) {
		{ awaiter.operator co_await() } -> Awaiter;
	};

	template<class T>
	struct AwaitableTraits { using Type = T; };

	template<Awaiter T>
	struct AwaitableTraits<T> {
		using ReturnType            = decltype(std::declval<T>().await_resume());
		using NonVoidReturnType     = NonVoidHelper<ReturnType>::Type;
		using Type                  = ReturnType;
		using AwaiterType           = T;
	};

	template <class A>
	requires(!Awaiter<A> && Awaitable<A>)
	struct AwaitableTraits<A>: AwaitableTraits<decltype(std::declval<A>().operator co_await())> {};

} // namespace spy