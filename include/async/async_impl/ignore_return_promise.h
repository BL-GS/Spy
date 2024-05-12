/*
 * @author: BL-GS 
 * @date:   24-5-3
 */

#pragma once

#if defined(__unix__) && __has_include(<cxxabi.h>)
#include <cxxabi.h>
#endif

#include <coroutine>
#include <exception>

#include "util/logger.h"

namespace spy {

	template <class FinalAwaiter = std::suspend_always>
	struct IgnoreReturnPromise {
	public:
		IgnoreReturnPromise &operator=(IgnoreReturnPromise &&) = delete;

	public:
		auto initial_suspend() noexcept     { return std::suspend_always(); }

		auto final_suspend() noexcept       { return FinalAwaiter(); }

		void unhandled_exception() noexcept {
			try {
				throw;
			} catch (std::exception const &e) {
				auto name = typeid(e).name();
				spy_error("co_spawn coroutine terminated after thrown exception '{}': {}", name, e.what());
			} catch (...) {
				spy_error("co_spawn coroutine terminated after thrown exception\n");
			}
		}

		void return_void() noexcept {}

		auto get_return_object() { return std::coroutine_handle<IgnoreReturnPromise>::from_promise(*this); }

	public:
		void set_previous(std::coroutine_handle<>) noexcept {}
	};

	struct AutoDestroyFinalAwaiter {
		bool await_ready() const noexcept { return false; }
		void await_suspend(std::coroutine_handle<> coroutine) const noexcept { coroutine.destroy(); }
		void await_resume() const noexcept {}
	};

} // namespace spy
