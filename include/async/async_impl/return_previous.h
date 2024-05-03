/*
 * @author: BL-GS 
 * @date:   24-5-1
 */

#pragma once

#include <coroutine>

#include "async/async_impl/previous_awaiter.h"

namespace spy {

	struct ReturnPreviousPromise {
	public:
		std::coroutine_handle<> previous_handle;

	public:
		ReturnPreviousPromise &operator=(ReturnPreviousPromise &&) = delete;

	public:
		auto initial_suspend() noexcept {
			return std::suspend_always();
		}

		auto final_suspend() noexcept {
			return PreviousAwaiter(previous_handle);
		}

		void unhandled_exception() {
			throw;
		}

		void return_value(std::coroutine_handle<> previous) noexcept {
			previous_handle = previous;
		}

		auto get_return_object() {
			return std::coroutine_handle<ReturnPreviousPromise>::from_promise(*this);
		}
	};

	struct [[nodiscard]] ReturnPreviousTask {
	public:
		using promise_type = ReturnPreviousPromise;

	public:
		std::coroutine_handle<promise_type> handle;

	public:
		ReturnPreviousTask(std::coroutine_handle<promise_type> coroutine) noexcept : handle(coroutine) {}

		ReturnPreviousTask(ReturnPreviousTask &&) = delete;

		~ReturnPreviousTask() { handle.destroy(); }
	};

}