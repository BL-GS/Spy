/*
 * @author: BL-GS 
 * @date:   24-5-1
 */

#pragma once

#include <coroutine>

namespace spy {

	struct PreviousAwaiter {
	public:
		std::coroutine_handle<> previous_handle;

	public:
		bool await_ready() const noexcept {
			return false;
		}

		std::coroutine_handle<>
		await_suspend(std::coroutine_handle<> coroutine) const noexcept {
			return previous_handle;
		}

		void await_resume() const noexcept {}
	};

} // namespace spy