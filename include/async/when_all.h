/*
 * @author: BL-GS 
 * @date:   24-5-1
 */

#pragma once

#include <span>
#include <vector>
#include <coroutine>
#include <exception>

#include "util/type/uninitialized.h"
#include "async/concept.h"
#include "async/task.h"
#include "async/async_impl/return_previous.h"

namespace spy {

	struct WhenAllCtlBlock {
		/// The counter denote the number of tasks left
		std::size_t             counter;
		/// The previous handle of coroutine
		std::coroutine_handle<> previous_handle{};
		/// The pointer to the exception frame
		std::exception_ptr      exception_ptr{};
	};

	struct WhenAllAwaiter {
	public:
		/// Control metadata of coroutine
		WhenAllCtlBlock &                   control_block;
		/// View of all tasks to be executed
		std::span<const ReturnPreviousTask> tasks;

	public:
		bool await_ready() const noexcept { return false; }

		std::coroutine_handle<> await_suspend(std::coroutine_handle<> coroutine) const {
			// Return the handle directly if there is no task
			if (tasks.empty()) { return coroutine; }
			// Record the last handle in case of exception
			control_block.previous_handle = coroutine;
			// Execute all tasks
			for (const auto &task: tasks.subspan(0, tasks.size() - 1)) { task.handle.resume(); }
			return tasks.back().handle;
		}

		void await_resume() const {
			if (control_block.exception_ptr) [[unlikely]] {
				std::rethrow_exception(control_block.exception_ptr);
			}
		}
	};

	template <class T>
	ReturnPreviousTask when_all_helper(auto &&t, WhenAllCtlBlock &control, Uninitialized<T> &result) {
		try {
			result.put_value(co_await std::forward<decltype(t)>(t));
		} catch (...) {
			control.exception_ptr = std::current_exception();
			co_return control.previous_handle;
		}
		// Decrement counter if a coroutine has been completed
		--control.counter;
		// Return all results if all coroutines have finished
		if (control.counter == 0) {
			co_return control.previous_handle;
		}
		// Return nothing if there is any coroutine unfinished
		co_return std::noop_coroutine();
	}

	template <class = void>
	ReturnPreviousTask when_all_helper(auto &&t, WhenAllCtlBlock &control, Uninitialized<void> &result) {
		try {
			co_await std::forward<decltype(t)>(t);
		} catch (...) {
			control.exception_ptr = std::current_exception();
			co_return control.previous_handle;
		}
		// Decrement counter if a coroutine has been completed
		--control.counter;
		// Return all results if all coroutines have finished
		if (control.counter == 0) {
			co_return control.previous_handle;
		}
		// Return nothing if there is any coroutine unfinished
		co_return std::noop_coroutine();
	}

	template <std::size_t... Is, class... Ts>
	Task<std::tuple<typename AwaitableTraits<Ts>::NonVoidRetType...>>
	when_all_impl(std::index_sequence<Is...>, Ts &&...ts) {
		// Generate a counter with init value of tasks.size()
		// which will denote the end of coroutine if equals to 0
		WhenAllCtlBlock control{sizeof...(Ts)};
		// Return values via tuple
		std::tuple<Uninitialized<typename AwaitableTraits<Ts>::RetType>...> result;
		//
		ReturnPreviousTask taskArray[]{
				when_all_helper(ts, control, std::get<Is>(result))...};
		// Execute coroutine
		co_await WhenAllAwaiter(control, taskArray);
		// Return all results
		co_return std::tuple<typename AwaitableTraits<Ts>::NonVoidRetType...>(
				std::get<Is>(result).move_value()...);
	}

	/*!
	 * @brief  Execute all coroutine tasks at once
	 * @tparam T The type of task
	 * @param ts All tasks to be executed
	 * @return Coroutine handle after each execution and return.
	 */
	template <Awaitable... Ts>
		requires(sizeof...(Ts) != 0)
	auto when_all(Ts &&...ts) {
		return when_all_impl(
				std::make_index_sequence<sizeof...(Ts)>{},
				std::forward<Ts>(ts)...
		);
	}

	/*!
	 * @brief Execute all coroutine tasks at once
	 * @tparam T The type of task
	 * @tparam Alloc The allocator for task array
	 * @param tasks The vector contains all tasks to be executed
	 * @return Coroutine handle after each execution and return.
	 */
	template <Awaitable T, class Alloc = std::allocator<T>>
	Task<std::conditional_t<
			std::is_void_v<typename AwaitableTraits<T>::RetType>,
			std::vector<typename AwaitableTraits<T>::RetType, Alloc>, void>>
	when_all(std::vector<T, Alloc> const &tasks) {
		// Generate a counter with init value of tasks.size()
		// which will denote the end of coroutine if equals to 0
		WhenAllCtlBlock control{tasks.size()};
		// Initialize allocator for tasks
		Alloc alloc = tasks.get_allocator();
		std::vector<Uninitialized<typename AwaitableTraits<T>::RetType>, Alloc>
																		 result_array(tasks.size(), alloc);
		// Execution
		{
			// Generate the array of all tasks
			std::vector<ReturnPreviousTask, Alloc> task_array(alloc);
			task_array.reserve(tasks.size());
			for (std::size_t i = 0; i < tasks.size(); ++i) {
				task_array.push_back(when_all_helper(tasks[i], control, result_array[i]));
			}
			// Start coroutine
			co_await WhenAllAwaiter(control, task_array);
		}
		// Collect result if return non-void values
		if constexpr (!std::is_void_v<typename AwaitableTraits<T>::RetType>) {
			std::vector<typename AwaitableTraits<T>::RetType, Alloc> res(alloc);
			res.reserve(tasks.size());
			for (auto &result: result_array) {
				res.push_back(result.move_value());
			}
			co_return res;
		}
	}

} // namespace spy