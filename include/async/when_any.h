/*
 * @author: BL-GS 
 * @date:   24-5-1
 */

#pragma once

#include <vector>
#include <variant>
#include <span>
#include <coroutine>

#include "util/type/uninitialized.h"
#include "async/concept.h"
#include "async/task.h"
#include "async/async_impl/return_previous.h"

namespace spy {

	struct WhenAnyCtlBlock {
	public:
		static constexpr std::size_t NULL_INDEX = std::size_t(-1);

	public:
		/// Index of the last finished task
		std::size_t             idx{NULL_INDEX};
		/// The handle of the last coroutine in case of exception
		std::coroutine_handle<> previous_handle{};
		/// The pointer to the exception frame
		std::exception_ptr      exception_ptr{};
	};

	struct WhenAnyAwaiter {
	public:
		/// Control metadata of coroutine
		WhenAnyCtlBlock &                   control_block;
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
	ReturnPreviousTask whenAnyHelper(auto &&t, WhenAnyCtlBlock &control,
									 Uninitialized<T> &result, std::size_t index) {
		try {
			result.put_value((co_await std::forward<decltype(t)>(t), NonVoidHelper<>()));
		} catch (...) {
			control.exception_ptr = std::current_exception();
			co_return control.previous_handle;
		}
		// Record the index of the completed task
		control.idx = index;
		co_return control.previous_handle;
	}

	template <std::size_t... Is, class... Ts>
	Task<std::variant<typename AwaitableTraits<Ts>::NonVoidRetType...>>
	whenAnyImpl(std::index_sequence<Is...>, Ts &&...ts) {
		// Generate metadata for coroutines
		WhenAnyCtlBlock control{};
		std::tuple<Uninitialized<typename AwaitableTraits<Ts>::RetType>...> result;
		// Generate the array of all tasks
		ReturnPreviousTask taskArray[]{ whenAnyHelper(ts, control, std::get<Is>(result), Is)... };
		// Start coroutine
		co_await WhenAnyAwaiter(control, taskArray);
		// Collect results
		Uninitialized<std::variant<typename AwaitableTraits<Ts>::NonVoidRetType...>> varResult;
		((control.idx == Is &&
		  (varResult.put_value(
				  std::in_place_index<Is>,
				  std::get<Is>(result).moveValue()),
				  0)),
				...);
		co_return varResult.move_value();
	}

	/*!
	 * @brief Execute coroutine tasks at once and return if anyone finishes
	 * @tparam T The type of task
	 * @tparam Alloc The allocator for task array
	 * @param tasks The vector contains all tasks to be executed
	 * @return Coroutine handle after each execution and return.
	 */
	template <Awaitable... Ts>
		requires(sizeof...(Ts) != 0)
		auto when_any(Ts &&...ts) {
		return whenAnyImpl(std::make_index_sequence<sizeof...(Ts)>{},
				std::forward<Ts>(ts)...);
	}

	/*!
	 * @brief Execute coroutine tasks at once and return if anyone finishes
	 * @tparam T The type of task
	 * @tparam Alloc The allocator for task array
	 * @param tasks The vector contains all tasks to be executed
	 * @return Coroutine handle after each execution and return.
	 */
	template <Awaitable T, class Alloc = std::allocator<T>>
	Task<typename AwaitableTraits<T>::RetType>
			when_any(std::vector<T, Alloc> const &tasks) {
		// Generate a metadata for coroutines
		WhenAnyCtlBlock control{tasks.size()};
		// Initialize allocator for tasks
		Alloc alloc = tasks.get_allocator();
		Uninitialized<typename AwaitableTraits<T>::RetType> result;
		// Execution
		{
			// Generate the array of all tasks
			std::vector<ReturnPreviousTask, Alloc> taskArray(alloc);
			taskArray.reserve(tasks.size());
			for (auto &task: tasks) {
				taskArray.push_back(when_all_helper(task, control, result));
			}
			// Start coroutine
			co_await WhenAnyAwaiter(control, taskArray);
		}
		co_return result.move_value();
	}

} // namespace spy