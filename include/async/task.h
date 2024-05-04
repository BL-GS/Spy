/*
 * @author: BL-GS 
 * @date:   24-5-1
 */

#pragma once

#include <coroutine>
#include <exception>
#include <utility>

#include "util/type/uninitialized.h"

namespace spy {

	struct PromiseBase {
	public:
		struct FinalAwaiter {
			/*!
			 * @brief Suspend at the beginning of coroutine
			 */
			bool await_ready() const noexcept { return false; }


			template <class T_Promise>
			std::coroutine_handle<> await_suspend(std::coroutine_handle<T_Promise> coroutine) const noexcept {
				return static_cast<PromiseBase &>(coroutine.promise()).previous_handle_;
			}

			/*!
			 * @brief No return
			 */
			void await_resume() const noexcept {}
		};

	protected:
		std::exception_ptr      exception_ptr_{};

	private:
		std::coroutine_handle<> previous_handle_;

	public:
		PromiseBase &operator=(PromiseBase &&) = delete;

	public:
		void set_previous(std::coroutine_handle<> previous) noexcept { previous_handle_ = previous; }

	public:
		/*!
		 * @brief Stop at the beginning
		 */
		auto initial_suspend() noexcept { return std::suspend_always(); }

		/*!
		 * @brief Return the FinalAwaiter at the end of coroutine
		 */
		auto final_suspend() noexcept  { return FinalAwaiter(); }

		void unhandled_exception() noexcept { exception_ptr_ = std::current_exception(); }
	};

	/*!
	 * @brief Promise type for future task
	 * @tparam T The type of result
	 */
	template <class T>
	struct Promise : PromiseBase {
	private:
		/// Uninitialized storage for the final result
		Uninitialized<T> result_;

	public:
		void return_value(T &&ret) 		{ result_.put_value(std::move(ret)); }

		void return_value(const T &ret) { result_.put_value(ret); }

		T result() {
			if (exception_ptr_) [[unlikely]] { std::rethrow_exception(exception_ptr_); }
			return result_.move_value();
		}

		auto get_return_object() {
			return std::coroutine_handle<Promise>::from_promise(*this);
		}
	};

	/*!
	 * @brief Specification of Promise for void.
	 */
	template <>
	struct Promise<void> : PromiseBase {
		void return_void() noexcept {}

		void result() {
			// rethrow exception if the coroutine gets exception
			if (exception_ptr_) [[unlikely]] { std::rethrow_exception(exception_ptr_); }
		}

		auto get_return_object() {
			return std::coroutine_handle<Promise>::from_promise(*this);
		}
	};

	/*!
	 * @brief Default awaiter for Task
	 * @tparam T The type of result
	 * @tparam T_Promise The promised type of result
	 */
	template <class T, class T_Promise>
	struct TaskAwaiter {
	public:
		using promise_type = T_Promise;
		using HandleType  = std::coroutine_handle<promise_type>;

	public:
		HandleType handle;

	public:
		/*!
		 * @brief Not stop at entry
		 */
		bool await_ready() const noexcept { return false; }

		/*!
		 * @brief Store the last handle and return to the caller
		 * @param coroutine the last coroutine handle
		 * @return The handle with promised data
		 */
		HandleType await_suspend(std::coroutine_handle<> coroutine) const noexcept {
			T_Promise &promise = handle.promise();
			promise.set_previous(coroutine);
			return handle;
		}

		/*!
		 * @brief Return the final result of coroutine
		 * @return The promised data
		 */
		T await_resume() const { return handle.promise().result(); }
	};

	/*!
	 * @brief Default handle for coroutine task.
	 * @details It acts as:
	 * - no wait at the beginning
	 * - store the handle each time `co_await`
	 * - return the final result at the end
	 */
	template <class T = void, class T_Promise = Promise<T>>
	struct [[nodiscard]] Task {
		using promise_type = T_Promise;
		using HandleType  = std::coroutine_handle<promise_type>;
		using AwaiterType = TaskAwaiter<T, promise_type>;

	private:
		HandleType handle_;

	public:
		Task(HandleType coroutine = nullptr) noexcept : handle_(coroutine) {}

		Task(Task &&that) noexcept : handle_(that.handle_) { that.handle_ = nullptr; }

		Task &operator=(Task &&that) noexcept { std::swap(handle_, that.handle_); }

		~Task() { if (handle_) { handle_.destroy(); } }

		auto operator co_await() const noexcept { return AwaiterType(handle_); }

	public:
		/*!
		 * @brief Get the current coroutine handle
		 */
		HandleType get() const noexcept { return handle_; }

		/*!
		 * @brief Release coroutine handle and set it as `nullptr`
		 * @return The origin coroutine handle
		 * @note The handle has not been freed
		 */
		HandleType release() noexcept { return std::exchange(handle_, nullptr); }
	};

} // namespace spy