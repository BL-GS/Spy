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
			bool await_ready() const noexcept {
				return false;
			}

			template <class P>
			std::coroutine_handle<>
			await_suspend(std::coroutine_handle<P> coroutine) const noexcept {
				return static_cast<PromiseBase &>(coroutine.promise()).previous_ptr_;
			}

			void await_resume() const noexcept {}
		};

	protected:
		std::exception_ptr exception_ptr_{};

	private:
		std::coroutine_handle<> previous_ptr_;

		PromiseBase &operator=(PromiseBase &&) = delete;

	public:
		void setPrevious(std::coroutine_handle<> previous) noexcept {
			previous_ptr_ = previous;
		}

	public:
		auto initial_suspend() noexcept {
			return std::suspend_always();
		}

		auto final_suspend() noexcept {
			return FinalAwaiter();
		}

		void unhandled_exception() noexcept {
			exception_ptr_ = std::current_exception();
		}
	};

	template <class T>
	struct Promise : PromiseBase {
	private:
		Uninitialized<T> result_;

	public:
		void return_value(T &&ret) {
			result_.put_value(std::move(ret));
		}

		void return_value(T const &ret) {
			result_.put_value(ret);
		}

		T result() {
			if (exception_ptr_) [[unlikely]] {
                std::rethrow_exception(exception_ptr_);
            }
			return result_.move_value();
		}

		auto get_return_object() {
			return std::coroutine_handle<Promise>::from_promise(*this);
		}
	};

	template <>
	struct Promise<void> : PromiseBase {
		void return_void() noexcept {}

		void result() {
			if (exception_ptr_) [[unlikely]] {
				std::rethrow_exception(exception_ptr_);
			}
		}

		auto get_return_object() {
			return std::coroutine_handle<Promise>::from_promise(*this);
		}
	};

	template <class T, class P>
	struct TaskAwaiter {
	public:
		std::coroutine_handle<P> handle;

	public:
		bool await_ready() const noexcept {
			return false;
		}

		std::coroutine_handle<P>
		await_suspend(std::coroutine_handle<> coroutine) const noexcept {
			P &promise = handle.promise();
			promise.setPrevious(coroutine);
			return handle;
		}

		T await_resume() const {
			return handle.promise().result();
		}
	};

	template <class T = void, class P = Promise<T>>
	struct [[nodiscard]] Task {
		using promise_type = P;

	private:
		std::coroutine_handle<promise_type> handle_;

	public:
		/* Task(std::coroutine_handle<promise_type> coroutine) noexcept */
		/*     : mCoroutine(coroutine) { */
		/* } */

		/* Task(Task &&) = delete; */

		Task(std::coroutine_handle<promise_type> coroutine = nullptr) noexcept : handle_(coroutine) {}

		Task(Task &&that) noexcept : handle_(that.handle_) { that.handle_ = nullptr; }

		Task &operator=(Task &&that) noexcept { std::swap(handle_, that.handle_); }

		~Task() {
			if (handle_) { handle_.destroy(); }
		}

		auto operator co_await() const noexcept {
			return TaskAwaiter<T, P>(handle_);
		}

	public:
		std::coroutine_handle<promise_type> get() const noexcept {
			return handle_;
		}

		std::coroutine_handle<promise_type> release() noexcept {
			return std::exchange(handle_, nullptr);
		}
	};

} // namespace spy