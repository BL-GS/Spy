#pragma once

#include <coroutine>
#include <condition_variable>
#include <memory>

#include "util/type/non_void.h"
#include "util/type/uninitialized.h"
#include "async/async_impl/basic_loop.h"
#include "async/async_impl/ignore_return_promise.h"
#include "async/task.h"

namespace spy {

	struct FutureTokenBase {
    private:
		std::coroutine_handle<> coroutine_owning_{nullptr};
		std::atomic<void *>     coroutine_waiting_{nullptr};

    public:
		FutureTokenBase &operator=(FutureTokenBase &&) = delete;

		~FutureTokenBase() {
			if (coroutine_owning_) [[likely]] { coroutine_owning_.destroy(); }
		}

    public:
		void set_owning_coroutine(std::coroutine_handle<> coroutine) noexcept {
			coroutine_owning_ = coroutine;
		}

		std::coroutine_handle<>
		set_waiting_coroutine(std::coroutine_handle<> coroutine) {
			void *expect{nullptr};
			if (coroutine_waiting_.compare_exchange_strong(expect, coroutine.address())) {
				return std::noop_coroutine();
			} else {
				return coroutine;
			}
		}

		std::coroutine_handle<> get_waiting_coroutine() {
			auto p = coroutine_waiting_.exchange((void *)-1, std::memory_order_acq_rel);
			return p ? std::coroutine_handle<>::from_address(p) : nullptr;
		}
	};

	template <class T = void>
	struct FutureToken : FutureTokenBase {
	private:
		std::exception_ptr  exception_ptr_;
		Uninitialized<T>    data_;
    
    public:
    	FutureToken &operator=(FutureToken &&) = delete;

    public:
		template <class U>
		void set_value(U &&value) {
			data_.putValue(std::forward<U>(value));
			if (auto coroutine = get_waiting_coroutine()) {
                BasicLoop &cur_basic_loop = get_thread_local_basic_loop();
				cur_basic_loop.enqueue(coroutine);
			}
		}

		void set_exception(std::exception_ptr exception_ptr) {
			exception_ptr_ = exception_ptr;
			if (auto coroutine = get_waiting_coroutine()) {
                BasicLoop &cur_basic_loop = get_thread_local_basic_loop();
				cur_basic_loop.enqueue(coroutine);
			}
		}

		T fetch_value() {
			if (exception_ptr_) [[unlikely]] { std::rethrow_exception(exception_ptr_); }
			if constexpr (!std::is_void_v<T>) { return data_.moveValue(); }
		}
	};

	template <class T>
	struct FutureAwaiter {
    private:
		FutureToken<T> *token_;

    public:
		explicit FutureAwaiter(FutureToken<T> *token) : token_(token) {}

    public:
		bool await_ready() const noexcept { return false; }

		std::coroutine_handle<>
		await_suspend(std::coroutine_handle<> coroutine) const {
			return token_->set_waiting_coroutine(coroutine);
		}

		T await_resume() const { return token_->fetch_value(); }
	};

	template <class T = void>
	struct [[nodiscard]] Future {
    private:
		std::unique_ptr<FutureToken<T>> token_;

    public:
		explicit Future() : token_(std::make_unique<FutureToken<T>>()) {}

		auto operator co_await() const  { return FutureAwaiter<T>(token_.get()); }

    public:
    	FutureToken<T> *get_token() const noexcept { return token_.get();       }

		Task<T>         wait()      const          { co_return co_await *this;  }
	};


	template <class T, class T_Promise>
	inline Task<void, IgnoreReturnPromise<AutoDestroyFinalAwaiter>>
	loop_enqueue_detach_starter(Task<T, T_Promise> task) {
		co_await task;
	}

	template <class T, class T_Promise>
	inline void loop_enqueue_detach(BasicLoop &loop, Task<T, T_Promise> task) {
		auto wrapped = loop_enqueue_detach_starter(std::move(task));
		auto coroutine = wrapped.get();
		loop.enqueue(coroutine);
		wrapped.release();
	}

	template <class T, class T_Promise>
	inline Task<void, IgnoreReturnPromise<>>
	loop_enqueue_future_starter(Task<T, T_Promise> task, FutureToken<T> *token) {
		try {
			token->set_value((co_await task, NonVoidHelper<>()));
		} catch (...) {
	        token->set_exception(std::current_exception());
	    }
	}

	template <class T, class T_Promise>
	inline Future<T> loop_enqueue_future(BasicLoop &loop, Task<T, T_Promise> task) {
		Future<T> future;
		auto *token = future.get_token();
		auto wrapped = loop_enqueue_future_starter(std::move(task), token);
		auto coroutine = wrapped.get();
		token->set_owning_coroutine(coroutine);
		wrapped.release();

		loop.enqueue(coroutine);
		return future;
	}

	template <class T>
	inline Task<void, IgnoreReturnPromise<>>
	loop_enqueue_future_notifier(std::condition_variable &cv, Future<T> &future,
	                          Uninitialized<T> &result,
	                          std::exception_ptr &exception
	) {
		try {
			result.put_value((co_await future, NonVoidHelper<>()));
		} catch (...) {
	        exception = std::current_exception();
	    }
		cv.notify_one();
	}

	template <class T, class T_Promise>
	inline T loop_enqueue_synchronized(BasicLoop &loop, Task<T, T_Promise> task) {
		auto future = loop_enqueue_future(loop, std::move(task));

		std::condition_variable cond;

		Uninitialized<T> result;
		std::exception_ptr exception;
		auto notifier = loop_enqueue_future_notifier(cond, future, result, exception);
		loop.enqueue(notifier.get());

		{
			std::mutex mutex;
			std::unique_lock lock(mutex);
			cond.wait(lock);
			lock.unlock();
		}

		if (exception) [[unlikely]] { std::rethrow_exception(exception); }
		if constexpr (!std::is_void_v<T>) { return result.move_value(); }
	}

	/// @brief Batch several futures and operate altogether
	struct FutureGroup {
	public:
		std::vector<Future<>> future_array;

	public:
		/*!
		 * @brief Add new future into the batch
		 * @param future New future
		 * @return self
		 */
		FutureGroup &add(Future<> future) {
			future_array.push_back(std::move(future));
			return *this;
		}

		/*!
		 * @brief Wait for all futures complete by coroutine
		 * @return A future each time
		 * @note All futures will be release once the coroutine finishes
		 */
		Task<> wait() {
			for (auto &future: future_array) { co_await future; }
			future_array.clear();
		}
	};
} // namespace spy