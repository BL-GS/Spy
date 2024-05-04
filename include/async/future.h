#pragma once

#include <coroutine>
#include <condition_variable>
#include <memory>
#include <utility>

#include "util/type/non_void.h"
#include "util/type/uninitialized.h"
#include "async/async_impl/basic_loop.h"
#include "async/async_impl/ignore_return_promise.h"
#include "async/task.h"

namespace spy {

	struct FutureTokenBase {
	public:
		using HandleType = std::coroutine_handle<>;

    private:
		HandleType              coroutine_owning_{nullptr};
		std::atomic<void *>     coroutine_waiting_{nullptr};

    public:
		FutureTokenBase &operator=(FutureTokenBase &&) = delete;

		~FutureTokenBase() {
			if (coroutine_owning_) [[likely]] { coroutine_owning_.destroy(); }
		}

    public:
		void set_owning_coroutine(HandleType coroutine) noexcept { coroutine_owning_ = coroutine; }

		HandleType set_waiting_coroutine(HandleType coroutine) {
			void *expect{nullptr};
			if (coroutine_waiting_.compare_exchange_strong(expect, coroutine.address())) {
				return std::noop_coroutine();
			} else {
				return coroutine;
			}
		}

		HandleType get_waiting_coroutine() {
			auto ptr = coroutine_waiting_.exchange((void *)-1, std::memory_order_acq_rel);
			return ptr ? HandleType::from_address(ptr) : nullptr;
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
			data_.put_value(std::forward<U>(value));
			if (auto coroutine = get_waiting_coroutine()) {
                BasicLoop &cur_basic_loop = get_thread_local_basic_loop();
				cur_basic_loop.enqueue(coroutine);
			}
		}

		void set_exception(std::exception_ptr exception_ptr) {
			exception_ptr_ = std::move(exception_ptr);
			if (auto coroutine = get_waiting_coroutine()) {
                BasicLoop &cur_basic_loop = get_thread_local_basic_loop();
				cur_basic_loop.enqueue(coroutine);
			}
		}

		T fetch_value() {
			if (exception_ptr_) [[unlikely]] { std::rethrow_exception(exception_ptr_); }
			if constexpr (!std::is_void_v<T>) { return data_.move_value(); }
		}
	};

	template <class T>
	struct FutureAwaiter {
		using TokenType  = FutureToken<T>;
		using HandleType = std::coroutine_handle<>;

    private:
		TokenType *token_;

    public:
		explicit FutureAwaiter(TokenType *token) : token_(token) {}

    public:
		bool await_ready() const noexcept { return false; }

		HandleType await_suspend(HandleType coroutine) const {
			return token_->set_waiting_coroutine(coroutine);
		}

		T await_resume() const { return token_->fetch_value(); }
	};

	template <class T = void>
	struct [[nodiscard]] Future {
	public:
		using AwaiterType       = FutureAwaiter<T>;
		using TaskType          = Task<T>;
		using TokenType         = FutureToken<T>;

    private:
		std::unique_ptr<TokenType> token_ptr_;

    public:
		explicit Future() : token_ptr_(std::make_unique<TokenType>()) {}

		auto operator co_await() const  { return AwaiterType(token_ptr_.get()); }

    public:
    	TokenType *get_token() const noexcept { return token_ptr_.get();       }

		TaskType   wait()      const          { co_return co_await *this;  }
	};


	template <class T, class T_Promise>
	inline Task<void, IgnoreReturnPromise<AutoDestroyFinalAwaiter>>
	loop_enqueue_detach_starter(Task<T, T_Promise> task) {
		co_await task;
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

	template <class T>
	inline Task<void, IgnoreReturnPromise<>>
	loop_enqueue_future_notifier(std::condition_variable &cond, Future<T> &future,
	                             Uninitialized<T> &result, std::exception_ptr &exception ) {
		try {
			result.put_value((co_await future, NonVoidHelper<>()));
		} catch (...) {
	        exception = std::current_exception();
	    }
		cond.notify_one();
	}

	/*!
	 * @brief Enqueue a coroutine task and execute it without any explicit synchronization and return.
	 * @tparam T The type of result
	 * @tparam T_Promise The type of promise
	 * @param loop The basic loop to execute the coroutine task
	 * @param task The coroutine task
	 */
	template <class T, class T_Promise>
	inline void loop_enqueue_detach(BasicLoop &loop, Task<T, T_Promise> task) {
		auto wrapped = loop_enqueue_detach_starter(std::move(task));
		auto coroutine = wrapped.get();
		loop.enqueue(coroutine);
		wrapped.release();
	}

	/*!
	 * @brief Enqueue a coroutine task and return a future result
	 * @tparam T The type of result
	 * @tparam T_Promise The type of promise
	 * @param loop The basic loop to execute the coroutine task
	 * @param task The coroutine task
	 * @return The future result
	 */
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

	/*!
	 * @brief Enqueue a coroutine task and wait for result synchronously
	 * @tparam T The type of result
	 * @tparam T_Promise The type of promise
	 * @param loop The basic loop to execute the coroutine task
	 * @param task The coroutine task
	 * @return The result of coroutine task
	 */
	template <class T, class T_Promise>
	inline T loop_enqueue_synchronized(BasicLoop &loop, Task<T, T_Promise> task) {
		// Enqueue task
		auto future = loop_enqueue_future(loop, std::move(task));
		Uninitialized<T>   result;
		std::exception_ptr exception;
		// Synchronization with the task
		std::condition_variable cond;
		auto notifier = loop_enqueue_future_notifier(cond, future, result, exception);
		loop.enqueue(notifier.get());
		{ // Wait for notification
			std::mutex mutex;
			std::unique_lock lock(mutex);
			cond.wait(lock);
			lock.unlock();
		}
		// Throw exception if get one in coroutine
		if (exception) [[unlikely]] { std::rethrow_exception(exception); }
		// Return values
		if constexpr (!std::is_void_v<T>) { return result.move_value(); }
	}

	/// @brief Batch several futures and operate altogether
	template<class T = void>
	struct FutureGroup {
	public:
		using TaskType   = Task<T>;
		using FutureType = Future<T>;

	public:
		std::vector<FutureType> future_array;

	public:
		/*!
		 * @brief Add new future into the batch
		 * @param future New future
		 * @return self
		 */
		FutureGroup &add(FutureType future) {
			future_array.push_back(std::move(future));
			return *this;
		}

		/*!
		 * @brief Wait for all futures complete by coroutine
		 * @return A future each time
		 * @note All futures will be release once the coroutine finishes
		 */
		TaskType wait() {
			for (auto &future: future_array) { co_await future; }
			future_array.clear();
		}
	};
} // namespace spy