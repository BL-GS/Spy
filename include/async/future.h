#pragma once

#include <coroutine>
#include <memory>

#include "util/type/uninitialized.h"
#include "async/async_impl/basic_loop.h"
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
			if (coroutine_waiting_.compare_exchange_strong(
					expect, coroutine.address(), std::memory_order_acq_rel)) {
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
		std::exception_ptr exception_ptr_;

		Uninitialized<T> data_;
    
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

		void set_exception(std::exception_ptr e) {
			exception_ptr_ = e;
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
    	FutureToken<T> *get_token() const noexcept { return token_.get(); }

		Task<T> wait() const            { co_return co_await *this; }
	};

} // namespace spy