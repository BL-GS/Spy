#pragma once

#include <coroutine>
#include <exception>
#include <span>
#include <utility>

#include "util/type/uninitialized.h"
#include "phase.h"
#include "abstract_backend.h"

namespace spy::cpu {

    struct OperatorResult {
        OperatorStatus      status;
        OperatorPhaseType   phase;

        OperatorResult(): status(OperatorStatus::Success), phase(OperatorPhaseType::Init) {}
        OperatorResult(OperatorPhaseType phase): status(OperatorStatus::Success), phase(phase) {}
        OperatorResult(OperatorStatus status, OperatorPhaseType phase): status(status), phase(phase) {}
    };

	/*!
	 * @brief Promise type for future task
	 * @tparam T The type of result
	 */
	struct TaskPromise {
    public:
		struct FinalAwaiter {
			/*!
			 * @brief Suspend at the beginning of coroutine
			 */
			bool await_ready() const noexcept { return false; }


			std::coroutine_handle<> await_suspend(std::coroutine_handle<TaskPromise> coroutine) const noexcept {
				return static_cast<TaskPromise &>(coroutine.promise()).previous_handle_;
			}

			/*!
			 * @brief No return
			 */
			void await_resume() const noexcept {}
		};

	private:
		/// Uninitialized storage for the final result
		Uninitialized<OperatorResult> result_;

        std::exception_ptr      exception_ptr_{};

        std::coroutine_handle<> previous_handle_;

	public:
		void return_value(OperatorResult &&ret) 	    { result_.put_value(std::move(ret)); }

		void return_value(const OperatorResult &ret)    { result_.put_value(ret); }

		OperatorResult result() {
			if (exception_ptr_) [[unlikely]] { std::rethrow_exception(exception_ptr_); }
			return result_.move_value();
		}

		auto get_return_object() {
			return std::coroutine_handle<TaskPromise>::from_promise(*this);
		}

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
	 * @brief Default awaiter for Task
	 * @tparam T The type of result
	 * @tparam T_Promise The promised type of result
	 */
	template <class T, class T_Promise>
	struct TaskAwaiter {
	public:
		using promise_type = T_Promise;
		using HandleType   = std::coroutine_handle<promise_type>;

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
	struct [[nodiscard]] Task {
		using promise_type = TaskPromise;
		using HandleType   = std::coroutine_handle<promise_type>;
		using AwaiterType  = TaskAwaiter<OperatorResult, promise_type>;

	private:
		HandleType handle_;

	public:
		Task(HandleType coroutine = nullptr) noexcept : handle_(coroutine) {}

		Task(Task &&that) noexcept : handle_(that.handle_) { that.handle_ = nullptr; }

		Task &operator=(Task &&that) noexcept { std::swap(handle_, that.handle_); return *this; }

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

	struct ControlHeader {
		int num_task = 0;
		/// callback
		std::function<void()> callback = nullptr;

		ControlHeader() = default;
		ControlHeader(int num_task): num_task(num_task) {}

		virtual ~ControlHeader() { if (callback) { callback(); } }

		virtual void init([[maybe_unused]]const OperatorEnvParam &param) { }
	};

	struct MuxtexControlHeader: ControlHeader {
		std::mutex mutex;

		MuxtexControlHeader() = default;
		MuxtexControlHeader(int num_task): ControlHeader(num_task) {}
		~MuxtexControlHeader() override = default;
	};

	struct BufferControlHeader: ControlHeader {
		CPUBackend  *			backend_ptr;
		std::span<uint8_t>		data_span;

		BufferControlHeader() = default;
		BufferControlHeader(int num_task): ControlHeader(num_task) {}
		BufferControlHeader(int num_task, CPUBackend *backend_ptr, int size): ControlHeader(num_task), backend_ptr(backend_ptr) {
			data_span = std::span<uint8_t>(static_cast<uint8_t *>(backend_ptr->alloc_memory(size)), size);
		}
		~BufferControlHeader() override { 
			if (backend_ptr != nullptr) { 
				backend_ptr->dealloc_memory(data_span.data(), data_span.size()); 
			} 
		}
	};

	struct OperatorEnvParam {
	public:
		using TaskFunc  = OperatorResult (*)(CPUBackend *, const OperatorEnvParam &, OperatorNode *);

	public:
		/// The concurrency of operator execution
		int concurrency			 = 0;
		/// The id of thread executing the operator
		int tid					 = 0;
		/// 
		TaskFunc func			 = nullptr;
		/// 
		OperatorNode * node_ptr	 = nullptr;
		/// The pointer to the control header
		std::shared_ptr<ControlHeader> header_ptr = nullptr;

	public:
		OperatorEnvParam fork(int new_tid) const { 
			return {
				.concurrency = concurrency,
				.tid		 = new_tid,
				.func	     = func,
				.node_ptr	 = node_ptr,
				.header_ptr  = header_ptr
			};
		}
	};

} // namespace spy::cpu