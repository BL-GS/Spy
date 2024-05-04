 #pragma once
 
#include <cstddef>
#include <vector>
#include <coroutine>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <liburing.h>
#include <fcntl.h>

#include "async/async_impl/basic_loop.h"

 namespace spy {

    class UringLoop {
    private:
        io_uring ring;

        std::vector<std::coroutine_handle<>> coroutine_batch;

    public:
        explicit UringLoop(size_t entries = 32) {
			std::memset(&ring, 0, sizeof(ring));
            const int ret = io_uring_queue_init(entries, &ring, 0);
			if (ret != 0) { throw std::system_error(-ret, std::system_category()); }
        }

        ~UringLoop() noexcept { io_uring_queue_exit(&ring); }

    public:
		/*!
		 * @brief Submit buffered tasks to the uring execution queue
		 */
        void submit() { io_uring_submit(&ring); }

		/*!
		 * @brief Execute single tasks
		 */
        void run_single();

		/*!
		 * @brief Execute several tasks and synchronize with them
		 * @param num_task The number of tasks
		 * @param timeout The maximum time for waiting single task
		 * @return false if timeout or interval happen, otherwise true.
		 */
        bool run_batched_and_wait(size_t num_task, struct __kernel_timespec *timeout);

		/*!
		 * @brief Execute several tasks and return immediately
		 * @param num_task The number of tasks
		 */
        void run_batched_and_nowait(size_t num_task);

    public:
		/*!
		 * @brief Get the reference of the uring structure
		 */
        io_uring &  get_ring()          { return ring; }

	    /*!
		 * @brief Get the pointer to the uring structure
		 */
		io_uring *  get_ring_ptr()      { return &ring; }

		/*!
		 * @brief The number of events in the commited queue
		 */
        size_t      has_event() const   { return io_uring_cq_ready(&ring); }
    };

	 inline UringLoop &get_thread_local_uring_loop();

	/// @brief The unit operator for asynchronous uring execution
    struct UringOperator {
        friend class UringLoop;
    public:
        struct Awaiter {
        public:
            UringOperator *op_ptr;

        public:
            bool await_ready() const noexcept { return false; }

            void await_suspend(std::coroutine_handle<> coroutine) {
                op_ptr->previous_handle = coroutine;
                op_ptr->res_            = -ENOSYS;
                // submit
				UringLoop &loop = get_thread_local_uring_loop();
                loop.submit();
            }

            int await_resume() const noexcept { return op_ptr->res_; }
        };

    private:
        std::coroutine_handle<> previous_handle;

        union {
            int             res_;
            io_uring_sqe *  sqe_ptr_;
        };
        
    public:
        explicit UringOperator(const auto &func) {
			UringLoop &loop = get_thread_local_uring_loop();
            sqe_ptr_ = io_uring_get_sqe(loop.get_ring_ptr());
            if (sqe_ptr_ == nullptr) [[unlikely]] { throw std::bad_alloc(); }
            func(sqe_ptr_);
	        // Set this operator as the consequent logic
	        io_uring_sqe_set_data(sqe_ptr_, this);
        }

        UringOperator(UringOperator &&) = delete;

		~UringOperator() noexcept = default;

        Awaiter operator co_await() { return Awaiter{this}; }

    public:
        friend UringOperator &uring_join(UringOperator &&lhs, UringOperator &&rhs) {
            lhs.sqe_ptr_->flags |= IOSQE_IO_LINK;
            rhs.previous_handle  = std::noop_coroutine();
            return lhs;
        }
    };


    void UringLoop::run_single() {
        BasicLoop &basic_loop = get_thread_local_basic_loop();
		// Wait for one IO tasks finished
        io_uring_cqe *cqe_ptr;
        io_uring_wait_cqe(&ring, &cqe_ptr);
        auto *op_ptr = reinterpret_cast<UringOperator *>(cqe_ptr->user_data);

        op_ptr->res_ = cqe_ptr->res;
        io_uring_cqe_seen(&ring, cqe_ptr);
        basic_loop.enqueue(op_ptr->previous_handle);
    }

    bool UringLoop::run_batched_and_wait(size_t num_task, struct __kernel_timespec *timeout) {
        BasicLoop &basic_loop = get_thread_local_basic_loop();

        io_uring_cqe *cqe_ptr;
        int res = io_uring_wait_cqes(&ring, &cqe_ptr, num_task, timeout, nullptr);

        if (res == -EINTR) { return false; }
        if (res == -ETIME) { return false; }

        unsigned head;
        unsigned num_got = 0;

        io_uring_for_each_cqe(&ring, head, cqe_ptr) {
            auto *op_ptr = reinterpret_cast<UringOperator *>(io_uring_cqe_get_data(cqe_ptr));
            op_ptr->res_ = cqe_ptr->res;

            basic_loop.enqueue(op_ptr->previous_handle);
            ++num_got;
        }

        io_uring_cq_advance(&ring, num_got);
        return true;
    }

    void UringLoop::run_batched_and_nowait(size_t num_task) {
        BasicLoop &basic_loop = get_thread_local_basic_loop();

        io_uring_cqe *cqe_ptr = nullptr;
        int res = io_uring_wait_cqes(&ring, &cqe_ptr, num_task, nullptr, nullptr);

        if (res == -EINTR) { return; }

        unsigned head;
        unsigned num_got = 0;

        io_uring_for_each_cqe(&ring, head, cqe_ptr) {
            auto *op_ptr = reinterpret_cast<UringOperator *>(cqe_ptr->user_data);
            op_ptr->res_ = cqe_ptr->res;

            basic_loop.enqueue(op_ptr->previous_handle);
            ++num_got;
        }

        io_uring_cq_advance(&ring, num_got);
    }


	 inline UringOperator uring_nop() {
		 return UringOperator([&](io_uring_sqe *sqe) { io_uring_prep_nop(sqe); });
	 }

	 inline UringOperator uring_openat(int dirfd, char const *path, int flags,
	                             mode_t mode) {
		 return UringOperator([&](io_uring_sqe *sqe) {
			 io_uring_prep_openat(sqe, dirfd, path, flags, mode);
		 });
	 }

	 inline UringOperator uring_openat_direct(int dirfd, char const *path, int flags,
	                                    mode_t mode, int file_index) {
		 return UringOperator([&](io_uring_sqe *sqe) {
			 io_uring_prep_openat_direct(sqe, dirfd, path, flags, mode, file_index);
		 });
	 }

	 inline UringOperator uring_statx(int dirfd, char const *path, int flags,
	                            unsigned int mask, struct statx *statxbuf) {
		 return UringOperator([&](io_uring_sqe *sqe) {
			 io_uring_prep_statx(sqe, dirfd, path, flags, mask, statxbuf);
		 });
	 }

	 inline UringOperator uring_read(int fd, std::span<char> buf, std::uint64_t offset) {
		 return UringOperator([&](io_uring_sqe *sqe) {
			 io_uring_prep_read(sqe, fd, buf.data(), buf.size(), offset);
		 });
	 }

	 inline UringOperator uring_write(int fd, std::span<char const> buf,
	                            std::uint64_t offset) {
		 return UringOperator([&](io_uring_sqe *sqe) {
			 io_uring_prep_write(sqe, fd, buf.data(), buf.size(), offset);
		 });
	 }

	 inline UringOperator uring_read_fixed(int fd, std::span<char> buf,
	                                 std::uint64_t offset, int buf_index) {
		 return UringOperator([&](io_uring_sqe *sqe) {
			 io_uring_prep_read_fixed(sqe, fd, buf.data(), buf.size(), offset,
			                          buf_index);
		 });
	 }

	 inline UringOperator uring_write_fixed(int fd, std::span<char const> buf,
	                                  std::uint64_t offset, int buf_index) {
		 return UringOperator([&](io_uring_sqe *sqe) {
			 io_uring_prep_write_fixed(sqe, fd, buf.data(), buf.size(), offset,
			                           buf_index);
		 });
	 }

	 inline UringOperator uring_close(int fd) {
		 return UringOperator([&](io_uring_sqe *sqe) { io_uring_prep_close(sqe, fd); });
	 }

	 inline UringOperator uring_shutdown(int fd, int how) {
		 return UringOperator(
				 [&](io_uring_sqe *sqe) { io_uring_prep_shutdown(sqe, fd, how); });
	 }

	 inline UringOperator uring_fsync(int fd, unsigned int flags) {
		 return UringOperator(
				 [&](io_uring_sqe *sqe) { io_uring_prep_fsync(sqe, fd, flags); });
	 }

	 inline UringOperator uring_cancel(UringOperator *op, unsigned int flags) {
		 return UringOperator(
				 [&](io_uring_sqe *sqe) { io_uring_prep_cancel(sqe, op, flags); });
	 }

	 inline UringOperator uring_timeout(struct __kernel_timespec *ts, unsigned int count,
	                              unsigned int flags) {
		 return UringOperator([&](io_uring_sqe *sqe) {
			 io_uring_prep_timeout(sqe, ts, count, flags);
		 });
	 }

	 inline UringOperator uring_link_timeout(struct __kernel_timespec *ts,
	                                   unsigned int flags) {
		 return UringOperator([&](io_uring_sqe *sqe) {
			 io_uring_prep_link_timeout(sqe, ts, flags);
		 });
	 }

	 inline UringOperator uring_timeout_update(UringOperator *op, struct __kernel_timespec *ts,
	                                     unsigned int flags) {
		 return UringOperator([&](io_uring_sqe *sqe) {
			 io_uring_prep_timeout_update(
					 sqe, ts, reinterpret_cast<std::uintptr_t>(op), flags);
		 });
	 }

	 inline UringOperator uring_timeout_remove(UringOperator *op, unsigned int flags) {
		 return UringOperator([&](io_uring_sqe *sqe) {
			 io_uring_prep_timeout_remove(sqe, reinterpret_cast<std::uintptr_t>(op),
			                              flags);
		 });
	 }

	 inline UringOperator uring_splice(int fd_in, std::int64_t off_in, int fd_out,
	                             std::int64_t off_out, std::size_t nbytes,
	                             unsigned int flags) {
		 return UringOperator([&](io_uring_sqe *sqe) {
			 io_uring_prep_splice(sqe, fd_in, off_in, fd_out, off_out, nbytes,
			                      flags);
		 });
	 }
	
} // namespace spy