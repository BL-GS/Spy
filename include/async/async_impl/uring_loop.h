 #pragma once
 
 #include <cstddef>
 #include <vector>
 #include <coroutine>
 #include <liburing.h>

 #include "async/async_impl/basic_loop.h"

 namespace spy {

    class UringLoop {
    private:
        io_uring ring;

        std::vector<std::coroutine_handle<>> coroutine_batch;

    public:
        explicit UringLoop(size_t entries = 512) {
            io_uring_queue_init(entries, &ring, 0);
        }

        ~UringLoop() noexcept {
            io_uring_queue_exit(&ring);
        }

    public:
        void submit() { io_uring_submit(&ring); }

        void run_single();

        bool run_batched_and_wait(size_t num_task, struct __kernel_timespec *timeout);

        void run_batched_and_nowait(size_t num_task);

    public:
        io_uring &  get_ring()          { return ring; }

        size_t      has_event() const   { return io_uring_cq_ready(&ring); }
    };


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
                op_ptr->loop_ptr_->submit();
            }

            int await_resume() const noexcept { return op_ptr->res_; }
        };

    private:
        std::coroutine_handle<> previous_handle;

        UringLoop *loop_ptr_;
        union {
            int             res_;
            io_uring_sqe *  sqe_ptr_;
        };
        
    public:
        explicit UringOperator(UringLoop *loop_ptr, const auto &func): loop_ptr_(loop_ptr) { 
            sqe_ptr_ = io_uring_get_sqe(&loop_ptr->get_ring());
            if (sqe_ptr_ == nullptr) [[unlikely]] { throw std::bad_alloc(); }

            io_uring_sqe_set_data(sqe_ptr_, this);
            func(sqe_ptr_);
        }

        UringOperator(UringOperator &&) = delete;

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
            auto *op_ptr = reinterpret_cast<UringOperator *>(cqe_ptr->user_data);
            op_ptr->res_ = cqe_ptr->res;

            basic_loop.enqueue(op_ptr->previous_handle);
            ++num_got;
        }

        io_uring_cq_advance(&ring, num_got);
        return true;
    }

    void UringLoop::run_batched_and_nowait(size_t num_task) {
        BasicLoop &basic_loop = get_thread_local_basic_loop();

        io_uring_cqe *cqe_ptr;
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

    inline UringLoop &get_thread_local_uring_loop();

} // namespace spy