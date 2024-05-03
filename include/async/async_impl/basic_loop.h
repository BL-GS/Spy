#pragma once

#include <coroutine>
#include <concurrentqueue/concurrentqueue.h>

namespace spy {

    class BasicLoop {
    private:
        moodycamel::ConcurrentQueue<std::coroutine_handle<>> handle_queue_;

    public:
        BasicLoop()             = default;

        BasicLoop(BasicLoop &&) = delete;

    public:
        bool run() {
            std::coroutine_handle<> coroutine;
            bool success = handle_queue_.try_dequeue(coroutine);
            if (!success) { return false; }

            coroutine.resume();
            return true;
        }

        void enqueue(std::coroutine_handle<> coroutine) {
            handle_queue_.enqueue(coroutine);
        }
    };

    inline BasicLoop &get_thread_local_basic_loop();

} // namespace spy