#pragma once

#include <chrono>
#include <string>
#include "perf/listener/abstract_listener.h"

namespace spy::perf {

    class TimerProfiler final: public AbstractProfilerListener {
    private:
        std::chrono::steady_clock             clock_;
        std::chrono::steady_clock::time_point start_time_;

    public:
        TimerProfiler() = default;

        ~TimerProfiler() = default;

    public:
        void start() override { start_time_ = clock_.now(); }

        ProfileRecord profile() override {
            return {{
                { "time", std::to_string(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                            clock_.now() - start_time_
                        ).count()) 
                    }
            }};
        }
    };

} // namespace spy::perf