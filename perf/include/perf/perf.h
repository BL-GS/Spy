#pragma once

#include <chrono>

#include "util/shell/logger.h"
#include "perf/csv.h"

namespace spy {

    struct PerformanceProfilerRecord {
    private:
        enum class Status { Init, Start, End };

    public: 
        Status                                  status_;

        std::string                             name_;
        std::chrono::steady_clock::time_point   start_point_;
        std::chrono::steady_clock::time_point   end_point_;

    public:
        PerformanceProfilerRecord(std::string_view name): 
            status_(Status::Init), name_(name) {}

    public:
        void start() { 
            status_      = Status::Start;
            start_point_ = std::chrono::steady_clock::now(); 
        }

        void end() {
            status_     = Status::End;
            end_point_  = std::chrono::steady_clock::now();
        }

        bool complete() const { 
            return status_ == Status::End;
        }
    };

    class PerformanceProfiler {
    private:
        AppendOnlyCsv   csv_file_;

        size_t          counter_;


    public:
        PerformanceProfiler(const std::string &name): 
            csv_file_(name, {"idx", "name", "start", "end"}), counter_(0) {}
        
    public:
        void add_record(const PerformanceProfilerRecord &record) {
            spy_assert(record.complete(), 
                "Expect record {} to be an completed profiler record", record.name_
            );

            csv_file_.add_row({std::to_string(counter_), record.name_, 
                std::to_string(record.start_point_.time_since_epoch().count()), 
                std::to_string(record.end_point_.time_since_epoch().count())
            });
            ++counter_;
        }


    };

} // namespace spy