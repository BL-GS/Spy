#pragma once

#include <memory>

#include "perf/listener/abstract_listener.h"

namespace spy::perf {

    struct ProcessorProfilerInfo {
    public:
        uint64_t total;
        uint64_t user;
        uint64_t system;
        uint64_t io;
        uint64_t idle;

    public:
        void operator-=(const ProcessorProfilerInfo &other) {
            total   -= other.total;
            user    -= other.user;
            system  -= other.system;
            io      -= other.io;
            idle    -= other.idle;
        }

        ProcessorProfilerInfo operator -(const ProcessorProfilerInfo &other) const {
            ProcessorProfilerInfo res = *this;
            res -= other;
            return res;
        }

    public:
        uint64_t usage() const {
            return (total - idle) * 100 / total;
        }
    };

    class ProcessorProfiler: public AbstractProfilerListener {
    protected:
        std::unique_ptr<ProcessorProfilerInfo> prev_info_;

    public:
        ProcessorProfiler();

        ProcessorProfiler(ProcessorProfiler &&other) noexcept = default;

        ~ProcessorProfiler() noexcept override = default;

    public:
        ProfileRecord get_hardware_info() const override;
        
    public:
        void start() override;

        ProfileRecord profile() override;
    };

} // namespace spy::perf