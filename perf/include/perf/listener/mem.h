#pragma once

#include "perf/listener/abstract_listener.h"

namespace spy::perf {

    class MemoryProfiler: public AbstractProfilerListener {
    public:
        MemoryProfiler() = default;

        MemoryProfiler(MemoryProfiler &&other) noexcept = default;

        ~MemoryProfiler() noexcept override = default;

    public:
        ProfileRecord get_hardware_info() const override;
        
    public:
        void start() override;

        ProfileRecord profile() override;
    };

} // namespace spy::perf