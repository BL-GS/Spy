#include <memory>
#include <string>
#include <magic_enum.hpp>

#include "perf/listener/exception.h"
#include "perf/listener/abstract_listener.h"
#include "perf/listener/timer.h"
#include "perf/listener/cpu.h"
#include "perf/listener/mem.h"

namespace spy::perf {

    enum class ListenerType {
        Timer,
        Processor,
        Memory
    };

    struct ListenerFactory {
        static std::unique_ptr<AbstractProfilerListener> build_listener(ListenerType type) {
            switch (type) { 
                case ListenerType::Timer:
                    return std::make_unique<TimerProfiler>();
                case ListenerType::Processor:
                    return std::make_unique<ProcessorProfiler>();
                case ListenerType::Memory:
                    return std::make_unique<MemoryProfiler>();
            }

            const std::string type_name(magic_enum::enum_name(type));
            throw SpyPerfException("unknown profiler: {}", type_name);
            return {};
        }
    };

} // namespace spy::perf