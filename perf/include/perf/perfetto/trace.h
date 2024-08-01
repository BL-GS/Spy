#pragma once

#include <cstdint>
#include <string_view>

namespace spy::perf {

    enum class TraceEventType {
        Setup,
        Control,
        Operator,
        IO
    };

    void spy_begin_event(TraceEventType type, std::string_view name);

    void spy_end_event(TraceEventType type);

    void spy_begin_event(TraceEventType type, std::string_view name, uint64_t track_id);

    void spy_end_event(TraceEventType type, uint64_t track_id);    

    #define SPY_PERF_EVENT_TRACE(type, name, ...)           \
            spy_begin_event(TraceEventType:: type, name);   \
            __VA_ARGS__                                     \
            spy_end_event(TraceEventType:: type);


} // namespace spy::perf