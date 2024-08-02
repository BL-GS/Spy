#pragma once

#include <cstdint>
#include <string_view>

namespace spy::perf {

    enum class TraceEventType {
        /// System setup events
        Setup,
        /// Execution Stream control events
        Control,
        /// Operator execution events
        Operator,
        /// File operation or other IO events
        IO,
        /// System resources monitor events
        System
    };
}

#ifdef SPY_PERFETTO_TRACING

#include <memory>
#include <string_view>
#include <fstream>
#include <perfetto.h>

#include "util/shell/logger.h"

namespace spy::perf {

    constexpr std::string_view TRACK_DATA_SAVE_PATH = "spy.perfetto-trace";

    constexpr std::string_view SETUP_CATEGORY    = "spy.setup";
    constexpr std::string_view CONTROL_CATEGORY  = "spy.control";
    constexpr std::string_view OPERATOR_CATEGORY = "spy.operator";
    constexpr std::string_view IO_CATEGORY       = "spy.io";
    constexpr std::string_view SYSTEM_CATEGORY   = "spy.system";

} // namespace spy::perf

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category(spy::perf::SETUP_CATEGORY.data()).SetDescription("Set up the system and maintain it"),
    perfetto::Category(spy::perf::CONTROL_CATEGORY.data()).SetDescription("Control the task distribution and synchronization"),
    perfetto::Category(spy::perf::OPERATOR_CATEGORY.data()).SetDescription("Execute computing tasks"),
    perfetto::Category(spy::perf::IO_CATEGORY.data()).SetDescription("Interaction with data communication"),
    perfetto::Category(spy::perf::SYSTEM_CATEGORY.data()).SetDescription("Interaction with data communication")
);

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace spy::perf {

    using StaticString = perfetto::StaticString;

    class SpyPerfettoTracer {

    private:
        std::string save_path_;

        std::unique_ptr<perfetto::TracingSession> tracing_;
        
    public:
        SpyPerfettoTracer(std::string_view save_path): save_path_(save_path) {

            perfetto::TracingInitArgs args;
            args.backends            = perfetto::kInProcessBackend;
            args.use_monotonic_clock = true;

            perfetto::Tracing::Initialize(args);
            perfetto::TrackEvent::Register();

            perfetto::protos::gen::TrackEventConfig track_event_config;
            track_event_config.add_enabled_categories("*");
            track_event_config.add_enabled_categories(SETUP_CATEGORY.data());
            track_event_config.add_enabled_categories(CONTROL_CATEGORY.data());
            track_event_config.add_enabled_categories(OPERATOR_CATEGORY.data());
            track_event_config.add_enabled_categories(IO_CATEGORY.data());

            perfetto::TraceConfig config;
            config.add_buffers()->set_size_kb(512 * 1024);  // Record up to 10 MiB.
            auto *ds_config = config.add_data_sources()->mutable_config();
            ds_config->set_name("track_event");
            ds_config->set_track_event_config_raw(track_event_config.SerializeAsString());

            tracing_ = perfetto::Tracing::NewTrace();
            tracing_->Setup(config);
            tracing_->StartBlocking();
        }

        ~SpyPerfettoTracer() noexcept {
            perfetto::TrackEvent::Flush();
            tracing_->StopBlocking();
            std::vector<char> trace_data(tracing_->ReadTraceBlocking());

            std::ofstream output;
            output.open(save_path_, std::ios::out | std::ios::binary);
            if (!output.is_open()) {
                spy_error("failed to open file {} for writing tace data", save_path_);
                return;
            }

            output.write(trace_data.data(), trace_data.size());
            if (output.fail()) {
                spy_error("failed writing trace data to file: {}", save_path_);
                return;
            }

            output.close();

            spy_info("saving trace data({} MB) to {}", trace_data.size() / 1024.0 / 1024, save_path_);

            tracing_.reset();
            perfetto::Tracing::Shutdown();            
        }

    public:
        constexpr void begin_event(TraceEventType type, std::string_view name) {
            switch (type) {
            case TraceEventType::Setup: {
                TRACE_EVENT_BEGIN(SETUP_CATEGORY.data(), perfetto::DynamicString{name.data()});
                } return;
            case TraceEventType::Control: {
                TRACE_EVENT_BEGIN(CONTROL_CATEGORY.data(), perfetto::DynamicString{name.data()});
                } return;
            case TraceEventType::Operator: {
                TRACE_EVENT_BEGIN(OPERATOR_CATEGORY.data(), perfetto::DynamicString{name.data()});
                } return;
            case TraceEventType::IO: {
                TRACE_EVENT_BEGIN(IO_CATEGORY.data(), perfetto::DynamicString{name.data()});
                } return;
            case TraceEventType::System: {
                TRACE_EVENT_BEGIN(SYSTEM_CATEGORY.data(), perfetto::DynamicString{name.data()});
                } return;
            }
            spy::spy_unreachable();
        }

        constexpr void begin_event(TraceEventType type, std::string_view name, uint64_t track_id) {
            switch (type) {
            case TraceEventType::Setup: {
                TRACE_EVENT_BEGIN(SETUP_CATEGORY.data(), perfetto::DynamicString{name.data()}, perfetto::Track(track_id));
                } return;
            case TraceEventType::Control: {
                TRACE_EVENT_BEGIN(CONTROL_CATEGORY.data(), perfetto::DynamicString{name.data()}, perfetto::Track(track_id));
                } return;
            case TraceEventType::Operator: {
                TRACE_EVENT_BEGIN(OPERATOR_CATEGORY.data(), perfetto::DynamicString{name.data()}, perfetto::Track(track_id));
                } return;
            case TraceEventType::IO: {
                TRACE_EVENT_BEGIN(IO_CATEGORY.data(), perfetto::DynamicString{name.data()}, perfetto::Track(track_id));
                } return;
            case TraceEventType::System: {
                TRACE_EVENT_BEGIN(SYSTEM_CATEGORY.data(), perfetto::DynamicString{name.data()}, perfetto::Track(track_id));
                } return;
            }
            spy::spy_unreachable(); 
        }

        constexpr void end_event(TraceEventType type) {
            switch (type) {
            case TraceEventType::Setup: {
                TRACE_EVENT_END(SETUP_CATEGORY.data());
                } return;
            case TraceEventType::Control: {
                TRACE_EVENT_END(CONTROL_CATEGORY.data());
                } return;
            case TraceEventType::Operator: {
                TRACE_EVENT_END(OPERATOR_CATEGORY.data());
                } return;
            case TraceEventType::IO: {
                TRACE_EVENT_END(IO_CATEGORY.data());
                } return;
            case TraceEventType::System: {
                TRACE_EVENT_END(SYSTEM_CATEGORY.data());
                } return;
            }
            spy::spy_unreachable();
        }

        constexpr void end_event(TraceEventType type, uint64_t track_id) {
            switch (type) {
            case TraceEventType::Setup: {
                TRACE_EVENT_END(SETUP_CATEGORY.data(), perfetto::Track(track_id));
                } return;
            case TraceEventType::Control: {
                TRACE_EVENT_END(CONTROL_CATEGORY.data(), perfetto::Track(track_id));
                } return;
            case TraceEventType::Operator: {
                TRACE_EVENT_END(OPERATOR_CATEGORY.data(), perfetto::Track(track_id));
                } return;
            case TraceEventType::IO: {
                TRACE_EVENT_END(IO_CATEGORY.data(), perfetto::Track(track_id));
                } return;
            case TraceEventType::System: {
                TRACE_EVENT_END(SYSTEM_CATEGORY.data(), perfetto::Track(track_id));
                } return;
            }
            spy::spy_unreachable();
        }
    };

    inline static SpyPerfettoTracer g_tracker(TRACK_DATA_SAVE_PATH);

    inline void spy_begin_event(TraceEventType type, std::string_view name) {
        g_tracker.begin_event(type, name);
    }

    inline void spy_end_event(TraceEventType type) {
        g_tracker.end_event(type);
    }

    inline void spy_begin_event(TraceEventType type, std::string_view name, uint64_t track_id) {
        g_tracker.begin_event(type, name, track_id);
    }

    inline void spy_end_event(TraceEventType type, uint64_t track_id) {
        g_tracker.end_event(type, track_id);
    }

} // namespace spy::perf

#else // SPY_PERFETTO_TRACING

namespace spy::perf {

    inline void spy_begin_event([[maybe_unused]]TraceEventType type, [[maybe_unused]]std::string_view name) {}

    inline void spy_end_event([[maybe_unused]]TraceEventType type) {}

    inline void spy_begin_event([[maybe_unused]]TraceEventType type, [[maybe_unused]]std::string_view name, uint64_t track_id) {}

    inline void spy_end_event([[maybe_unused]]TraceEventType type, [[maybe_unused]]uint64_t track_id) {}

} // namespace spy::perf

#endif // SPY_PERFETTO_TRACING


namespace spy::perf {

    /* Derives */

    inline void spy_begin_setup_event(std::string_view name) { spy_begin_event(TraceEventType::Setup, name); }
    inline void spy_begin_control_event(std::string_view name) { spy_begin_event(TraceEventType::Control, name); }
    inline void spy_begin_operator_event(std::string_view name) { spy_begin_event(TraceEventType::Operator, name); }
    inline void spy_begin_io_event(std::string_view name) { spy_begin_event(TraceEventType::IO, name); }
    inline void spy_begin_system_event(std::string_view name) { spy_begin_event(TraceEventType::System, name); }

    inline void spy_end_setup_event() { spy_end_event(TraceEventType::Setup); }
    inline void spy_end_control_event() { spy_end_event(TraceEventType::Control); }
    inline void spy_end_operator_event() { spy_end_event(TraceEventType::Operator); }
    inline void spy_end_io_event() { spy_end_event(TraceEventType::IO); }
    inline void spy_end_system_event() { spy_end_event(TraceEventType::System); }

    inline void spy_begin_setup_event(std::string_view name, uint64_t track_id) { spy_begin_event(TraceEventType::Setup, name, track_id); }
    inline void spy_begin_control_event(std::string_view name, uint64_t track_id) { spy_begin_event(TraceEventType::Control, name, track_id); }
    inline void spy_begin_operator_event(std::string_view name, uint64_t track_id) { spy_begin_event(TraceEventType::Operator, name, track_id); }
    inline void spy_begin_io_event(std::string_view name, uint64_t track_id) { spy_begin_event(TraceEventType::IO, name, track_id); }
    inline void spy_begin_system_event(std::string_view name, uint64_t track_id) { spy_begin_event(TraceEventType::System, name, track_id); }

    inline void spy_end_setup_event(uint64_t track_id) { spy_end_event(TraceEventType::Setup, track_id); }
    inline void spy_end_control_event(uint64_t track_id) { spy_end_event(TraceEventType::Control, track_id); }
    inline void spy_end_operator_event(uint64_t track_id) { spy_end_event(TraceEventType::Operator, track_id); }
    inline void spy_end_io_event(uint64_t track_id) { spy_end_event(TraceEventType::IO, track_id); }
    inline void spy_end_system_event(uint64_t track_id) { spy_end_event(TraceEventType::System, track_id); }

    /* Macros */

    #define SPY_PERF_EVENT_TRACE(type, name, ...)           \
            spy_begin_event(TraceEventType:: type, name);   \
            __VA_ARGS__                                     \
            spy_end_event(TraceEventType:: type);

    #define SPY_PERF_EVENT_TRACK_TRACE(type, track_id, name, ...)       \
            spy_begin_event(TraceEventType:: type, name, track_id);     \
            __VA_ARGS__                                                 \
            spy_end_event(TraceEventType:: type, track_id);

} // namespace spy::perf