#include "perf/perfetto/trace.h"

#ifdef SPY_PERFETTO_TRACING

#include <memory>
#include <string_view>
#include <fstream>
#include <perfetto.h>
#include <magic_enum_switch.hpp>

#include "util/shell/logger.h"

constexpr std::string_view TRACK_DATA_SAVE_PATH = "spy.perfetto-trace";

constexpr std::string_view SETUP_CATEGORY = "spy-setup";

constexpr std::string_view CONTROL_CATEGORY = "spy-control";

constexpr std::string_view OPERATOR_CATEGORY = "spy-operator";

constexpr std::string_view IO_CATEGORY = "spy-io";

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category(SETUP_CATEGORY.data()).SetDescription("Set up the system and maintain it"),
    perfetto::Category(CONTROL_CATEGORY.data()).SetDescription("Control the task distribution and synchronization"),
    perfetto::Category(OPERATOR_CATEGORY.data()).SetDescription("Execute computing tasks"),
    perfetto::Category(IO_CATEGORY.data()).SetDescription("Interaction with data communication")
);

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace spy::perf {

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
        void begin_event(TraceEventType type, std::string_view name) {
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
            }
            spy::spy_unreachable();
            
        }

        void begin_event(TraceEventType type, std::string_view name, uint64_t track_id) {
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
            }
            spy::spy_unreachable();
            
        }

        void end_event(TraceEventType type) {
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
            }
            spy::spy_unreachable();
        }

        void end_event(TraceEventType type, uint64_t track_id) {
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
            }
            spy::spy_unreachable();
        }
    };

    static SpyPerfettoTracer g_tracker(TRACK_DATA_SAVE_PATH);

    void spy_begin_event(TraceEventType type, std::string_view name) {
        g_tracker.begin_event(type, name);
    }

    void spy_end_event(TraceEventType type) {
        g_tracker.end_event(type);
    }

    void spy_begin_event_at_track(TraceEventType type, std::string_view name, uint64_t track_id) {
        g_tracker.begin_event(type, name, track_id);
    }

    void spy_end_event_at_track(TraceEventType type, uint64_t track_id) {
        g_tracker.end_event(type, track_id);
    }

} // namespace spy::perf

#else

namespace spy::perf {

    void spy_start_tracing() {}

    void spy_stop_tracing(const char *save_path) { (void)save_path; }

    void spy_enable_tracing() {}

    void spy_disable_tracing() {}

    void spy_begin_event(const char *name) { (void)name; }

    void spy_end_event() {}

    void spy_begin_event_at_track(const char *name, uint64_t track_id) { (void)name; (void)track_id; }

    void spy_end_event_at_track(uint64_t track_id) { (void)track_id; }

} // namespace spy::perf

#endif // SPY_PERFETTO_TRACING
