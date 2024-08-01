#include "perf/perfetto/trace.h"

#ifdef SPY_PERFETTO_TRACING

#include <atomic>
#include <memory>
#include <fstream>

#include "perfetto.h"

#define TRACE_CATEGORY "spy-perf"

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category(TRACE_CATEGORY).SetDescription("ASYNC FFN"),
);

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace spy {
    
    std::unique_ptr<perfetto::TracingSession> tracing;
    std::atomic<bool> tracing_enabled{false};

    void spy_start_tracing(void) {
        perfetto::TracingInitArgs args;
        args.backends = perfetto::kInProcessBackend;
        args.use_monotonic_clock = true;
        perfetto::Tracing::Initialize(args);
        perfetto::TrackEvent::Register();

        perfetto::TraceConfig cfg;
        cfg.add_buffers()->set_size_kb(512 * 1024);  // Record up to 10 MiB.
        auto *ds_cfg = cfg.add_data_sources()->mutable_config();
        ds_cfg->set_name("track_event");
        perfetto::protos::gen::TrackEventConfig track_event_cfg;
        track_event_cfg.add_enabled_categories("*");
        track_event_cfg.add_enabled_categories(TRACE_CATEGORY);
        ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

        tracing = perfetto::Tracing::NewTrace();
        tracing->Setup(cfg);
        tracing->StartBlocking();
    }

    void spy_stop_tracing(const char *save_path) {
        puts("Saving trace data...");

        perfetto::TrackEvent::Flush();
        tracing->StopBlocking();
        std::vector<char> trace_data(tracing->ReadTraceBlocking());

        std::ofstream output;
        output.open(save_path, std::ios::out | std::ios::binary);
        if (!output.is_open()) {
            fprintf(stderr, "Failed to open file %s for writing\n", save_path);
            return;
        }
        output.write(trace_data.data(), trace_data.size());
        if (output.fail()) {
            fprintf(stderr, "Failed to write data to file %s\n", save_path);
            output.close();
            return;
        }
        output.close();

        printf(
            "%s: Saved %.3lf MiB trace data to \"%s\"\n",
            __func__,
            trace_data.size() / 1024.0 / 1024,
            save_path
        );

        tracing.reset();
        perfetto::Tracing::Shutdown();
    }


    void spy_enable_tracing(void) {
        tracing_enabled.store(true);
    }

    void spy_disable_tracing(void) {
        tracing_enabled.store(false);
    }

    void spy_begin_event(const char *name) {
        if (tracing_enabled.load(std::memory_order_relaxed)) {
            TRACE_EVENT_BEGIN(TRACE_CATEGORY, perfetto::DynamicString{name});
        }
    }

    void spy_end_event(void) {
        if (tracing_enabled.load(std::memory_order_relaxed)) {
            TRACE_EVENT_END(TRACE_CATEGORY);
        }
    }

    void spy_begin_event_at_track(const char *name, uint64_t track_id) {
        if (tracing_enabled.load(std::memory_order_relaxed)) {
            TRACE_EVENT_BEGIN(TRACE_CATEGORY, perfetto::DynamicString{name}, perfetto::Track(track_id));
        }
    }

    void spy_end_event_at_track(uint64_t track_id) {
        if (tracing_enabled.load(std::memory_order_relaxed)) {
            TRACE_EVENT_END(TRACE_CATEGORY, perfetto::Track(track_id));
        }
    }
} // namespace spy

#else

namespace spy {

    void spy_start_tracing() {}

    void spy_stop_tracing(const char *save_path) { (void)save_path; }

    void spy_enable_tracing() {}

    void spy_disable_tracing() {}

    void spy_begin_event(const char *name) { (void)name; }

    void spy_end_event() {}

    void spy_begin_event_at_track(const char *name, uint64_t track_id) { (void)name; (void)track_id; }

    void spy_end_event_at_track(uint64_t track_id) { (void)track_id; }

} // namespace spy

#endif // SPY_PERFETTO_TRACING
