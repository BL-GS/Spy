#pragma once

#include <cstdint>

namespace spy {

    void spy_start_tracing();

    void spy_stop_tracing(const char *save_path);

    void spy_enable_tracing();

    void spy_disable_tracing();

    void spy_begin_event(const char *name);

    void spy_end_event();

    void spy_begin_event_at_track(const char *name, uint64_t track_id);

    void spy_end_event_at_track(uint64_t track_id);    


} // namespace spy