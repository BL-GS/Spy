#include <string_view>

#define BACKEND_SCHEDULER_HEADER_MACRO

#include "backend/backend.h"
#include "scheduler/scheduler.h"
#include "scheduler/default_scheduler.h"

namespace spy {

    std::unique_ptr<GraphScheduler> GraphSchedulerBuilder::build_scheduler(std::string_view name, Backend *backend_ptr) {
        if (name == "greedy") {
            return std::make_unique<DefaultGraphScheduler>(backend_ptr);
        }

        spy_warn("unknown policy of graph scheduler: {}, use `greedy` as default instead", name);
        return std::make_unique<DefaultGraphScheduler>(backend_ptr);
    }

} // namespace spy