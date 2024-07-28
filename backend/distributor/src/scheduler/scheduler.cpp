#include <string_view>

#include "backend/config.h"
#include "scheduler/policy/default.h"
#include "scheduler/scheduler.h"

namespace spy {

    std::unique_ptr<GraphScheduler> GraphSchedulerBuilder::build_scheduler(std::string_view name, AbstractBackend *backend_ptr) {
        if (name == "greedy") {
            return std::make_unique<DefaultGraphScheduler>(backend_ptr);
        }

        spy_warn("unknown policy of graph scheduler: {}, use `greedy` as default instead", name);
        return std::make_unique<DefaultGraphScheduler>(backend_ptr);
    }

} // namespace spy