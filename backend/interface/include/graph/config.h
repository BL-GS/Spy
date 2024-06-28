#pragma once

#include "graph/type.h"
#include "graph/graph.h"

namespace spy {

    class GraphScheduler {
    protected:
        std::vector<AbstractBackend *> backend_array_;

    public:
        GraphScheduler(const std::vector<AbstractBackend *> &backend_array): backend_array_(backend_array) {}

        virtual ~GraphScheduler() noexcept = default;

    public:
        virtual void execute(GraphView &graph_view) = 0;

    };

    template<SchedulerType T_scheduler_type>
    class GraphSchedulerImpl {};

} // namespace spy