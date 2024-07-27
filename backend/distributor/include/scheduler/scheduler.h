#pragma once

#include <memory>

#include "graph/graph.h"
#include "backend/config.h"
#include "scheduler/common.h"

namespace spy {

    class GraphScheduler {
	protected:
		AbstractBackend *backend_ptr_;

	public:
		GraphScheduler(AbstractBackend *backend_ptr): backend_ptr_(backend_ptr) {}

        virtual ~GraphScheduler() noexcept = default;

    public:
        virtual void execute(Graph &graph, GraphControlHeader &control_header) = 0;
    };

	/// 
	/// @brief A factory class scheduling graph scheduler dynamically or statically.
	/// @note Please look up the `GraphSchedulerImpl<SchedulerType>` for the detail of scheduler.
	/// 
	class GraphSchedulerBuilder {
	public:
		/*!
		 * @brief Distribute the scheduler dynamically
		 */
		static std::unique_ptr<GraphScheduler> build_scheduler(std::string_view name, AbstractBackend *backend_ptr);
	};
    
} // namespace spy