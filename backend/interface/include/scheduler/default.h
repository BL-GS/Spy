#pragma once

#include "backend/config.h"
#include "scheduler/scheduler.h"

namespace spy {

	///
	/// @brief Default scheduler. 
	/// @details 
	/// - **Topology Algorithm**: This scheduler uses a simple topology algorithm which step to the next node with no alive input.
	/// - **Schedule Strategy**: This scheduler give the responsibility of decresing dependency count to the worker threads, 
	///		which means it will distribute all avaiable tasks to the worker backends. On the other hand, if there is no task in the queue,
	///		it will wait until any worker finishing its job, counting down the dependency counter and push a new OperatorNode into the task queue.
	///		Specially, it execute the view node directly if it acquires little operation(task_num).
	/// - **Memory Allocation Strategy**: It gives the whole determination to the backend.
	/// 
    class DefaultGraphScheduler final: public GraphScheduler {
    public:
        DefaultGraphScheduler(AbstractBackend *backend_ptr): GraphScheduler(backend_ptr) {}

        virtual ~DefaultGraphScheduler() = default;

    public:
        void execute(Graph &graph) override;
    };

} // namespace spy
