#pragma once

namespace spy {

    /// The type of data node. Which assist the allocation and schedule of Graph Scheduler.
    /// TODO: It should be determined by the schedule policy.
    enum class DataNodeType: int {
        /// The location and the size remains fixed during the runtime.
        /// It cannot be the output node.
        /// The scheduler SHOULD NOT allocate or deallocate it.
        Constant = 0, 
        /// The content remain fixed, but it wasn't built as a constant node.
        /// Compared to Constant node, it can be the output node, which means its content can be changed during runtime.
        /// The scheduler SHOULD NOT allocate or deallocate it.
        Buffered = 1,
        /// The location and the size may change during the runtime.
        /// The scheduler CAN determine the allocation and deallocation.
        Variable = 2,
        /// The data is owned by another tensor.
        /// The scheduler SHOULD NOT allocate or deallocate it+
        View     = 3,
        /// By default, we set node as variable, scheduler conventionally allocate it every time.
        Default  = Variable
    };

    enum class SchedulerType {
        Default
    };

} // namespace spy