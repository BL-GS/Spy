#pragma once

#include <functional>

#include "graph/graph.h"
#include "backend/backend.h"

namespace spy::gpu {

    class GPUBackend;

    struct OperatorEnvParam {
        using TaskFunc = OperatorStatus (*)(GPUBackend *, const OperatorEnvParam &);

        cudaStream_t            stream;
        OperatorNode *          node_ptr;
        TaskFunc                func;
        std::function<void()>   callback;
    };

} // namespace spy::gpu