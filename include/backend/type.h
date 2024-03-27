#pragma once

namespace spy {

    enum class MemoryLevel: int {
        Cache       = 0, 
        ScratchPad  = 1, 
        Memory      = 2, 
        Disk        = 3
    };

    enum class BackendType { CPU, GPU };

} // namespace spy