/*
 * @author: BL-GS 
 * @date:   24-4-13
 */

#pragma once

#include <memory>
#include <magic_enum.hpp>

#include "util/logger.h"
#include "backend/cpu/type.h"
#include "backend/cpu/default.h"

namespace spy {

    enum class CPUBackendPolicy {
        Default
    };

    class CPUBackendFactory {

    public:
        static std::unique_ptr<CPUBackend> make_cpu_backend(CPUBackendPolicy policy, int num_thread, int64_t mem_size) {
            switch (policy) {
            case CPUBackendPolicy::Default:
                return std::make_unique<DefaultCPUBackend>(num_thread, mem_size);

            default:
                spy_assert(false, "Unknown policy for CPU Backend: {}", magic_enum::enum_name(policy));
            }
        }

    };

} // namespace spy