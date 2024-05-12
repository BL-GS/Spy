/*
 * @author: BL-GS 
 * @date:   24-4-24
 */

#pragma once

#include <memory>
#include <magic_enum.hpp>

#include "util/logger.h"
#include "backend/gpu/type.h"
#include "backend/gpu/default.h"

namespace spy {

	enum class GPUBackendPolicy {
		Default
	};

	class GPUBackendFactory {

	public:
		static std::unique_ptr<GPUBackend> make_cpu_backend(GPUBackendPolicy policy, int num_thread, int64_t mem_size) {
			switch (policy) {
				case GPUBackendPolicy::Default:
					return std::make_unique<DefaultGPUBackend>(num_thread, mem_size);

				default:
					spy_assert(false, "Unknown policy for CPU Backend: {}", magic_enum::enum_name(policy));
			}
		}

	};

} // namespace spy
