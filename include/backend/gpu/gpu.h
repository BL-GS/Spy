/*
 * @author: BL-GS 
 * @date:   24-4-24
 */

#pragma once

#include <memory>
#include <magic_enum.hpp>

#include "util/shell/logger.h"
#include "backend/gpu/type.h"
#include "backend/gpu/default.h"

namespace spy {

	enum class GPUBackendPolicy {
		Default
	};

	class GPUBackendFactory {

	public:
		static std::unique_ptr<GPUBackend> make_gpu_backend(GPUBackendPolicy policy, int device_id) {
			switch (policy) {
				case GPUBackendPolicy::Default:
					return std::make_unique<DefaultGPUBackend>(device_id);

				default:
					spy_assert(false, "Unknown policy for GPU Backend: {}", magic_enum::enum_name(policy));
			}
		}

	};

} // namespace spy
