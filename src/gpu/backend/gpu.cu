/*
 * @author: BL-GS 
 * @date:   24-4-25
 */


#include <cuda.h>
#include <cuda_runtime.h>

#include "backend/gpu/type.h"
#include "gpu_device.h"

namespace spy {

	using GPUBackendMetadata = gpu::DeviceContext;

	inline GPUBackendMetadata &get_gpu_metadata(void *metadata_ptr) {
		return *static_cast<GPUBackendMetadata *>(metadata_ptr);
	}

	GPUBackend::GPUBackend(int device_id) {
		metadata_ptr_ = new GPUBackendMetadata(device_id);
	}

	size_t GPUBackend::get_max_memory_capacity() const {
		auto &metadata = get_gpu_metadata(metadata_ptr_);
		return metadata.get_max_memory_capacity();
	}

	size_t GPUBackend::get_avail_memory_capacity() const {
		auto &metadata = get_gpu_metadata(metadata_ptr_);
		return metadata.get_avail_memory_capacity();
	}
}
