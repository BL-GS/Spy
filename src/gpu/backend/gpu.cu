/*
 * @author: BL-GS 
 * @date:   24-4-25
 */


#include <cuda.h>
#include <cuda_runtime.h>

#include "backend/gpu/type.h"
#include "backend/gpu/default.h"
#include "gpu_device.h"

namespace spy {

	void print_cuda_devices() {
		int num_device = 0;
		cudaGetDeviceCount(&num_device);

		spy_info("Total Device Number: {}", num_device);
		for (int i = 0; i < num_device; i++) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			spy_info("-- Device Number:                {}", i);
			spy_info("-- Device Name:                  {}", prop.name);
			spy_info("-- Memory Clock Rate (KHz):      {}", prop.memoryClockRate);
			spy_info("-- Memory Bus Width (bits):      {}", prop.memoryBusWidth);
			spy_info("-- Peak Memory Bandwidth (GB/s): {}", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		}
	}

	using GPUBackendMetadata = gpu::DeviceContext;

	inline GPUBackendMetadata &get_gpu_metadata(void *metadata_ptr) {
		return *static_cast<GPUBackendMetadata *>(metadata_ptr);
	}

	GPUBackend::GPUBackend(int device_id) {
		metadata_ptr_ = new GPUBackendMetadata(device_id);
	}

	size_t GPUBackend::get_max_memory_capacity() const {
		const auto &metadata = get_gpu_metadata(metadata_ptr_);
		return metadata.get_max_memory_capacity();
	}

	size_t GPUBackend::get_avail_memory_capacity() const {
		const auto &metadata = get_gpu_metadata(metadata_ptr_);
		return metadata.get_avail_memory_capacity();
	}

	void GPUBackend::sync(int task_token) {
		auto &metadata = get_gpu_metadata(metadata_ptr_);
		if (task_token >= 0 && task_token < GPUBackendMetadata::MAX_NUM_STREAM) { 
			gpu::gpu_check(cudaStreamSynchronize(metadata.get_stream(task_token)));
		} else {
			gpu::gpu_check(cudaDeviceSynchronize());
		}
	}

	void *DefaultGPUBackend::alloc_memory(size_t size) {
		auto &metadata = get_gpu_metadata(metadata_ptr_);
		auto *pool_ptr = metadata.get_memory_pool();
		return pool_ptr->allocate(size);
	}

	void DefaultGPUBackend::dealloc_memory(void *ptr, size_t size) {
		auto &metadata = get_gpu_metadata(metadata_ptr_);
		auto *pool_ptr = metadata.get_memory_pool();
		pool_ptr->deallocate(ptr, size);
	}

}
