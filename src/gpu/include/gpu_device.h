/*
 * @author: BL-GS 
 * @date:   24-4-25
 */

#pragma once

#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu_util.h"

namespace spy::gpu {

	struct Device {
	public:
		int     device_id;
		int     compute_capability;
		size_t  shared_memory_per_block;

		bool    virtual_memory_support;
		size_t  virtual_memory_granularity;

		size_t  total_vram;

	public:
		Device(int device_id) {
			CUdevice device;
			CU_CHECK(cuDeviceGet(&device, device_id));

			int device_vmm = 0;
			CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));
			if (device_vmm != 0) {
				CUmemAllocationProp alloc_prop = {};
				alloc_prop.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
				alloc_prop.location.type    = CU_MEM_LOCATION_TYPE_DEVICE;
				alloc_prop.location.id      = device_id;
				CU_CHECK(cuMemGetAllocationGranularity(&virtual_memory_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
			}
			virtual_memory_support = (device_vmm != 0);

			cudaDeviceProp prop;
			CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
			fprintf(stderr, "  Device %d: %s, compute capability %d.%d, VMM: %s\n", device_id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");
			total_vram = prop.totalGlobalMem;

			compute_capability      = 100 * prop.major + 10 * prop.minor;
			shared_memory_per_block = prop.sharedMemPerBlock;
		}

	public:
		size_t get_max_memory_capacity() const {
			size_t free, total;
			CUDA_CHECK(cudaSetDevice(device_id));
			CUDA_CHECK(cudaMemGetInfo(&free, &total));
			return total;
		}

		size_t get_avail_memory_capacity() const {
			size_t free, total;
			CUDA_CHECK(cudaSetDevice(device_id));
			CUDA_CHECK(cudaMemGetInfo(&free, &total));
			return free;
		}
	};

}