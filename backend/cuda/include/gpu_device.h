/*
 * @author: BL-GS 
 * @date:   24-4-25
 */

#pragma once

#include <cstddef>
#include <array>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <memory>

#include "util/log/logger.h"
#include "gpu_util.h"
#include "gpu_mem.h"

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
			gpu_check(cuInit(device_id));
			CUdevice device;
			gpu_check(cuDeviceGet(&device, device_id));

			int device_vmm = 0;
			gpu_check(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));
			if (device_vmm != 0) {
				CUmemAllocationProp alloc_prop = {};
				alloc_prop.type             = CU_MEM_ALLOCATION_TYPE_PINNED;
				alloc_prop.location.type    = CU_MEM_LOCATION_TYPE_DEVICE;
				alloc_prop.location.id      = device_id;
				gpu_check(cuMemGetAllocationGranularity(&virtual_memory_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
			}
			virtual_memory_support = (device_vmm != 0);

			cudaDeviceProp prop;
			gpu_check(cudaGetDeviceProperties(&prop, device_id));
			spy_info("Device {}: {}; Compute Capability: {}.{}; VMM: {}", device_id, prop.name, prop.major, prop.minor, device_vmm);
			total_vram = prop.totalGlobalMem;

			compute_capability      = 100 * prop.major + 10 * prop.minor;
			shared_memory_per_block = prop.sharedMemPerBlock;
		}

		virtual ~Device() noexcept = default;

	public:
		size_t get_max_memory_capacity() const {
			size_t free, total;
			gpu_check(cudaSetDevice(device_id));
			gpu_check(cudaMemGetInfo(&free, &total));
			return total;
		}

		size_t get_avail_memory_capacity() const {
			size_t free, total;
			gpu_check(cudaSetDevice(device_id));
			gpu_check(cudaMemGetInfo(&free, &total));
			return free;
		}
	};

	struct DeviceContext: public Device {
	public:
		static constexpr size_t MAX_NUM_STREAM = 8;

	private:
		std::array<cudaStream_t, MAX_NUM_STREAM> stream_array;

		cublasHandle_t 						cublas_handle;

		std::unique_ptr<DeviceMemoryPool> 	pool_ptr;

	public:
		DeviceContext(int device_id): stream_array{nullptr}, cublas_handle(nullptr), Device(device_id) {
			gpu_check(cudaSetDevice(device_id));
			for (size_t i = 0; i < MAX_NUM_STREAM; ++i)  {
				gpu_check(cudaStreamCreateWithFlags(&stream_array[i], cudaStreamNonBlocking));
			}
			gpu_check(cublasCreate_v2(&cublas_handle));
			gpu_check(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

			pool_ptr = std::make_unique<DefaultDeviceMemoryPool>(device_id);
		}

		~DeviceContext() noexcept override {
			for (cudaStream_t stream: stream_array) {
				gpu_check(cudaStreamDestroy(stream));
			}
			gpu_check(cublasDestroy_v2(cublas_handle));
		}

	public:
		cudaStream_t get_stream(size_t idx = 0) { return stream_array[idx]; }

		cublasHandle_t get_cublas_handle() const { return cublas_handle; }

		DeviceMemoryPool *get_memory_pool() const { return pool_ptr.get(); }
	};
} // namespace spy::gpu
