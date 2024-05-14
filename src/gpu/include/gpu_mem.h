#pragma once

#include <cstddef>

#include "gpu_util.h"

namespace spy::gpu {

    
	class DeviceMemoryPool {
    public:
		virtual void *allocate(size_t size) = 0;
		virtual void  deallocate(void *ptr, size_t size) = 0;
	};

    class DefaultDeviceMemoryPool: public DeviceMemoryPool {
    public:
        static constexpr size_t MAX_NUM_BUFFER = 256;

    private:
        int device_id_;

    public:
        DefaultDeviceMemoryPool(int device_id): device_id_(device_id) {}

    public:
        void * allocate(size_t size) override { 
            gpu_check(cudaSetDevice(device_id_));
            void *res_ptr = nullptr;
            gpu_check(cudaMalloc(&res_ptr, size)); 
            return res_ptr;
        }

        void deallocate(void *ptr, [[maybe_unused]]size_t size) override {
            gpu_check(cudaSetDevice(device_id_));
            gpu_check(cudaFree(ptr));
        }
    };




} // namespace spy