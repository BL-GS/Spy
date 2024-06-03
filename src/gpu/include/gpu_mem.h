#pragma once

#include <cstddef>
#include <sys/socket.h>

#include "gpu_util.h"
#include "util/shell/logger.h"

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

    template<class T = uint8_t>
    class DeviceUniquePointer {
    private:
        DeviceMemoryPool *  pool_ptr_;
        T *                 mem_ptr_;
        size_t              mem_size_;

    public:
        DeviceUniquePointer(DeviceMemoryPool *pool_ptr): pool_ptr_(pool_ptr), mem_ptr_(nullptr), mem_size_(0) {}

        DeviceUniquePointer(DeviceMemoryPool *pool_ptr, size_t num): pool_ptr_(pool_ptr), mem_ptr_(nullptr), mem_size_(num * sizeof(T)) {
            mem_ptr_ = static_cast<T *>(pool_ptr->allocate(mem_size_));
        }

        DeviceUniquePointer(DeviceUniquePointer &&other) noexcept : pool_ptr_(other.pool_ptr_), mem_ptr_(other.mem_ptr_), mem_size_(other.mem_size_) {
            other.pool_ptr_ = nullptr;
            other.mem_ptr_  = nullptr;
            other.mem_size_ = 0;
        }

        ~DeviceUniquePointer() { 
            deallocate();
        }

    public:
        void allocate(size_t num) {
            spy_assert(mem_ptr_ == nullptr, "Cannot overwrite a DeviceUniquePointer");
            mem_size_ = num * sizeof(T);
            mem_ptr_  = static_cast<T *>(pool_ptr_->allocate(mem_size_));
        }

        void deallocate() {
            if (mem_ptr_ != nullptr) { pool_ptr_->deallocate(mem_ptr_, mem_size_); }
        }

    public:
        T *get()      const { return mem_ptr_; }

        size_t size() const { return mem_size_; }
    };

} // namespace spy