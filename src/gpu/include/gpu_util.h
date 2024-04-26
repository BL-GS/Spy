/*
 * @author: BL-GS 
 * @date:   24-4-25
 */

#pragma once

/*!
 * @brief This file should only be used by cuda files
 */

#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <cublas.h>

namespace spy::gpu {

	inline void print_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
		int id = -1; // in case cudaGetDevice fails
		cudaGetDevice(&id);

		fprintf(stderr, "CUDA error: %s\n", msg);
		fprintf(stderr, "  current device: %d, in function %s at %s:%d\n", id, func, file, line);
		fprintf(stderr, "  %s\n", stmt);
	}

	static const char * cublas_get_error_str(const cublasStatus_t err) {
		switch (err) {
			case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
			case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
			case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
			case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
			case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
			case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
			case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
			case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
			case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
			default: return "unknown error";
		}
	}


#define CUDA_CHECK(err)                                                                             \
    do {                                                                                            \
        const cudaError_t err_ = (err);                                                             \
        if (err_ != cudaSuccess) {                                                                  \
            print_cuda_error("Failed CUDA CHECK", __FUNCTION__, __FILE__, __LINE__,                 \
				cudaGetErrorString(err));                                                           \
            assert(false);                                                                          \
            exit(-1);                                                                               \
        }                                                                                           \
    } while (0)

#define CU_CHECK(err)                                                                               \
	do {                                                                                            \
        const CUresult err_ = (err);                                                                \
        if (err_ != CUDA_SUCCESS) {                                                                 \
			const char *err_str = nullptr;                                                          \
			cuGetErrorString(err, &err_str);                                                        \
            print_cuda_error("Failed CU CHECK", __FUNCTION__, __FILE__, __LINE__, err_str);         \
            assert(false);                                                                          \
            exit(-1);                                                                               \
        }                                                                                           \
    } while (0)


#define CUBLAS_CHECK(err)                                                                           \
	do {                                                                                            \
        const cublasStatus_t err_ = (err);                                                          \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                        \
			const char *err_str = nullptr;                                                          \
            print_cuda_error("Failed CU CHECK", __FUNCTION__, __FILE__, __LINE__,                   \
				cublas_get_error_str(err));                                                         \
            assert(false);                                                                          \
            exit(-1);                                                                               \
        }                                                                                           \
    } while (0)


}
