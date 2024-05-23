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
#include <cuda_runtime.h>
#include <cublas.h>
#include <exception>
#include <source_location>
#include <spdlog/spdlog.h>

#include "util/shell/logger.h"

namespace spy::gpu {

	static const char * cublas_get_error_str(const cublasStatus_t err) {
		switch (err) {
			case CUBLAS_STATUS_SUCCESS: 			return "CUBLAS_STATUS_SUCCESS";
			case CUBLAS_STATUS_NOT_INITIALIZED: 	return "CUBLAS_STATUS_NOT_INITIALIZED";
			case CUBLAS_STATUS_ALLOC_FAILED: 		return "CUBLAS_STATUS_ALLOC_FAILED";
			case CUBLAS_STATUS_INVALID_VALUE: 		return "CUBLAS_STATUS_INVALID_VALUE";
			case CUBLAS_STATUS_ARCH_MISMATCH: 		return "CUBLAS_STATUS_ARCH_MISMATCH";
			case CUBLAS_STATUS_MAPPING_ERROR: 		return "CUBLAS_STATUS_MAPPING_ERROR";
			case CUBLAS_STATUS_EXECUTION_FAILED: 	return "CUBLAS_STATUS_EXECUTION_FAILED";
			case CUBLAS_STATUS_INTERNAL_ERROR: 		return "CUBLAS_STATUS_INTERNAL_ERROR";
			case CUBLAS_STATUS_NOT_SUPPORTED: 		return "CUBLAS_STATUS_NOT_SUPPORTED";
			default: 								return "unknown error";
		}
	}

	inline void gpu_check(cudaError_t err, std::source_location loc = std::source_location::current()) {
		if (err != cudaSuccess) [[unlikely]] {
			spdlog::error("Failed cuda check: {}:{}:{} - {}", loc.file_name(), loc.line(), loc.column(), loc.function_name());
			spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
			std::terminate();			
		}
	}

	inline void gpu_check(CUresult err, std::source_location loc = std::source_location::current()) {
		if (err != CUDA_SUCCESS) [[unlikely]] {
			const char *err_name   = "unknown";
			const char *err_string = "unknown";
			cuGetErrorName(err, &err_name);
			cuGetErrorString(err, &err_string);

			spdlog::error("Failed cu check: {}:{}:{} - {}", loc.file_name(), loc.line(), loc.column(), loc.function_name());
			spdlog::error("Error {}: {}", err_name, err_string);
			std::terminate();			
		}
	}

	inline void gpu_check(cublasStatus_t err, std::source_location loc = std::source_location::current()) {
		if (err != CUBLAS_STATUS_SUCCESS)[[unlikely]] {
			spdlog::error("Failed cublas check: {}:{}:{} - {}", loc.file_name(), loc.line(), loc.column(), loc.function_name());
			spdlog::error("Error: {}", cublas_get_error_str(err));
			std::terminate();			
		}
	}

	inline void print_cuda_devices() {
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
}
