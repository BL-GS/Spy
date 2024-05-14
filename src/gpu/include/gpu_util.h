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
#include <exception>
#include <source_location>
#include <spdlog/spdlog.h>

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
		spdlog::error("Failed cuda check: {}:{}:{} - {}", loc.file_name(), loc.line(), loc.column(), loc.function_name());
		spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
		std::terminate();
	}

	inline void gpu_check(CUresult err, std::source_location loc = std::source_location::current()) {
		const char *err_name   = nullptr;
		const char *err_string = nullptr;
		cuGetErrorName(err, &err_name);
		cuGetErrorString(err, &err_string);

		spdlog::error("Failed cu check: {}:{}:{} - {}", loc.file_name(), loc.line(), loc.column(), loc.function_name());
		spdlog::error("Error {}: {}", err_name, err_string);
		std::terminate();
	}

	inline void gpu_check(cublasStatus_t err, std::source_location loc = std::source_location::current()) {
		spdlog::error("Failed cublas check: {}:{}:{} - {}", loc.file_name(), loc.line(), loc.column(), loc.function_name());
		spdlog::error("Error: {}", cublas_get_error_str(err));
		std::terminate();
	}
}
