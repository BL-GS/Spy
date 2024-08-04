cmake_minimum_required(VERSION 3.20)

set(SPY_DEFAULT_CUDA_ARCHITECTURES 75)
IF (NOT DEFINED SPY_CUDA_ARCHITECTURES)
    MESSAGE(STATUS "Not specify cuda architecture, use ${SPY_DEFAULT_CUDA_ARCHITECTURES} by default")
    set(CMAKE_CUDA_ARCHITECTURES ${SPY_DEFAULT_CUDA_ARCHITECTURES})
else()
    set(CMAKE_CUDA_ARCHITECTURES ${SPY_CUDA_ARCHITECTURES})
endif()


project(SpyBackendCUDA CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD  20)

set(CMAKE_CUDA_FLAGS            ${CMAKE_CUDA_FLAGS} "-std=c++20")

# Libraries --------------------------------------------------------------
find_package(magic_enum CONFIG REQUIRED)

find_package(CUDAToolkit)
# Project ----------------------------------------------------------------
file(GLOB_RECURSE backend_files ${PROJECT_SOURCE_DIR}/backend/*.cu)
file(GLOB_RECURSE operator_files ${PROJECT_SOURCE_DIR}/operator/*.cu)

add_library(${PROJECT_NAME} ${backend_files} ${operator_files})
target_include_directories(${PROJECT_NAME} PRIVATE
        # Parent interface
        ${spy_include_dir}
        # Internal interface
        ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    # Internal library
    SpyUtil
    SpyBackendInterface
    # External library
    magic_enum::magic_enum

    CUDA::cudart_static 
    CUDA::cublas 
    CUDA::cublasLt
    CUDA::cuda_driver
    # Internal library
    ${cuda_target}
)