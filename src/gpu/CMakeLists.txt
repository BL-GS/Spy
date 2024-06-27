CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

SET(SPY_DEFAULT_CUDA_ARCHITECTURES 75)
IF (NOT DEFINED SPY_CUDA_ARCHITECTURES)
    MESSAGE(STATUS "Not specify cuda architecture, use ${SPY_DEFAULT_CUDA_ARCHITECTURES} by default")
    SET(CMAKE_CUDA_ARCHITECTURES ${SPY_DEFAULT_CUDA_ARCHITECTURES})
ELSE()
    SET(CMAKE_CUDA_ARCHITECTURES ${SPY_CUDA_ARCHITECTURES})
ENDIF()


PROJECT(spy_gpu CXX CUDA)

SET(CMAKE_CUDA_STANDARD 20)
SET(CMAKE_CXX_STANDARD  20)

SET(CMAKE_CUDA_FLAGS            ${CMAKE_CUDA_FLAGS} "-std=c++20")

# Libraries --------------------------------------------------------------
FIND_PACKAGE(magic_enum CONFIG REQUIRED)

FIND_PACKAGE(CUDAToolkit)
# Project ----------------------------------------------------------------
FILE(GLOB_RECURSE backend_files ${PROJECT_SOURCE_DIR}/backend/*.cu)
FILE(GLOB_RECURSE operator_files ${PROJECT_SOURCE_DIR}/operator/*.cu)

ADD_LIBRARY(${PROJECT_NAME} ${backend_files} ${operator_files})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE
        # Parent interface
        ${spy_include_dir}
        # Internal interface
        ${PROJECT_SOURCE_DIR}/include
)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE
    # Internal library
    SpyUtil
    # External library
    magic_enum::magic_enum

    CUDA::cudart_static 
    CUDA::cublas 
    CUDA::cublasLt
    CUDA::cuda_driver
    # Internal library
    ${cuda_target}
)