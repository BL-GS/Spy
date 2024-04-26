CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

PROJECT(spy_gpu CXX CUDA)

SET(CMAKE_CUDA_STANDARD 20)
SET(CMAKE_CXX_STANDARD 20)

SET(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-std=c++20")

IF (NOT ${spy_interface})
    MESSAGE(FATAL_ERROR "Variable spy_interface hasn't been set")
ENDIF ()

# Libraries --------------------------------------------------------------
FIND_PACKAGE(fmt	 	CONFIG REQUIRED)
FIND_PACKAGE(magic_enum CONFIG REQUIRED)

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
    # External library
    fmt::fmt-header-only
    magic_enum::magic_enum
    # Internal library
    ${spy_interface}
    ${cuda_target}
)