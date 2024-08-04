cmake_minimum_required(VERSION 3.20)
project(SpyBackend)

add_subdirectory(${PROJECT_SOURCE_DIR}/interface)
add_subdirectory(${PROJECT_SOURCE_DIR}/distributor)
add_subdirectory(${PROJECT_SOURCE_DIR}/host)

set(spy_backend_deps SpyBackendInterface SpyBackendDistributor SpyBackendHost)

option(SPY_BACKEND_CUDA "Enable backend: GPU"  OFF)
IF (${SPY_BACKEND_CUDA})
	add_definitions(-DSPY_BACKEND_CUDA)
	add_subdirectory(${PROJECT_SOURCE_DIR}/cuda)
    set(spy_backend_deps ${spy_backend_deps} SpyBackendCUDA)
ENDIF ()

# Project setup
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} INTERFACE
    # Internal libraries
    SpyUtil
    SpyPerf
    ${spy_backend_deps}
)