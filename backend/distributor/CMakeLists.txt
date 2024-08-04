cmake_minimum_required(VERSION 3.20)
project(SpyBackendDistributor)

# Library
file(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Project setup
add_library(${PROJECT_NAME} STATIC ${source_files})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PUBLIC
    # Internal libraries
    SpyUtil
    SpyPerf
    SpyOrchestration
    SpyStorage
    SpyBackendInterface
    # External libraries
    magic_enum::magic_enum
)