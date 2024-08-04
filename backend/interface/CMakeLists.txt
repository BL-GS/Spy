cmake_minimum_required(VERSION 3.20)
project(SpyBackendInterface)

# Library
find_package(magic_enum CONFIG REQUIRED)

# Project setup
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} INTERFACE
    # Internal libraries
    SpyUtil
    SpyOrchestration
    # External libraries
    magic_enum::magic_enum
)