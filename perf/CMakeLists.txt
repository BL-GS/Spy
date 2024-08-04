cmake_minimum_required(VERSION 3.20)
project(SpyPerf)

find_package(magic_enum CONFIG REQUIRED)
find_package(unofficial-perfetto CONFIG REQUIRED)

# Options
option(SPY_PERFETTO_TRACING "SpyPerf: enable perfetto tracing" OFF)
if(${SPY_PERFETTO_TRACING})
    add_definitions(-DSPY_PERFETTO_TRACING)
endif()

# Project setup
file(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(${PROJECT_NAME} STATIC ${source_files})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PUBLIC
    SpyUtil
    # External libraries
    magic_enum::magic_enum
    unofficial::perfetto::perfetto
)