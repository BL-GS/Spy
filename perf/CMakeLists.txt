CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
PROJECT(SpyPerf)

FIND_PACKAGE(magic_enum CONFIG REQUIRED)
FIND_PACKAGE(unofficial-perfetto CONFIG REQUIRED)

# Options
OPTION(SPY_PERFETTO_TRACING "SpyPerf: enable perfetto tracing" OFF)
IF(${SPY_PERFETTO_TRACING})
    ADD_DEFINITIONS(-DSPY_PERFETTO_TRACING)
ENDIF()

# Project setup
FILE(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp)

ADD_LIBRARY(${PROJECT_NAME} STATIC ${source_files})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC
    SpyUtil
    # External libraries
    magic_enum::magic_enum
    unofficial::perfetto::perfetto
)