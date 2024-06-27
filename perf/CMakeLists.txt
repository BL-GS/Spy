CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
PROJECT(SpyPerf)

# Library
FIND_PACKAGE(spdlog		CONFIG REQUIRED)

# Project construction
FILE(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/*.cpp)

# Project setup
ADD_LIBRARY(${PROJECT_NAME} STATIC ${source_files})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} 
    # External libraries
    spdlog::spdlog_header_only
)