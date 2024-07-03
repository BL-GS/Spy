CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

PROJECT(SpyBackendHost)

# Libraries --------------------------------------------------------------
FIND_PACKAGE(magic_enum CONFIG REQUIRED)
FIND_PACKAGE(unofficial-concurrentqueue CONFIG REQUIRED)

ADD_DEFINITIONS(-DUNOFFICIAL_CONCURRENTQUEUE)

FILE(GLOB_RECURSE backend_files ${PROJECT_SOURCE_DIR}/backend/*.cpp)
FILE(GLOB_RECURSE operator_files ${PROJECT_SOURCE_DIR}/operator/*.cpp)

# Main project ----------------------------------------------------------
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
    SpyBackendInterface
    # External library
    magic_enum::magic_enum
    unofficial::concurrentqueue::concurrentqueue
)