cmake_minimum_required(VERSION 3.20)

project(SpyBackendHost)

# Libraries --------------------------------------------------------------
find_package(magic_enum CONFIG REQUIRED)
find_package(unofficial-concurrentqueue CONFIG REQUIRED)

add_definitions(-DUNOFFICIAL_CONCURRENTQUEUE)

file(GLOB_RECURSE backend_files ${PROJECT_SOURCE_DIR}/backend/*.cpp)
file(GLOB_RECURSE operator_files ${PROJECT_SOURCE_DIR}/operator/*.cpp)

# Main project ----------------------------------------------------------
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
    SpyPerf
    SpyBackendInterface
    # External library
    magic_enum::magic_enum
    unofficial::concurrentqueue::concurrentqueue
)